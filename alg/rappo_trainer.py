from trl.trainer.dpo_trainer import DPOTrainer
from typing import Any, Callable, Literal, Optional, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
)
import torch
import torch.nn as nn
import torch.nn.functional as F

class RappoTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.remove_num = 1
    
    def get_batch_loss_metrics(
        self,
        model: Union[PreTrainedModel, nn.Module],
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        if self.args.use_liger_loss:
            model_output = self._compute_loss_liger(model, batch)
            losses = model_output["loss"]
            chosen_rewards = model_output["chosen_rewards"]
            rejected_rewards = model_output["rejected_rewards"]
        else:
            model_output = self.concatenated_forward(model, batch)

            # if ref_chosen_logps and ref_rejected_logps in batch use them, otherwise use the reference model
            if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
                ref_chosen_logps = batch["ref_chosen_logps"]
                ref_rejected_logps = batch["ref_rejected_logps"]
            else:
                ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)
            # print(f"[DEBUG] rank: {self.accelerator.process_index} ref_chosen_logps: {ref_chosen_logps} ref_rejected_logps: {ref_rejected_logps}")
            reference_ratio = ref_chosen_logps - ref_rejected_logps
            condition = (torch.exp(reference_ratio) < 1.0)  # 筛选符合条件的样本
            filter_removing_indicies = torch.nonzero(condition).squeeze(-1)
            print(f"[DEBUG] rank: {self.accelerator.process_index} filter_removing_indicies: {filter_removing_indicies}")


            # Initialize combined losses
            losses = 0
            chosen_rewards = 0
            rejected_rewards = 0

            # Compute losses for each loss type
            for idx, loss_type in enumerate(self.loss_type):
                # Compute individual loss using standard DPO loss function
                _losses, _chosen_rewards, _rejected_rewards = self.dpo_loss(
                    model_output["chosen_logps"],
                    model_output["rejected_logps"],
                    ref_chosen_logps,
                    ref_rejected_logps,
                    loss_type,
                    model_output,
                )
                print(f"[DEBUG] rank: {self.accelerator.process_index} _losses: {_losses}")
                # 在 global batch 中删除 self.remove_name 个样本
                removed_tag = torch.ones_like(_losses, device=_losses.device)
                global_removed_num = 0
                while global_removed_num < self.remove_num:
                    # 找到 filter_removing_indicies 对应 _losses 中最大的 loss，并将其对应的 removed_tag 置为 0
                    if filter_removing_indicies.numel() > 0:
                        local_max_removing_loss = _losses[filter_removing_indicies].max().item()
                        local_max_removing_loss_idx = filter_removing_indicies[torch.argmax(_losses[filter_removing_indicies])].item()
                    else:
                        local_max_removing_loss = 0.0
                        local_max_removing_loss_idx = -1
                    print(f"[DEBUG] rank: {self.accelerator.process_index} local_max_removing_loss: {local_max_removing_loss}")
                    global_max_removing_loss = self.accelerator.gather_for_metrics(torch.tensor(local_max_removing_loss, device=_losses.device)).max().item()
                    remove_signal = 0
                    if local_max_removing_loss == global_max_removing_loss and local_max_removing_loss_idx != -1:
                        print(f"[DEBUG] rank: {self.accelerator.process_index} max_removing_index: {local_max_removing_loss_idx}")
                        remove_signal = (1 << self.accelerator.process_index)
                    global_remove_signal = self.accelerator.gather_for_metrics(torch.tensor(remove_signal, device=_losses.device)).sum().item()
                    print(f"[DEBUG] rank: {self.accelerator.process_index} global_remove_signal: {global_remove_signal}")
                    for i in range(self.accelerator.process_index):
                        global_removed_num = global_removed_num + ((global_remove_signal >> i) & 1)
                    print(f"[DEBUG] rank: {self.accelerator.process_index} pre global_removed_num: {global_removed_num}")
                    if remove_signal != 0 and global_removed_num < self.remove_num:                        
                        # 当前这个 rank 的样本被删除
                        removed_tag[local_max_removing_loss_idx] = 0
                        # 把 max_removing_index 从 filter_removing_indicies 中删除
                        filter_removing_indicies = filter_removing_indicies[torch.nonzero(filter_removing_indicies != local_max_removing_loss_idx).squeeze(-1)]
                        print(f"[DEBUG] rank: {self.accelerator.process_index} after removing filter_removing_indicies: {filter_removing_indicies}")
                        global_removed_num += 1
                    for i in range(self.accelerator.process_index + 1, self.accelerator.num_processes):
                        if global_removed_num < self.remove_num:
                            global_removed_num = global_removed_num + ((global_remove_signal >> i) & 1)
                    print(f"[DEBUG] rank: {self.accelerator.process_index} global_removed_num: {global_removed_num}")
                    

                # Add weighted contributions
                print(f"[DEBUG] rank: {self.accelerator.process_index} removed_tag: {removed_tag}")
                weight = self.loss_weights[idx] if self.loss_weights else 1.0
                losses = losses + _losses * weight * removed_tag
                chosen_rewards = chosen_rewards + _chosen_rewards * weight
                rejected_rewards = rejected_rewards + _rejected_rewards * weight

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = losses + self.args.rpo_alpha * model_output["nll_loss"]  # RPO loss from V3 of the paper

        if self.use_weighting:
            losses = losses * model_output["policy_weights"]

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        metrics[f"{prefix}rewards/margins"] = (
            self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item()
        )
        metrics[f"{prefix}logps/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["chosen_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logps/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["rejected_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["mean_chosen_logits"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["mean_rejected_logits"]).detach().mean().item()
        )
        if self.args.rpo_alpha is not None or "sft" in self.loss_type:
            metrics[f"{prefix}nll_loss"] = (
                self.accelerator.gather_for_metrics(model_output["nll_loss"]).detach().mean().item()
            )
        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = (
                self.accelerator.gather_for_metrics(model_output["aux_loss"]).detach().mean().item()
            )

        return losses.mean(), metrics