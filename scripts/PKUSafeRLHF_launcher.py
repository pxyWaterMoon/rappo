from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def construct_perference(example, perference_type="helpful"):
    prompt = {"content": example["prompt"], "role": "user"}
    if perference_type == "helpful":
        chosen_id = example["better_response_id"]
    elif perference_type == "safe":
        chosen_id = example["safer_response_id"]
    else:
        raise ValueError("perference_type must be either 'helpful' or 'safe'")
    chosen_response = {"content": example[f"response_{chosen_id}"], "role": "assistant"}
    rehected_response = {"content": example[f"response_{1 - chosen_id}"], "role": "assistant"}
    return {"chosen": [prompt, chosen_response], "rejected": [prompt, rehected_response]}

def argument_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=str, default="rappo", choices=["dpo", "rappo", "cpo", "simpo", "ipo", "kto", "orpo", "r_dpo"])
    parser.add_argument("--dataset_path", type=str, default="../dataset/PKU-SafeRLHF")
    parser.add_argument("--model_name_or_path", type=str, default="../model/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--perference_type", type=str, default="helpful", choices=["helpful", "safe"])
    parser.add_argument("--output_dir", type=str, default="./output/Qwen2-0.5B-rappo")
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser

def main(args):
    # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, device_map="auto", temperature=args.temperature)
    # ref_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, device_map="auto", temperature=args.temperature)

    # train_dataset = construct_perference_dataset(args.dataset_path, split="train", perference_type=args.perference_type)
    train_dataset = load_dataset(args.dataset_path, split="train")
    train_dataset = train_dataset.map(
        construct_perference,
        fn_kwargs={"perference_type": args.perference_type},
        remove_columns=[col for col in train_dataset.column_names if col not in ["chosen", "rejected"]],
        desc="Constructing preference dataset",
    )
    # print("[DEBUG] train_dataset columns:", train_dataset.column_names)
    # print("[DEBUG] train_dataset example:", train_dataset[0])
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.trainer == "rappo":
        from rappo.alg.rappo_trainer import RappoTrainer
        from trl import DPOConfig
        training_args = DPOConfig(
            output_dir=args.output_dir,
            save_strategy="epoch",
            save_total_limit=1,
            save_safetensors=True,
            per_device_train_batch_size=16,
        )
        trainer = RappoTrainer(model=args.model_name_or_path, ref_model=args.model_name_or_path, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
    elif args.trainer == "dpo":
        from trl import DPOTrainer, DPOConfig
        training_args = DPOConfig(
            output_dir=args.output_dir,
            save_strategy="epoch",
            save_total_limit=1,
            save_safetensors=True,
            per_device_train_batch_size=16,
        )
        trainer = DPOTrainer(model=args.model_name_or_path, ref_model=args.model_name_or_path, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
    elif args.trainer == "cpo":
        from trl import CPOTrainer, CPOConfig
        training_args = CPOConfig(
            output_dir=args.output_dir,
            save_strategy="epoch",
            save_total_limit=1,
            save_safetensors=True,
            per_device_train_batch_size=16,
        )
        trainer = CPOTrainer(model=args.model_name_or_path, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
    elif args.trainer == "simpo":
        from trl import CPOTrainer, CPOConfig
        training_args = CPOConfig(
            output_dir=args.output_dir,
            save_strategy="epoch",
            save_total_limit=1,
            save_safetensors=True,
            per_device_train_batch_size=16,
            loss_type="simpo",
            beta=2.5,
            cpo_alpha=0.0,
            simpo_gamma=1.25
        )
        trainer = CPOTrainer(model=args.model_name_or_path, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
    elif args.trainer == "ipo":
        from trl import CPOTrainer, CPOConfig
        training_args = CPOConfig(
            output_dir=args.output_dir,
            save_strategy="epoch",
            save_total_limit=1,
            save_safetensors=True,
            per_device_train_batch_size=16,
            loss_type="ipo",
            beta=0.5
        )
        trainer = CPOTrainer(model=args.model_name_or_path, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
    elif args.trainer == "kto":
        from trl import KTOTrainer, KTOConfig
        training_args = KTOConfig(
            output_dir=args.output_dir,
            save_strategy="epoch",
            save_total_limit=1,
            save_safetensors=True,
            per_device_train_batch_size=16,
        )
        trainer = KTOTrainer(model=args.model_name_or_path, ref_model=args.model_name_or_path, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
    elif args.trainer == "orpo":
        from trl import ORPOTrainer, ORPOConfig
        training_args = ORPOConfig(
            output_dir=args.output_dir,
            save_strategy="epoch",
            save_total_limit=1,
            save_safetensors=True,
            per_device_train_batch_size=16,
        )
        train_dataset = train_dataset.filter(lambda x: x["chosen"] != x["rejected"] and len(x["chosen"]) > 0 and len(x["rejected"]) > 0)
        trainer = ORPOTrainer(model=args.model_name_or_path, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
    elif args.trainer == "r_dpo":
        from trl import DPOTrainer, DPOConfig
        training_args = DPOConfig(
            output_dir=args.output_dir,
            save_strategy="epoch",
            save_total_limit=1,
            save_safetensors=True,
            per_device_train_batch_size=16,
            r_dpo_alpha=0.1,
        )
        trainer = DPOTrainer(model=args.model_name_or_path, ref_model=args.model_name_or_path, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
    else:
        raise ValueError("trainer must be either 'dpo', 'rappo', 'cpo', 'simpo', 'ipo', 'kto', 'orpo', 'r_dpo'")
    # trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
    trainer.train()

if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    main(args)