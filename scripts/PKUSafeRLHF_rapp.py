from datasets import load_dataset
from trl import DPOConfig
from alg.rappo_trainer import RappoTrainer
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
    parser.add_argument("--dataset_path", type=str, default="../dataset/PKU-SafeRLHF")
    parser.add_argument("--model_name_or_path", type=str, default="../model/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--perference_type", type=str, default="helpful", choices=["helpful", "safe"])
    parser.add_argument("--output_dir", type=str, default="./output/Qwen2-0.5B-rappo")
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser

def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, device_map="auto", temperature=args.temperature)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, device_map="auto")
    training_args = DPOConfig(
        output_dir=args.output_dir,
        save_steps=None,
        save_strategy="no",
        per_device_train_batch_size=16,
    )
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

    trainer = RappoTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
    # trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
    trainer.train()

if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    main(args)