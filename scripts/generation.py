# from  Copyright 2023 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import json
import os
import numpy as np
import torch
from tqdm.auto import tqdm
import sys
from transformers import pipeline
from transformers import StoppingCriteria
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset


# PROMPT_INPUT = '<s>[INST] <<SYS>>\n You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.\n<</SYS>>\n{input}\n[/INST]'



def argument_parser():
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate models with gpt4',
    )
    parser.add_argument(
        '--num_per_prompt',
        type=int,
        default=1,
        help='Number of answers generated per prompt.',
    )
    # Model
    model_parser = parser.add_argument_group('model')
    model_parser.add_argument(
        '--model_name_or_path',
        type=str,
        help='the name or path of the first model (champion) in the arena to load from',
        required=True,
    )
    model_parser.add_argument(
        '--tokenizer_name_or_path',
        type=str,
        help='the name or path of the first model (champion) in the arena to load from',
        default=None,
    )
    model_parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='The maximum sequence length of the model.',
    )
    # Dataset
    dataset_parser = parser.add_argument_group('dataset')
    dataset_parser.add_argument(
        '--dataset_path',
        type=str,
        help='the path of the dataset to load',
        required=True,
    )
    # Logging
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Where to store the eval output.',
    )

    return parser.parse_args()


class KeywordStoppingCriteria(StoppingCriteria):
    def __init__(self, keyword_ids_list: list):
        self.keyword_ids_list = keyword_ids_list

    def __call__(
        self, input_ids: torch.LongTensor, scores=torch.FloatTensor, **kwargs
    ) -> bool:
        for keyword_ids in self.keyword_ids_list:
            if (
                len(input_ids[0]) >= len(keyword_ids)
                and input_ids[0][-len(keyword_ids):].tolist() == keyword_ids
            ):
                return True
        return False


def remove_keywords(answer, stop_words_list):
    for stop_words in stop_words_list:
        answer = answer.split(stop_words)[0]
    return answer

def generate_answer(dataset, model_name_or_path: str, tokenizer_name_or_path: str=None, max_length=512, num_per_prompt=1):
    pipe = pipeline(
        'text-generation',
        model=model_name_or_path,
        tokenizer=tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path,
        trust_remote_code=True,
        device_map='auto' if torch.cuda.is_available() else None,
        max_length=max_length,
    )
    
    answers = []
    
    print(f'Generating answers with {model_name_or_path}')
    # for data in tqdm(dataset):
    #     # prompt = [{"content": data['prompt'], "role": "user"}]
    #     # prompt = PROMPT_INPUT.format(input=data['prompt'])
    #     result = pipe(data["prompt"], do_sample=True, temperature=1.0, num_return_sequences=num_per_prompt)
    #     for i in range(num_per_prompt):
    #         answers.append(result[i]['generated_text'])
    # results = 
    for result in tqdm(pipe(KeyDataset(dataset, "prompt")[:10], do_sample=True, temperature=1.0, num_return_sequences=num_per_prompt, batch_size=16)):
        for i in range(num_per_prompt):
            answers.append(result[i]['generated_text'])
            print(result[i]['generated_text'])
    return answers

def duplicate_removal(dataset):
    key = "prompt"
    seen = set()
    def check_duplicate(example):
        value = example[key]
        if value in seen:
            return False
        seen.add(value)
        return True
    return dataset.filter(check_duplicate)

def dataset_format(dataset):
    def format_example(example):
        example["prompt"] = [{"content": example['prompt'], "role": "user"}]
        return example
    return dataset.map(format_example)

def main() -> None:
    """The main function."""
    args = argument_parser()
    num_per_prompt = args.num_per_prompt
    model_name = os.path.basename(os.path.normpath(args.model_name_or_path))

    dataset = load_dataset(args.dataset_path, split="test")
    dataset = duplicate_removal(dataset)
    dataset = dataset_format(dataset)
    
    answers = generate_answer(dataset, args.model_name_or_path, args.tokenizer_name_or_path, args.max_length, num_per_prompt)
    # print(an swers)
    if args.output_path is not None:
        output_dir = os.path.dirname(args.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(args.output_path, mode='w', encoding='utf-8') as f:
            json.dump(answers, f, indent=4, ensure_ascii=False)
    
if __name__ == '__main__':
    sys.exit(main())