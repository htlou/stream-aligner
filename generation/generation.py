
from __future__ import annotations
# --- code for answer generation of two models 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'

import argparse
import json
from transformers import AutoTokenizer
import transformers
import torch
from tqdm import tqdm

from models.constants import PROMPT_INPUT, PROMPT_INPUT_LLAMA3, PROMPT_INPUT_GEMMA, PROMPT_INPUT_LLAMA2
from models import load_pretrained_models
from models.utils import to_device
import sys
import ray
from vllm import LLM, SamplingParams
from pathlib import Path
from time import time
import logging
# Set up logging to file
logging.basicConfig(filename='time.log', level=logging.INFO, format='%(asctime)s: %(message)s')

# Given Question and Answer for Correction :  Q A1 -> correction (A3) and safe-instruction (A2)
# Input_FILE : qa1.json
# Output_FILE : correction.json

def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description='instruction and correction model answer generation',
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        help='the name or path of the first model (champion) in the arena to load from',
        required=True,
    )
    parser.add_argument(
        '--model_type',
        type=str,
        help='the type of given model', # llama3, llama2, etc.
        required=True,
    )
    parser.add_argument(
        '--file',
        type=str,
        help='the name or path of the first model (champion) in the arena to load from',
        required=True,
    )
    # Logging
    parser.add_argument(
        '--round',
        type=int,
        default=None,
        help='the round of the generation',
    )
    parser.add_argument(
        '--max_tokens',
        type=str,
        default=2048,
        help="max tokens"
    )
    return parser.parse_args()

def generate_answer_by_vllm(problems: list[str], model_name_or_path: str ,type:str, max_tokens=4096) ->list[str]: 
    ray.init()
    samplingparams = SamplingParams(
        top_k = 40,
        top_p = 0.95,
        temperature = 0.7,
        frequency_penalty = 1.2,
        max_tokens = max_tokens,
        skip_special_tokens = True,
    )
    llm = LLM(
        model= model_name_or_path,
        tokenizer= model_name_or_path,
        tokenizer_mode='auto',
        trust_remote_code= False,
        download_dir = None,
        tensor_parallel_size = 8,
        block_size = 16,
        gpu_memory_utilization= 0.95,
        max_num_seqs = 256,
    )
    # INPUT=PROMPT_CORRECTION_INPUT
    answers = []
    prompts = []
    print('=='*60)
    print(f'Generating <{type}> answers with {model_name_or_path}')
    for problem in problems:
        
        question = problem.get('prompt', problem.get('question'))
        prefix = problem.get('prefix', problem.get('answer', None))
        system_prompt = "You are a helpful assistant."
        if type == 'llama3':
            system_prompt = "You are a helpful AI assistant for solving mathematical problems. Output your reasoning process before reaching the result."
            prompt = PROMPT_INPUT_LLAMA3.format(system_prompt=system_prompt, user_prompt=question, content = prefix)
        elif type == 'gemma':
            prompt = PROMPT_INPUT_GEMMA.format(system_prompt=system_prompt, user_prompt=question, content = prefix)
        elif type == 'llama2':
            prompt = PROMPT_INPUT_LLAMA2.format(system_prompt=system_prompt, user_prompt=question, content = prefix)
        else:
            prompt = PROMPT_INPUT.format(problem['question'], problem['prefix'])
            
        prompts.append(prompt)
    outputs = llm.generate(prompts, samplingparams)
    for output in outputs:
        final_answer = output.outputs[0].text
        answers.append(final_answer)

    return answers

def main()->None:
    args = parse_arguments()
    print(args)
    print(f"start generation for round {args.round}")

    round = int(args.round)
    root = args.file
    input_file_path = f'./data/{root}/r{round}_input.json'
    with open(input_file_path,encoding='utf-8') as f:
        problems=json.load(f)
    
    results = []
    answer=generate_answer_by_vllm(problems,args.model_name_or_path,args.model_type, max_tokens=int(args.max_tokens))
    for problem, answer2 in tqdm(
        zip(problems,answer),
        total=len(problems),
    ):
        results.append(
            {
                'question': problem['question'],
                'solution': problem.get('solution', None),
                'prefix': problem['prefix'],
                'last': answer2,
                'flag': problem['flag'],
            },
        )
    output_dir = Path(f"./data/{root}/r{round}_base.json")
    # for problem, answer2 in tqdm(
    #     zip(problems,answer),
    #     total=len(problems),
    # ):
    #     results.append(
    #         {
    #             'question': problem['question'],
    #             'solution': problem.get('solution', None),
    #             'prefix': problem['prefix'],
    #             'answer': answer2,
    #             'flag': problem['flag'],
    #         },
    #     )
    # output_dir = Path(f"./data/{root}/r{round}_output.json")

    with open(output_dir,mode='w',encoding='utf-8') as f:
        json.dump(results,f,indent=4,ensure_ascii=False)

if __name__=='__main__':
    start_time = time()
    main()
    end_time = time()
    elapsed_time = end_time - start_time
    logging.info(f'Generation execution time: {elapsed_time:.2f} seconds')
    