import argparse
import json
import os
import sys
from pathlib import Path

def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description='transform gpt4 annotated to input',
    )
    parser.add_argument(
        '--round',
        type=int,
        help='the round of the generation',
        required=True,
    )
    parser.add_argument(
        '--file',
        type=str,
        help='the round of the generation',
        required=True,
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    round = args.round
    root = args.file
    with open(f'./data/{root}/r{round}_annotated.json', 'r') as f:
        data = json.load(f)

    new_input = []
    for item in data:
        # if not item['eos']:
        question = item['question']
        prefix = item['prefix']
        flag = item['flag']
        try:
            correction = item['correction']
        except:
            print(f"Error: {item}: Missing correction")
            # next item
            continue
        new_prefix = f"{prefix} {correction}"
        new_input.append({
            'question': question,
            'solution': item.get('solution', None),
            'prefix': new_prefix,
            'flag': flag,
        })

    new_input_file = Path(f'./data/{root}/r{round+1}_input.json')
    with open(new_input_file, 'w') as f:
        json.dump(new_input, f, indent=4)
    
    with open(f'./data/{root}/r0_base.json', 'r') as f:
        base_data = json.load(f)
    base_data_index = {item['question']: item for item in base_data}

    output = []
    for item in data:
        base = base_data_index.get(item['question'])
        if base:
            try:
                output.append({
                    'question': item['question'],
                    'solution': item.get('solution', None),
                    'answer': base['last'],
                    'output': f"{item['prefix']} {item['correction']}",
                    # 'output': f"{item['prefix']} {item['correction'][0]}",
                })
            except:
                print(f"Error: {item}: Missing correction")
                continue
    
    output_file = Path(f'./data/{root}/r{round}_output.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4)

if __name__=='__main__':
    main()