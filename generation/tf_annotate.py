import argparse
import json
import os
import sys
from pathlib import Path
import re

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

def is_empty_or_whitespace(s):
    return re.fullmatch(r'\s*', s) is not None

def split_first_sentence(text: str) -> (str, str):
    # Replace problematic abbreviations and numbered items with a placeholder
    text = re.sub(r'\b(i\.e\.|e\.g\.|\d\.)', lambda x: x.group().replace('.', '<DOT>'), text)
    # Split sentences on ".", "?", "!" that are not preceded by the special placeholder
    sentences = re.split(r'(?<!<DOT>)([.?!])', text)
    # Combine split elements to form complete sentences
    processed_sentences = []
    temp_sentence = ""
    for i, part in enumerate(sentences):
        if i % 2 == 0:
            # Text part
            temp_sentence = part.replace('<DOT>', '.')
        else:
            # Punctuation part
            complete_sentence = temp_sentence.strip() + part
            processed_sentences.append(complete_sentence)
            temp_sentence = ""  # Reset for next sentence
    first_sentence = processed_sentences[0] if processed_sentences else ""
    remaining_text = ' '.join(processed_sentences[1:]) if len(processed_sentences) > 1 else ""
    return first_sentence, remaining_text


def main():
    args = parse_arguments()
    round = args.round
    root = args.file
    with open(f'./data/{root}/r{round}_base.json', 'r') as f:
        data = json.load(f)

    new_input = []
    for item in data:
        if round > 8 and item['last'] == "":
            item['flag'] = True
        first_sentence, remaining_text = split_first_sentence(item['last'])
        item['last'] = first_sentence
        new_input.append(item)

    new_file = Path(f'./data/{root}/r{round}_tf.json')
    with open(new_file, 'w') as f:
        json.dump(new_input, f, indent=4)

    new_input = []
    if round !=0:
        with open(f'./data/{root}/r{round}_base.json', 'r') as f:
            data = json.load(f)
        with open(f'./data/{root}/r0_base.json', 'r') as f:
            raw_data = json.load(f)
        new_input = []
        for raw, item in zip(raw_data, data):
            new_item = {
                'question': item['question'],
                'solution': item.get('solution', None),
                'answer': raw['last'],
                'prefix': item['prefix'],
                'last': item['last'],
                'correction': item['prefix'] + item['last'],
                'flag': item['flag'],
            }
            new_input.append(new_item)
    # print(new_input)
    new_file = Path(f'./data/{root}/r{round}_output#2.json')
    with open(new_file, 'w') as f:
        json.dump(new_input, f, indent=4)


if __name__=='__main__':
    main()