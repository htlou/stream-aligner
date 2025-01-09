# Copyright 2023 PKU-Alignment Team. All Rights Reserved.
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
"""Constant variables."""

from __future__ import annotations
import os

__all__ = [
    'IGNORE_INDEX',
    'DEFAULT_BOS_TOKEN',
    'DEFAULT_EOS_TOKEN',
    'DEFAULT_PAD_TOKEN',
    'DEFAULT_UNK_TOKEN',
    'PROMPT_BEGIN',
    'PROMPT_USER',
    'PROMPT_ASSISTANT',
    'PROMPT_INPUT',
    'PROMPT_INPUT_LLAMA3',
    'PROMPT_INPUT_GEMMA',
    'PROMPT_DICT',
    'ADAM_BETAS',
]


IGNORE_INDEX: int = -100
DEFAULT_BOS_TOKEN: str = '<s>'
DEFAULT_EOS_TOKEN: str = '</s>'
DEFAULT_PAD_TOKEN: str = '<pad>'
DEFAULT_UNK_TOKEN: str = '<unk>'



config_set = os.getenv("CONFIG_SET", "default")  

PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
PROMPT_USER: str = 'USER: {input} '
PROMPT_ASSISTANT: str = 'ASSISTANT: {answer}'  # should not have a space at the end
PROMPT_INPUT: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT

PROMPT_INPUT_LLAMA3: str = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{content}'
PROMPT_INPUT_GEMMA: str = '<start_of_turn>user\n{input}<end_of_turn>\n<start_of_turn>model{content}'
PROMPT_INPUT_LLAMA2: str = '<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]\n{content}'
PROMPT_INPUT_QWEN: str = '<|im_start|>system\n{system_prompt}<|im_end|><|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant{content}'

PROMPT_DICT: dict[str, str] = {
    'prompt_begin': PROMPT_BEGIN,
    'prompt_user': PROMPT_USER,
    'prompt_assistant': PROMPT_ASSISTANT,
    'prompt_input': PROMPT_INPUT,
}

