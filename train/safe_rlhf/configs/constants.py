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
    'PROMPT_DICT',
    'ADAM_BETAS',
]


IGNORE_INDEX: int = -100
DEFAULT_BOS_TOKEN: str = '<s>'
DEFAULT_EOS_TOKEN: str = '</s>'
DEFAULT_PAD_TOKEN: str = '<pad>'
DEFAULT_UNK_TOKEN: str = '<unk>'


# PROMPT_BEGIN: str = '<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n'
# PROMPT_USER: str = '{input} '
# PROMPT_ASSISTANT: str = '[/INST]'

# PROMPT_BEGIN: str = '<<SYS>>\nYou are a helpful, respectful and honest assistant. According to the given question, edit the given answer to make it as helpfully as possible. You should give precise, helpful, humanized and detailed correction to th answer. Try your best to avoid decreasing the information provided by the answer.\n<</SYS>>\n\n'
# PROMPT_USER: str = '<s>[INST] {input} '
# PROMPT_ASSISTANT: str = '[/INST]'  # should not have a space at the end
# PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
# PROMPT_USER: str = '{input} '
# PROMPT_ASSISTANT: str = ''  # should not have a space at the end
PROMPT_BEGIN: str = '<start_of_turn>user\n'
PROMPT_USER: str = '{input}<end_of_turn>\n'
PROMPT_ASSISTANT: str = '<start_of_turn>model'  # should not have a space at the end
PROMPT_INPUT: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT

PROMPT_DICT: dict[str, str] = {
    'prompt_begin': PROMPT_BEGIN,
    'prompt_user': PROMPT_USER,
    'prompt_assistant': PROMPT_ASSISTANT,
    'prompt_input': PROMPT_INPUT,
}

ADAM_BETAS: tuple[float, float] = (0.9, 0.95)
