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
"""Scalable Correction dataset for supervised instruction fine-tuning2023-12-15 1:25 first commit"""
from __future__ import annotations
import json
# from datasets import load_dataset

from safe_rlhf.datasets.base import RawSample
from safe_rlhf.datasets.base import RawDataset
#CORRECTION_INSTRUCTION='Edit the following Question-Answer pair to make it more helpful and harmless: {inputline}'
# CORRECTION_SYS: str = """##Instruction: {instruction} """
CORRECTION_USER: str = """##Question: {prompt} | ##Answer_Prefix: {prefix} | ##Answer_Last_Sentence: {last} | ##Your_Revision: """

__all__ = [
    'CorrectionJSONDataset',
    'BeavertailsJSONDataset',
    'BeavertailsTestJSONDataset',
    'BeavertailsTrainJSONDataset',
]

# in /safe_rlhf/datasets/supervised.py, you can see how the PAD_token was added
class CorrectionJSONDataset(RawDataset):
    NAME: str = 'correction-plus'

    def __init__(self, path) -> None:  # noqa: ANN001
        self.path = path
        with open(self.path, encoding='utf-8') as f:
            self.data = json.load(f)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        # FIXME safe aligner
        # inputline = ' | '.join((data['question'], data['answer']))
        # input = CORRECTION_INSTRUCTION.format(inputline=inputline)
        # NOTE General aligner
        # system_prompt = CORRECTION_SYS.format(instruction = data['Instruction'])
        input = CORRECTION_USER.format(prompt = data['question'], prefix = data['prefix'], last = data['last'])
        answer = data['correction']
        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)


class BeavertailsJSONDataset(RawDataset):
    def __init__(self, path) -> None:  # noqa: ANN001
        self.path = path
        with open(self.path, encoding='utf-8') as f:
            self.data = json.load(f)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=(' '.join((data['prompt'], data['response']))),
            answer=data['response'],
        )

    def __len__(self) -> int:
        return len(self.data)


class BeavertailsTestJSONDataset(BeavertailsJSONDataset):
    NAME: str = 'beavertails-json/test'


class BeavertailsTrainJSONDataset(BeavertailsJSONDataset):
    NAME: str = 'beavertails-json/train'
