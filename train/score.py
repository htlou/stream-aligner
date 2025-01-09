import torch
from safe_rlhf.models.score_model.llama.modeling_llama import LlamaForScore
from safe_rlhf.trainers.supervised_trainer import SupervisedTrainer
from safe_rlhf.datasets import PreferenceDataset, SafetyPreferenceDataset
from safe_rlhf.utils import to_device
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import json
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class SimpleQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs_1 = self.tokenizer(
            item['prompt'] + item['response_1'],
            max_length=self.max_length, truncation=True, padding='max_length',
            return_tensors='pt'
        )
        inputs_2 = self.tokenizer(
            item['prompt'] + item['response_2'],
            max_length=self.max_length, truncation=True, padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids_1': inputs_1['input_ids'].squeeze(),
            'attention_mask_1': inputs_1['attention_mask'].squeeze(),
            'input_ids_2': inputs_2['input_ids'].squeeze(),
            'attention_mask_2': inputs_2['attention_mask'].squeeze(),
            'prompt': item['prompt'],
            'response_1': item['response_1'],
            'response_2': item['response_2']
        }
    


class ModelEvaluator:
    def __init__(self, model_paths, tokenizer_path, dataset_name):
        self.model_paths = model_paths
        self.tokenizer_path = tokenizer_path
        self.dataset_name = dataset_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

    def load_model_and_tokenizer(self, model_dir, checkpoint_file=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        if checkpoint_file:
            model_path = os.path.join(model_dir, checkpoint_file)
            config_path = model_dir  # Assuming config.json is in the model_dir
        else:
            model_path = model_dir
            config_path = model_dir
        self.model = LlamaForScore.from_pretrained(config_path, state_dict=torch.load(model_path))
        self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.eval()

    def load_dataset(self):
        self.load_tokenizer()
        if self.model_paths[0].endswith('rm/'):
            eval_dataset = PreferenceDataset(
                [(self.dataset_name, {'proportion': 1.0})],
                tokenizer=self.tokenizer,
            )
        else:
            eval_dataset = SafetyPreferenceDataset(
                [(self.dataset_name, {'proportion': 1.0})],
                tokenizer=self.tokenizer,
            )
        self.dataloader = DataLoader(
            eval_dataset,
            collate_fn=eval_dataset.get_collator(),
            shuffle=True,
            batch_size=16,
        )


    def get_checkpoint_paths(self, model_dir):
        checkpoint_files = [os.path.join(model_dir, 'pytorch_model.bin')]
        for file_name in sorted(os.listdir(model_dir)):
            if file_name.startswith("pytorch_model_step_") and file_name.endswith(".bin"):
                checkpoint_files.append(os.path.join(model_dir, file_name))
        return checkpoint_files

    def evaluate(self):
        # dataloader = DataLoader(self.dataset, batch_size=8, num_workers=8)
        results = []

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Evaluating"):
                # input_ids_1 = batch['input_ids_1'].to(self.device)
                # attention_mask_1 = batch['attention_mask_1'].to(self.device)
                # input_ids_2 = batch['input_ids_2'].to(self.device)
                # attention_mask_2 = batch['attention_mask_2'].to(self.device)

                # inputs_1 = {
                #     'input_ids': input_ids_1,
                #     'attention_mask': attention_mask_1
                # }
                # inputs_2 = {
                #     'input_ids': input_ids_2,
                #     'attention_mask': attention_mask_2
                # }

                # outputs_1 = self.model(**inputs_1)
                # outputs_2 = self.model(**inputs_2)
                # scores_1 = outputs_1.end_scores.squeeze(dim=-1)
                # scores_2 = outputs_2.end_scores.squeeze(dim=-1)

                batch = to_device(batch, self.device)
                if self.model_paths[0].endswith('rm/'):
                    scores_1 = self.model(
                        batch['better_input_ids'],
                        attention_mask=batch['better_attention_mask'],
                    ).end_scores.squeeze(dim=-1)
                    scores_2 = self.model(
                        batch['worse_input_ids'],
                        attention_mask=batch['worse_attention_mask'],
                    ).end_scores.squeeze(dim=-1)
                else:
                    scores_1 = self.model(
                        batch['safer_input_ids'],
                        attention_mask=batch['safer_attention_mask'],
                    ).end_scores.squeeze(dim=-1)
                    scores_2 = self.model(
                        batch['unsafer_input_ids'],
                        attention_mask=batch['unsafer_attention_mask'],
                    ).end_scores.squeeze(dim=-1)


                for i in range(len(scores_1)):
                    results.append({
                        # 'prompt': batch['prompt'][i],
                        # 'response_1': batch['response_1'][i],
                        # 'response_2': batch['response_2'][i],
                        'score_1': scores_1[i].item(),
                        'score_2': scores_2[i].item()
                    })

        return results

    def save_results(self, results, model_name):
        output_file = f'results_{model_name}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

    def run(self):
        self.load_dataset()
        for model_dir in self.model_paths:
            checkpoint_files = self.get_checkpoint_paths(model_dir)
            for checkpoint_file in tqdm(checkpoint_files, desc="Loading Checkpoints"):
                checkpoint_name = os.path.basename(checkpoint_file)
                self.load_model_and_tokenizer(model_dir, checkpoint_file)
                results = self.evaluate()
                model_name = os.path.basename(model_dir)
                self.save_results(results, f"{model_name}_{checkpoint_name}")

if __name__ == "__main__":
    tokenizer_path = '/data/models/safeRLHF/beaver-7b-v3.0-reward/'
    model_paths = [
        '/data/models/safeRLHF/beaver-7b-v3.0-reward/'
    ]
    dataset_name = "beavertails2_alpaca_2w_val" 

    evaluator = ModelEvaluator(model_paths, tokenizer_path, dataset_name)
    evaluator.run()