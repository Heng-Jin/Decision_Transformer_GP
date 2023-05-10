import os
import random
from dataclasses import dataclass
import json
import numpy as np
import torch
from datasets import load_dataset
from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments, DataCollator
from torch.utils.data import Dataset, DataLoader


os.environ["WANDB_DISABLED"] = "true" # we diable weights and biases logging for this tutorial
# dataset = load_dataset("edbeeching/decision_transformer_gym_replay", "halfcheetah-expert-v2")


class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data = []  
        for file_name in os.listdir(data_path):
            if not file_name.endswith('.json'):
                continue
            with open(os.path.join(data_path, file_name), 'r') as f:
                data = json.load(f)
                # x_obs = data['PI']['x_obs']
                # y_obs = data['PI']['y_obs']
                # self.data.append({'x_obs': x_obs, 'y_obs': y_obs})
                self.data.append(dict(data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class MyDataCollator:
    def __init__(self, dataset, mode):
        self.act_dim = 1 # len(dataset[0]['x_obs'][0])
        self.state_dim = 1 # len(dataset[0]['y_obs'][0])
        self.episode_len = len(dataset[0]['A'])
        self.mode = mode

    def __call__(self, examples):
        temp = torch.zeros((0, self.episode_len, 1))
        time_temp = torch.zeros((0, self.episode_len)).long()

        s, a, r, d, rtg, timesteps, mask = temp, temp, temp, temp, temp, time_temp, time_temp

        for example in examples:
            if self.mode == 'train':
                si = random.randint(0, self.episode_len)
            else:
                si = self.episode_len

            actions = torch.cat((torch.zeros((self.episode_len - si)), torch.tensor(example['A'][:si]).float()), dim=0)
            states = torch.cat((torch.zeros((self.episode_len - si)), torch.tensor(example['S'][:si]).float()), dim=0)
            rewards = torch.cat((torch.zeros((self.episode_len - si)), torch.tensor(example['R'][:si]).float()), dim=0)
            return_to_go = torch.cat((torch.zeros((self.episode_len - si)), torch.tensor(example['RTG'][:si]).float()), dim=0)
            # rewards = torch.zeros((self.episode_len)).float()
            # return_to_gos = torch.zeros((self.episode_len)).float()
            timestep = torch.cat((torch.zeros((self.episode_len - si)), torch.arange(0, si)), dim=0).long()
            mask_ = torch.cat((torch.zeros((self.episode_len - si)), torch.ones((si)).float()), dim=0)

            s = torch.cat((s, actions.unsqueeze(0).unsqueeze(-1)), dim=0)
            a = torch.cat((a, states.unsqueeze(0).unsqueeze(-1)), dim=0)
            r = torch.cat((r, rewards.unsqueeze(0).unsqueeze(-1)), dim=0)
            rtg = torch.cat((rtg, return_to_go.unsqueeze(0).unsqueeze(-1)), dim=0)

            # to do: change the rtg he r
            # rtg = torch.cat((rtg, return_to_gos.unsqueeze(0).unsqueeze(-1)), dim=0)
            timesteps = torch.cat((timesteps, timestep.unsqueeze(0)), dim=0)
            mask = torch.cat((mask, mask_.unsqueeze(0)), dim=0)

        return {
            "states": s,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "attention_mask": mask,
        }



class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        # add the DT loss
        action_preds = output[1]
        action_targets = kwargs["actions"]
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = torch.mean((action_preds - action_targets) ** 2)

        return {"loss": loss}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)

def compute_metrics(preds, labels):
    mse = np.mean(np.square(preds - labels))
    return {"mse": mse}

# collator = DecisionTransformerGymDataCollator(dataset["train"])
train_data_path = './train_data_strategy_1'
# eval_data_path = './Onefunction_PI_eval'

dataset = MyDataset(train_data_path)
# eval_dataset = MyDataset(eval_data_path)
collator = MyDataCollator(dataset, 'train')

config = DecisionTransformerConfig(state_dim=collator.state_dim, act_dim=collator.act_dim)
model = TrainableDT(config)


training_args = TrainingArguments(
    output_dir="output/",
    remove_unused_columns=False,
    num_train_epochs=200,
    per_device_train_batch_size=2,
    learning_rate=1e-4,
    weight_decay=1e-4,
    warmup_ratio=0.1,
    optim="adamw_torch",
    max_grad_norm=0.25,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

trainer.train()
