import os
import random
# from dataclasses import dataclass
import json
import numpy as np
import torch
import torch.nn as nn
# from datasets import load_dataset
from transformers import DecisionTransformerConfig, DecisionTransformerModel, Trainer, TrainingArguments, DataCollator
from torch.utils.data import Dataset, DataLoader
from utility import LossPlotCallback
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"#"3,5,7,9"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# os.environ["WANDB_DISABLED"] = "true" # we diable weights and biases logging for this tutorial

# train_data_path = './train_data_strategy_35+1_random_suffix'
# # eval_data_path = './Onefunction_PI_eval'
# output_dir = "output_35+1_random_suffix_retest_unpretrain/"
pretrain = False
pretrained_model = '/scratch/uceehj1/dt/output_35+1_random__/checkpoint-13500'
# function_path = 'Hundredfunction_Hundredmean'
# penalty = 0.02

train_data_path = sys.argv[1]
output_dir = sys.argv[2]
penalty = float(sys.argv[3])
function_path = sys.argv[4]

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
        self.act_dim = 1001 # len(dataset[0]['x_obs'][0])
        self.state_dim = 1 # len(dataset[0]['y_obs'][0])
        self.episode_len = 51
        self.mode = mode

    def __call__(self, examples):
        temp = torch.zeros((0, self.episode_len, 1))
        time_temp = torch.zeros((0, self.episode_len)).long()
        a_temp = torch.zeros((0, self.episode_len, self.act_dim))

        s, a, mu_o, sigma_o, r, d, rtg, timesteps, mask = temp, a_temp, a_temp, a_temp, temp, temp, temp, time_temp, time_temp

        for example in examples:
            if self.mode == 'train':
                si = random.randint(2, len(example['RTG']))
            else:
                si = self.episode_len

            actions = torch.cat((torch.zeros((self.episode_len - si), self.act_dim), torch.tensor(example['A_ind_onehot'][:si]).float()), dim=0)
            # actions_output = torch.cat((torch.zeros((self.episode_len - si), self.act_dim), torch.tensor(example['A_prob'][:si]).float()), dim=0)
            mus_output = torch.cat(
                (torch.zeros((self.episode_len - si), self.act_dim), torch.tensor(example['A_mu'][:si]).float()),
                dim=0)
            sigmas_output = torch.cat(
                (torch.zeros((self.episode_len - si), self.act_dim), torch.tensor(example['A_sigma'][:si]).float()),
                dim=0)
            states = torch.cat((torch.zeros((self.episode_len - si)), torch.tensor(example['S'][:si]).float()), dim=0)
            rewards = torch.cat((torch.zeros((self.episode_len - si)), torch.tensor(example['R'][:si]).float()), dim=0)
            return_to_go = torch.cat((torch.zeros((self.episode_len - si)), torch.tensor(example['RTG'][:si]).float()), dim=0)

            # rewards = torch.zeros((self.episode_len)).float()
            # return_to_gos = torch.zeros((self.episode_len)).float()
            timestep = torch.cat((torch.zeros((self.episode_len - si)), torch.arange(0, si)), dim=0).long()
            mask_ = torch.cat((torch.zeros((self.episode_len - si)), torch.ones((si)).float()), dim=0)

            s = torch.cat((s, states.unsqueeze(0).unsqueeze(-1)), dim=0)
            a = torch.cat((a, actions.unsqueeze(0)), dim=0)
            # a_o = torch.cat((a_o, actions_output.unsqueeze(0)), dim=0)
            mu_o = torch.cat((mu_o, mus_output.unsqueeze(0)), dim=0)
            sigma_o = torch.cat((sigma_o, sigmas_output.unsqueeze(0)), dim=0)
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
            # "actions_output": a_o,
            "mus_output": mu_o,
            "sigmas_output": sigma_o,

        }



class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)
        self.predict_mu = nn.Sequential(
            torch.nn.Linear(config.act_dim, config.act_dim),
            nn.ReLU(),
            nn.Linear(config.act_dim, config.act_dim))
        self.predict_sigma = nn.Sequential(
            torch.nn.Linear(config.act_dim, config.act_dim),
            nn.ReLU(),
            nn.Linear(config.act_dim, config.act_dim))


    def forward(self, **kwargs):
        # action_targets = kwargs.pop('actions_output', None)
        mu_targets = kwargs.pop('mus_output', None)
        sigma_targets = kwargs.pop('sigmas_output', None)
        output = super().forward(**kwargs)
        # add the DT loss
        action_preds = output[1]
        mu_preds = self.predict_mu(action_preds)
        sigma_preds = self.predict_sigma(action_preds)

        attention_mask = kwargs["attention_mask"]
        ones_indices = (attention_mask == 1.0).nonzero(as_tuple=True)

        for i in range(attention_mask.size(0)):
            current_indices = ones_indices[1][ones_indices[0] == i][:1]
            attention_mask[i, current_indices] = 0.0
        act_dim = action_preds.shape[2]

        mu_preds = mu_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        mu_targets = mu_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        sigma_preds = sigma_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        sigma_targets = sigma_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = torch.mean((mu_preds - mu_targets) ** 2) + torch.mean((sigma_preds - sigma_targets) ** 2)

        return {"loss": loss}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)

def compute_metrics(preds, labels):
    mse = np.mean(np.square(preds - labels))
    return {"mse": mse}



dataset = MyDataset(train_data_path)
# eval_dataset = MyDataset(eval_data_path)
collator = MyDataCollator(dataset, 'train')

config = DecisionTransformerConfig(state_dim=collator.state_dim, act_dim=collator.act_dim, hidden_size = collator.act_dim, action_tanh=False, max_length=52)
if pretrain == True:
    model = TrainableDT(config).from_pretrained(pretrained_model)
else:
    model = TrainableDT(config)

# 查看网络的所有参数和它们的requires_grad属性
for name, param in model.named_parameters():
    print(name, param.requires_grad)

loss_plot_callback = LossPlotCallback()

training_args = TrainingArguments(
    output_dir=output_dir,
    remove_unused_columns=False,
    num_train_epochs=15000,
    per_device_train_batch_size=128,
    learning_rate=2e-4,
    weight_decay=1e-4,
    warmup_ratio=0.1,
    optim="adamw_torch",
    max_grad_norm=0.25,
    logging_steps=50,
    report_to='tensorboard',
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

trainer.train()
# subprocess.call(['python', 'for_compare_test_4.py', output_dir, train_data_path, str(penalty), function_path])
