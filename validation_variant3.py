import numpy as np
from transformers import DecisionTransformerModel
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import json
import sys
import random
from collections import Counter
from scipy.stats import norm

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

trajectory_path = sys.argv[1]
model_path = sys.argv[2]
penalty = float(sys.argv[3])
function_path = sys.argv[4]
# function_path = r'./Hundredfunction_Hundredmean_test'
# model_path = r"./output_35"
# trajectory_path = './train_data_strategy_35_test'
# penalty = 0.00
print(model_path)

state_dim = 1 # state size
act_dim = 1001 # action size
max_length = 50

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
#         mu_targets = kwargs.pop('mus_output', None)
#         sigma_targets = kwargs.pop('sigmas_output', None)
        output = super().forward(**kwargs)
        # add the DT loss
        action_preds = output[1]
        mu_preds = self.predict_mu(action_preds)
        sigma_preds = self.predict_sigma(action_preds)

        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        mu_preds = mu_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        sigma_preds = sigma_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

#         loss = torch.mean((mu_preds - mu_targets) ** 2) + torch.mean((sigma_preds - sigma_targets) ** 2)

        return {"sigma":sigma_preds, "mu": mu_preds}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)


def get_action(model, states, actions, rewards, returns_to_go, timesteps):
    # This implementation does not condition on past rewards
    max_length = 50
    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    # The prediction is conditioned on up to 20 previous time-steps
    states = states[:, -max_length:]
    actions = actions[:, -max_length:]
    returns_to_go = returns_to_go[:, -max_length:]
    timesteps = timesteps[:, -max_length:]

    # pad all tokens to sequence length, this is required if we process batches
    padding = max_length - states.shape[1]
    attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
    attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
    states = torch.cat([torch.zeros((1, padding, state_dim)), states], dim=1).float()
    actions = torch.cat([torch.zeros((1, padding, act_dim)), actions], dim=1).float()
    returns_to_go = torch.cat([torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
    timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1)

    # perform the prediction
    result_dict = model(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=True, )

    return result_dict

def find_nearest(array, x):
    min_diff = float('inf')
    min_index = None
    for i, a in enumerate(array):
        diff = abs(a - x)
        if diff < min_diff:
            min_diff = diff
            min_index = i
    return min_index

def one_hot_encode_max(tensor):
    max_value, max_index = torch.max(tensor, dim=0)
    one_hot = torch.zeros_like(tensor)
    one_hot[max_index] = 1
    return one_hot, max_index

def find_first_zero_index(lst):
    try:
        index = lst.index(0)
        return index
    except ValueError:
        return None

def find_max_index(lst):
    max_value = max(lst)
    max_index = lst.index(max_value)
    return max_index

def extract_num(string):
    return int(string.split('-')[-1])

def find_checkpoint_folders(folder_path):
    checkpoint_folders = []
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            if "checkpoint" in dir_name:
                checkpoint_folders.append(os.path.join(root, dir_name))
    checkpoint_folders = sorted(checkpoint_folders, key=extract_num, reverse=True)
    return checkpoint_folders

def generate_line_chart(af_res_list, dt_res_list, af_trajectory_list, dt_trajectory_list, random_trajectory_list, plot_name):
    # 统计元素个数
    counter1 = Counter(af_res_list)
    counter2 = Counter(dt_res_list)

    af_trajectory_list = [list(column) for column in zip(*af_trajectory_list)]
    af_mean_list = [sum(sublist) / len(sublist) for sublist in af_trajectory_list]
    af_var_list = [np.var(sublist) for sublist in af_trajectory_list]
    dt_trajectory_list = [list(column) for column in zip(*dt_trajectory_list)]
    dt_mean_list = [sum(sublist) / len(sublist) for sublist in dt_trajectory_list]
    dt_var_list = [np.var(sublist) for sublist in dt_trajectory_list]
    random_trajectory_list = [list(column) for column in zip(*random_trajectory_list)]
    random_mean_list = [sum(sublist) / len(sublist) for sublist in random_trajectory_list]
    random_var_list = [np.var(sublist) for sublist in random_trajectory_list]

   # 获取0-50和None的取值范围
    values = list(range(51)) + [None]

    # 统计每个取值的个数
    counts1 = [counter1[value] for value in values]
    counts2 = [counter2[value] for value in values]
    new_dt_res_list = [x for x in dt_res_list if x is not None]
    percentage = len(new_dt_res_list) * 100.0 / len(dt_res_list)

    # 生成折线图
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6))
    ax1.plot(values, counts1, marker='o', label='Acquisition function', c='r')
    ax1.plot(values, counts2, marker='o', label='Decision transformer', c='b')
    ax1.set_xlabel('No. timestep')
    ax1.set_ylabel('number of trajectories')
    ax1.set_title("Number of queries finding the max value," + str(percentage) + "% of trajectories success")
    ax1.grid()
    ax1.legend()
    plt.tight_layout()
    plt.savefig(plot_name + ".png", dpi=300)

    # 在第二个子图中绘制数据
    fig, (ax2) = plt.subplots(1, 1, figsize=(8, 6))
    ax2.plot([i for i in range(len(af_mean_list))], af_mean_list, marker='o', label='Acquisition function', c='r')
    ax2.plot([i for i in range(len(dt_mean_list))], dt_mean_list, marker='o', label='Decision transformer', c='b')
    ax2.plot([i for i in range(len(random_mean_list))], random_mean_list, marker='o', label='Random trajectory', c='y')
    ax2.errorbar([i for i in range(len(af_mean_list))], af_mean_list, yerr=np.sqrt(af_var_list), linestyle='None'
                 , capsize=5, c='coral', alpha=0.5)
    ax2.errorbar([i for i in range(len(dt_mean_list))], dt_mean_list, yerr=np.sqrt(dt_var_list), linestyle='None',
                 capsize=5, c='blue', alpha=0.5)
    ax2.errorbar([i for i in range(len(random_mean_list))], random_mean_list, yerr=np.sqrt(random_var_list),
                 linestyle='None',
                 capsize=5, c='green', alpha=0.5)
    ax2.set_xlabel('No. timestep')
    ax2.set_ylabel('accumulated reward')
    ax2.set_title("accumulated rewards in number of queries")
    ax2.grid()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(plot_name + "_comparing.png", dpi=300)

def generate_cumulative_max_list(lst):
    # print(lst)
    target_length = 51
    max_list = []
    current_max = float('-inf')

    for num in lst:
        current_max = max(current_max, num)
        max_list.append(current_max)

    last_element = max_list[-1]
    max_list += [last_element] * (target_length - len(max_list))

    return max_list

def PI_function(y_obs, u, mu, sigma): # probability of improvement
    # Compute the mean and variance of the Gaussian Process at x
#     mu, sigma, std = gpr(x_obs, y_obs, x_s)

    std = np.sqrt(sigma)
    # Compute the best observed target value
    f_best = np.max(y_obs)
    
    # Compute the standard normal cumulative distribution function
    Z = (mu - f_best - u) / std
    phi = norm.cdf(Z)
    
    # Compute the Probability of Improvement
#     max_at=np.argmax(phi)
    max_value = np.nanmax(phi)
    max_at = np.argmax(phi == max_value)
    return max_at

def plot_result(model_name):
    model = TrainableDT.from_pretrained(model_name)
    af_res_list = []
    dt_res_list = []

    af_trajectory_list = []
    dt_trajectory_list = []
    random_trajectory_list = []

    for root, dirs, files in os.walk(trajectory_path):
        for file_name in files:
            if file_name.endswith(".json"):
                print(os.path.join(root, file_name))

                #load function data
                with open(os.path.join(function_path, file_name), 'r') as f:
                    data = json.load(f)
                xs = data['xs']
                ys = data['ys']
                # x_ind = np.array([xs.index(data['PI']['x_obs'][0])])

                ind_len = len(xs)
                if len(xs) != len(ys):
                    print("!!! Function Samples Number Mismatch !!!")

                #load trajectory data
                with open(os.path.join(trajectory_path, file_name), 'r') as f:
                    data = json.load(f)
                rtg = data['RTG']
                scale_dict = data['Reward_dict']
                af_s = data['S']
                

                max_value = max(ys)
                max_index = xs[ys.index(max_value)]
                af_max_t = find_max_index(af_s)

                target_return = torch.tensor(rtg[:2]).float().reshape(1, 2)
                states = torch.tensor(data['S'][:2]).float().reshape(2, state_dim)
                actions = torch.cat((torch.tensor(data['A_ind_onehot'][0]).float().reshape(1, act_dim), torch.zeros((0, act_dim)).float()), dim=0)
                rewards = torch.tensor(data['R'][0]).float().reshape(1) #torch.cat((torch.tensor(data['R'][0]).float(), torch.zeros((0)).float()), dim=0)
                timesteps = torch.tensor((0,1)).reshape(1, 2).long()

                dt_trajectory = [scale_dict[str(data['S'][1])]]
                max_temp, t_max = 0.0, None
                for t in range(50):
                    actions = torch.cat([actions, torch.zeros((1, act_dim))], dim=0)
                    rewards = torch.cat([rewards, torch.zeros(1)])
                    output_dict = get_action(model,
                                        states,
                                        actions,
                                        rewards,
                                        target_return,
                                        timesteps)

                    mu_preds = output_dict['mu'].detach().numpy().squeeze()[-1]
                    sigma_preds = output_dict['sigma'].detach().numpy().squeeze()[-1]
                    y_obs = states.numpy().squeeze(1)
                    
                    action_index = PI_function(y_obs, 0.05, mu_preds, sigma_preds)
                    action = np.eye(ind_len)[action_index]
#                     action, action_index = one_hot_encode_max(action_output)
                    
                    if t_max is None and max_index - 0.015 < xs[action_index] < max_index + 0.015:
                        print('DT第%d次,AF第%d次找到最大值', t, af_max_t)
                        t_max = t

                    state = torch.tensor(ys[action_index]).view(1, state_dim)
                    action = torch.tensor(action).view(1, act_dim)
                    states = torch.cat([states, state], dim=0).float()
                    actions[-1] = action
                    
                    if scale_dict[str(ys[action_index])] <= max_temp:
                        rewards[-1] = torch.tensor(-penalty)
                    else:
                        rewards[-1] = torch.tensor(scale_dict[str(ys[action_index])] - max_temp - penalty)
                        max_temp = scale_dict[str(ys[action_index])]

                    dt_trajectory.append(scale_dict[str(ys[action_index])])
                    rtg = (target_return[0, -1] - rewards[-1]).reshape(1, 1)
                    target_return = torch.cat([target_return, rtg], dim=1).float()
                    timesteps = torch.cat([timesteps, torch.ones((1, 1)).long() * (t + 1)], dim=1)

                af_res_list.append(af_max_t)
                dt_res_list.append(t_max)
                af_trajectory = [scale_dict.get(str(item)) for item in af_s[1:]]
                af_trajectory = generate_cumulative_max_list(af_trajectory)
                af_trajectory_list.append(af_trajectory)
                dt_trajectory = generate_cumulative_max_list(dt_trajectory)
                dt_trajectory_list.append(dt_trajectory)
                print(dt_trajectory[-1])

                random_ys = random.sample(ys, 50)
                random_ys = [scale_dict.get(str(item)) for item in random_ys]
                random_ys = generate_cumulative_max_list(random_ys)
                random_trajectory_list.append(random_ys)

    print(af_res_list, dt_res_list)

    save_data = {'af': af_trajectory_list, 'dt': dt_trajectory_list}
    with open(model_name + ".json", 'w') as f:
        json.dump(save_data, f)
    try:
        generate_line_chart(af_res_list, dt_res_list, af_trajectory_list, dt_trajectory_list, random_trajectory_list, model_name)
    except ZeroDivisionError:
        # 异常处理代码块
        print("figuring failure")


def main():
    checkpoint_folders = find_checkpoint_folders(model_path)
    for model_name in checkpoint_folders:
        print('inference' + model_name)
        plot_result(model_name)

if __name__ == "__main__":
    main()

