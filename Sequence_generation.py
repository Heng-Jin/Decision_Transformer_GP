import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt  

step_list = list()

folder_path = sys.argv[1]
save_path = sys.argv[2]
penalty = float(sys.argv[3])

if not os.path.exists(save_path):
    os.makedirs(save_path)
    
def standardize(lst):
    mean = sum(lst) / len(lst)
    variance = sum([((x - mean) ** 2) for x in lst]) / len(lst)
    std_dev = variance ** 0.5
    standardized_lst = [(x - mean) / std_dev for x in lst]
    return standardized_lst

def min_max_scale(lst):
    min_val = min(lst)
    max_val = max(lst)
    scaled_lst = [(i - min_val) / (max_val - min_val) for i in lst]
    return scaled_lst

def calculate_stats(lst):
    min_val = min(lst)
    max_val = max(lst)
    mean_val = sum(lst) / len(lst)
    variance_val = sum([((x - mean_val) ** 2) for x in lst]) / len(lst)
    return min_val, max_val, mean_val, variance_val

def get_sorted_json_files(folder_path):
    json_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            json_file_path = os.path.join(folder_path, filename)
            json_files.append((json_file_path, os.path.getmtime(json_file_path)))

    # 按照文件的最近更改时间排序，最近的排在前面
    sorted_json_files = sorted(json_files, key=lambda x: x[1], reverse=True)
    return sorted_json_files

sorted_json_files = get_sorted_json_files(folder_path)



for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        with open(os.path.join(folder_path, filename), 'r') as f:
            data = json.load(f)

        xs = data['xs']
        ys = data['ys']
#         phi_list = data['PI']['phi']
#         phi_list = [[0.0 if math.isnan(x) else x for x in sublist] for sublist in phi_list]
        mu_list = data['PI']['mu']
        mu_list = [[0.0 if math.isnan(x) else x for x in sublist] for sublist in mu_list]
        std_list = data['PI']['std']
        std_list = [[0.0 if math.isnan(x) else x for x in sublist] for sublist in std_list]
        
        obs_index = data['PI']['max_at']
        one_hot_obs_index = []
        for ind in obs_index:
            one_hot = np.eye(1001)[int(ind)]
            one_hot_obs_index.append(one_hot.tolist())
#         ys_scale_standard = standardize(ys)
        ys_scale_standard = min_max_scale(ys)
        print(calculate_stats(ys_scale_standard))
        scale_dict = dict(zip(ys, ys_scale_standard))
        x_obs = data['PI']['x_obs']
        y_obs = data['PI']['y_obs']
        max_value = max(ys)
        max_index = xs[ys.index(max_value)]
        
#         print("最大值索引：", max_index, "最大值：", max_value, "obs最大值索引：", x_obs[y_obs.index(max(y_obs))], "obs最大值：", max(y_obs))

        couter = 0
        for i in range(0,len(y_obs)):
            if max_index-0.010 < x_obs[i] < max_index+0.010:
                print("第几部找到max value：",i,couter)
                break
            else:
                couter +=1
                

            
        if couter < 3:
            continue
        step_list.append(couter)
#         y_obs = y_obs[:couter+1]
#         x_obs = x_obs[1:couter+2]
#         y_obs = y_obs[:couter+1]
#         x_obs = x_obs[1:]
        
        max_temp = 0.0
        r=list()
        couter = 51 #min(couter +2, 51)
        for i in range(couter): #min(couter +2, 51)
#             if scale_dict.get(y_obs[i]) is None:
#                 print(y_obs[i])
#             if i > couter:
#                 r.append(0)
#             else:
            if scale_dict[y_obs[i]]<=max_temp:
                r.append(-penalty)
            else:
                r.append(scale_dict[y_obs[i]]-max_temp-penalty)
                max_temp = scale_dict[y_obs[i]]


        rtg = [sum(r[i:]) for i in range(len(r))]
#         r= r[1:]
#         rtg = rtg[1:]
        y_obs = y_obs[:couter]
        x_obs = x_obs[:couter+1]
        y_obs.insert(0, 0.0)
        y_obs = y_obs[:51]
#         phi_list = phi_list[:couter]
        mu_list = mu_list[:couter]
        std_list = std_list[:couter]
        blank_temp = [0.0] * 1001
        ind_temp = xs.index(x_obs[0])
        one_hot_temp = np.eye(1001)[ind_temp]

        obs_index.insert(0, ind_temp)
        one_hot_obs_index.insert(0, one_hot_temp.tolist())
#         phi_list.insert(0, blank_temp)
        mu_list.insert(0, blank_temp)
        std_list.insert(0, blank_temp)
#         phi_list = phi_list[:51]
        mu_list = mu_list[:51]
        std_list = std_list[:51]

#         print(filename)
#         print(len(rtg))
#         print(len(y_obs))
#         print(len(x_obs))
#         print(len(r))
#         print(rtg)
#         print(y_obs)
#         print(x_obs)
#         print(r)
        
        if len(rtg)!=len(y_obs) or len(rtg)!=len(x_obs):
            print("error!!!", len(rtg), len(y_obs), len(x_obs))
            print(filename)
            print(len(rtg))
            print(len(y_obs))
            print(len(x_obs))
            print(len(r))
            print(rtg)
            print(y_obs)
            print(x_obs)
            print(r)
        
#         save_data = {'RTG':rtg, 'S':y_obs, 'A':x_obs, 'R':r, 'Reward_dict':scale_dict, 'A_mu':mu_list, 'A_sigma':std_list, 'A_prob':phi_list, 'A_ind':obs_index, 'A_ind_onehot':one_hot_obs_index}
        save_data = {'RTG':rtg, 'S':y_obs, 'A':x_obs, 'R':r, 'Reward_dict':scale_dict, 'A_mu':mu_list, 'A_sigma':std_list, 'A_ind':obs_index, 'A_ind_onehot':one_hot_obs_index}
        
        with open(os.path.join(save_path, filename), 'w') as f:
            json.dump(save_data, f)