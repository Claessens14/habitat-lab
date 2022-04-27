
import subprocess
from mdls.agent_reinforce import model_runner
import time
from tqdm import tqdm
import argparse

lr_lst =  [0.001, 0.01, 0.0001, 0.1]
tqdm_lr_lst = tqdm(lr_lst)
training_episodes_lst = [700] #1000, 5000]
policy_depth_lst = [2, 1, 3]
policy_width_lst =  [256, 128, 16, 8, 32, 64]
gamma_lst = [0.95, 0.99, 0.95]
script_id = int(time.time())




parser = argparse.ArgumentParser()
parser.add_argument("--comment", type=str)
args = parser.parse_args()
description = args.comment
if description is None: exit("\nMUST PROVIDE COMMENT ARGUEMENT. EXITING PROGRAMMING: 1\n")
description = description.replace(" ", "_")

for training_episodes in training_episodes_lst:
    for lr in lr_lst:
        for policy_width in policy_width_lst:
           # for policy_depth in policy_depth_lst:
            for gamma in gamma_lst:
                policy_depth = 2
                model_runner(script_id, description, learning_rate=lr, training_episodes=training_episodes, policy_depth=policy_depth, policy_width=policy_width, gamma=gamma)
