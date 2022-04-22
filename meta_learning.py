
import subprocess
from mdls.agent_reinforce import model_runner
import time


lr_lst =  [0.001, 0.01, 0.0001, 0.1]
training_episodes_lst = [250] #500, 1000, 5000]
policy_depth_lst = [2, 1, 3]
policy_width_lst =  [16, 8, 32, 64]
gamma_lst = [0.95, 0.99, 0.95]
script_id = int(time.time())

for training_episodes in training_episodes_lst:
    for lr in lr_lst:
        for policy_width in policy_width_lst:
           # for policy_depth in policy_depth_lst:
            for gamma in gamma_lst:
                policy_depth = 2
                model_runner(script_id, learning_rate=lr, training_episodes=training_episodes, policy_depth=policy_depth, policy_width=policy_width, gamma=gamma)

