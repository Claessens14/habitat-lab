
import subprocess
from mdls.agent_reinforce import model_runner



lr_lst =  [0.001, 0.01, 0.0001, 0.1]
training_episodes_lst = [1000, 5000]
policy_depth_lst = [2, 1, 3]
policy_width_lst =  [16, 8, 32, 64]


for training_episodes in training_episodes_lst:
    for lr in lr_lst:
        for policy_width in policy_width_lst:
            for policy_depth in policy_depth_lst:
                model_runner(learning_rate=lr, training_episodes=training_episodes, policy_depth=policy_depth, policy_width=policy_width)

