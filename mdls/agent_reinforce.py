import habitat
import habitat_sim

import datetime
import torch
import os
import cv2
import numpy as np
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.utils.render_wrapper import overlay_frame

'''
Agent that navigate based on a reinforcemnent learlning policy
April 5th, 2022

'''

LEARNING_RATE = 0.0001
TRAINING_EPISODES = 10000
GAMMA = 0.99
SAVE_INTERVAL = 500

def make_video_cv2(observations, prefix=""):
    output_path = "./video_dir/"
    os.makedirs(output_path, exist_ok=True)
    shp = observations[0].shape
    videodims = (shp[1], shp[0])
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    vid_name = output_path + prefix + ".mp4"
    video = cv2.VideoWriter(vid_name, fourcc, 10, videodims)
    for ob in observations:
        bgr_im_1st_person = ob[..., 0:3][..., ::-1]
        video.write(bgr_im_1st_person)
    video.release()
    print("Saved to", vid_name)

class ReinforceModel(torch.nn.Module):
    def __init__(self, num_input, num_action):
        super(ReinforceModel, self).__init__()
        self.num_input = num_input
        self.num_action = num_action
        
        self.layer1 = torch.nn.Linear(num_input, 32)
        self.layer2 = torch.nn.Linear(32, num_action)
    
    def forward(self, state_values): 
        '''
        state_values: torch tensor (distance_to_goal, observations[pointgoal_with_gps_compas], difference_last_location) 
        returns: 0-3 for indexing [FORWARD, RIGHT, LEFT ]
        '''
        h = torch.nn.functional.relu(self.layer1(state_values))
        action_probs = torch.nn.functional.softmax(self.layer2(h))
        m = torch.distributions.Categorical(action_probs)
        #import ipdb; ipdb.set_trace()
        action = m.sample()
        return action, m.log_prob(action)


def model_runner():    
    with habitat.Env(
        config=habitat.get_config(
            "configs/tasks/pointnav.yaml"
         )
        ) as env:
        

        print("==================================")
        print("envronment setup complete")
        
        action_space = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
        model = ReinforceModel(4, len(action_space))
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        running_reward = 0
        epoch_rewards_lst = []
        for episode in range(TRAINING_EPISODES):
            #hereb
            observations, log_prob_action_lst, episode_rewards = env.reset(), [], []
            step_count = 0
            past_location = None
            past_distance_to_goal = env._current_episode.info['geodesic_distance']
            all_obs = []
            while not env.episode_over:
                #print(step_count)
                # print(past_distance_to_goal)
                # print(env.get_metrics()['distance_to_goal'])
         
                #print("---")
                # print(env.episode_over)
                # env._current_episode.start_position
                # env._current_episode.info['geodesic_distance']
                # env._current_episode.goals[0].position
                # env._current_episode.start_rotation
         
                action, log_prob_action = model(torch.tensor([env.get_metrics()['distance_to_goal'],
                                                                observations['pointgoal_with_gps_compass'][0],
                                                                observations['pointgoal_with_gps_compass'][1],
                                                                past_distance_to_goal]))
                log_prob_action_lst.append(log_prob_action)
                past_distance_to_goal = env.get_metrics()['distance_to_goal']
                observations = env.step(action_space[action.item()])
                r = round((2 * env._current_episode.info['geodesic_distance'] - env.get_metrics()['distance_to_goal']) / (2*env._current_episode.info['geodesic_distance']), 3)
                if r < 0: r = 0
                episode_rewards.append(r)
        
                if episode % SAVE_INTERVAL == 0:  # draw every 10 episodes
                    info = env.get_metrics()
                    use_ob = observations_to_image(observations, info)
                    use_ob = overlay_frame(use_ob, info)
                    draw_ob = use_ob[:]
                    draw_ob = np.transpose(draw_ob, (1, 0, 2))
                    all_obs.append(draw_ob)
                    if env.episode_over: 
                        np_all_obs = np.array(all_obs)
                        np_all_obs = np.transpose(np_all_obs, (0, 2, 1, 3))
                        ct = datetime.datetime.now()
                        time_str = str(ct.strftime("%c").replace(" ", "-"))
                        fname = os.path.basename(__file__) + "-" + str(episode) + "-"  + time_str
                        make_video_cv2(np_all_obs, fname)
                        torch.save(epoch_rewards_lst, "./output/reward_data/" + "r=" + str(r) + "_"+ fname + ".pt")
                        fname = "ckpt_r=" + str(r) + "_" + fname + ".pt"
                        torch.save(model.state_dict(), "./ckpt/" + fname)
                step_count += 1
            running_reward = 0.05*sum(episode_rewards) + 0.95*running_reward
            #print(f"{episode} -> running_reward: {running_reward}")
            print(f"{episode} -> reward: {episode_rewards[-1]} %")
            epoch_rewards_lst.append(episode_rewards[-1])
            discounted_rewards = []
            for t in range(len(episode_rewards)):
                Gt = 0
                power = 0
                for future_reward in episode_rewards[t:]:
                    Gt = Gt + GAMMA**power * future_reward
                    power += 1
                discounted_rewards.append(Gt)
            discounted_rewards = torch.tensor(discounted_rewards)
            # Normalize
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()
            # REINFORCE
            policy_loss_lst = []
            for log_prob, Gt in zip(log_prob_action_lst, discounted_rewards):
                policy_loss_lst.append(-log_prob * Gt)
            optimizer.zero_grad()
            policy_loss_sum = torch.stack(policy_loss_lst).sum()
            policy_loss_sum.backward()
            optimizer.step()
        
        print("==================================")

if __name__ == "__main__":
    model_runner()



