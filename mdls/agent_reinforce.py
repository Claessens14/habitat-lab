import habitat
import habitat_sim

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
TRAINING_EPISODES = 1000


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
        returns: 0-3 for indexing [FORWARD, RIGHT, STOP, __]
        '''
        h = torch.nn.functional.relu(self.layer1(state_values))
        action_probs = torch.nn.functional.softmax(self.layer2(h))
        m = torch.distributions.Categorical(action_probs)
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
        all_rewards = [] # the inverse of the distance to goal
        best_rolling = []

        for episode in range(TRAINING_EPISODES):
            observations = env.reset()
            step_count = 0
            past_location = None
            past_distance_to_goal = env._current_episode.info['geodesic_distance']
            all_obs = []
            while not env.episode_over:
                print(step_count)
                print(past_distance_to_goal)
                print(env.get_metrics()['distance_to_goal'])
         
                print("---")
                # print(env.episode_over)
                # env._current_episode.start_position
                # env._current_episode.info['geodesic_distance']
                # env._current_episode.goals[0].position
                # env._current_episode.start_rotation
         
                # observations['pointgoal_with_gps_compass']
                #if step_count == 287:
                #    import ipdb; ipdb.set_trace()
               # if env.episode_over or env.get_metrics()['distance_to_goal'] < .2: # at the goal
               #     action = {'action': 'STOP', 'action_args': None} 
               #     observations = env.step(action)
               # elif step_count == 0:
               #     action = {'action': 'MOVE_FORWARD', 'action_args': None} 
               #     observations = env.step(action)
               # elif env.get_metrics()['distance_to_goal'] < past_distance_to_goal:   # diff is smaller
               #     # repeat
               #     past_distance_to_goal = env.get_metrics()['distance_to_goal']
               #     action = {'action': 'MOVE_FORWARD', 'action_args': None} 
               #     observations = env.step(action)
               # else:
               #     action = {'action': 'TURN_RIGHT', 'action_args': None} 
               #     observations = env.step(action)
            # (distance_to_goal, observations[pointgoal_with_gps_compas], difference_last_location)     
         #       import ipdb; ipdb.set_trace()
                action, log_prob_action = model(torch.tensor([env.get_metrics()['distance_to_goal'],
                                                                observations['pointgoal_with_gps_compass'][0],
                                                                observations['pointgoal_with_gps_compass'][1],
                                                                past_distance_to_goal]))
                past_distance_to_goal = env.get_metrics()['distance_to_goal']
                observations = env.step(action_space[action.item()])
                
                if episode % 10:  # draw every 10 episodes
                    info = env.get_metrics()
                    use_ob = observations_to_image(observations, info)
                    use_ob = overlay_frame(use_ob, info)
                    draw_ob = use_ob[:]
                    
                    if  True:
                        draw_ob = np.transpose(draw_ob, (1, 0, 2))
                        # do we really need printing??? 
                        all_obs.append(draw_ob)
                        #all_obs = np.append(all_obs, draw_ob)
                    if step_count == 250: 
                        all_obs = np.array(all_obs)
                        all_obs = np.transpose(all_obs, (0, 2, 1, 3))
                        make_video_cv2(all_obs, "interactive_play")
          
                step_count += 1
        print(env.episode_over)
        print("==================================")

if __name__ == "__main__":
    model_runner()



