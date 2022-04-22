import habitat
import habitat_sim

import time
import aim
import datetime
import torch
import os
import cv2
import numpy as np
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.utils.render_wrapper import overlay_frame

'''
Policy Based Navigation with REINFORCE optimization
'''

LEARNING_RATE = 0.0001
TRAINING_EPISODES = 10000
SAVE_INTERVAL = 500
DEVICE = 'cuda'

def make_video_cv2(observations, output_path):
    shp = observations[0].shape
    videodims = (shp[1], shp[0])
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    vid_name = output_path +  ".mp4"
    video = cv2.VideoWriter(vid_name, fourcc, 10, videodims)
    for ob in observations:
        bgr_im_1st_person = ob[..., 0:3][..., ::-1]
        video.write(bgr_im_1st_person)
    video.release()
    print("Saved to", vid_name)

class ReinforceModel(torch.nn.Module):
    def __init__(self, num_input, policy_width, num_action):
        super(ReinforceModel, self).__init__()
        self.num_input = num_input
        self.num_action = num_action
        
        self.layer1 = torch.nn.Linear(num_input, policy_width).to(device=DEVICE)
        self.layer2 = torch.nn.Linear(policy_width, num_action).to(device=DEVICE)
    
    def forward(self, state_values): 
        '''
        state_values: torch tensor 
        returns: 0-2 for indexing [FORWARD, RIGHT, LEFT]
        '''
        h = torch.nn.functional.relu(self.layer1(state_values)).to(device=DEVICE)
        action_probs = torch.nn.functional.softmax(self.layer2(h))
        m = torch.distributions.Categorical(action_probs)
        action = m.sample()
        return action, m.log_prob(action)


def model_runner(script_id, learning_rate=0.01, save_interval=100, training_episodes=1000, policy_depth=2, policy_width=16, gamma=0.95):    
    with habitat.Env(
        config=habitat.get_config(
            "configs/tasks/pointnav.yaml"
         )
        ) as env:

        ct = datetime.datetime.now()
        time_str = str(ct.strftime("%c").replace(" ", "-"))
        runtime_name = f"xp-new_r_and_gamma_srch___learning_rate-{learning_rate}___training_episodes-{training_episodes}___policy_width-{policy_width}___policy_depth-{policy_depth}___timestamp-{time_str}"
        runtime_name = runtime_name.replace(".", "_").replace(":", "_")
        runtime_dir_name = "./logs/" + runtime_name
        os.mkdir(runtime_dir_name)
        hparams = {
            "script_id": script_id,
            "learning_rate": learning_rate,
            "training_episodes":training_episodes,
            "policy_width": policy_width,
            "policy_depth": policy_depth,
            "id": int(time.time())
            #"batch_size": 32,
        }
        aim_sess = aim.Session(experiment=runtime_name)
        aim_sess.set_params(hparams, name="Hyper_Parameters")
        print("==================================")
        print("envronment setup complete")
        action_space = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
        model = ReinforceModel(3, policy_width, len(action_space))
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        running_reward = 0
        episode_rewards_lst = []
        avg_coverage_lst = []
        for episode in range(training_episodes):
            observations, log_prob_action_lst, step_rewards = env.reset(), [], []
            step_count = 0
            past_location = None
            past_distance_to_goal = env._current_episode.info['geodesic_distance']
            all_obs = []
            while not env.episode_over:
                #print(step_count)
                action, log_prob_action = model(torch.tensor([  observations['pointgoal_with_gps_compass'][0],
                                                                observations['pointgoal_with_gps_compass'][1],
                                                                observations['heading'][0]
                                                               ], device=DEVICE))#observations['compass'][0]  
                log_prob_action_lst.append(log_prob_action)
                try:
                    observations = env.step(action_space[action.item()])
                except Exception as e:
                    import ipdb; ipdb.set_trace()
                    print(e)
                #distance_diff = env._current_episode.info['geodesic_distance'] - env.get_metrics()['distance_to_goal']
                distance_diff = past_distance_to_goal - env.get_metrics()['distance_to_goal']
                past_distance_to_goal = env.get_metrics()['distance_to_goal']
                distance_diff_relu = distance_diff if distance_diff > 0 else 0
                distance_diff_relu_relative  = round(distance_diff_relu / env._current_episode.info['geodesic_distance'], 3)
                #print(distance_diff_relu_relative) 
               # r = round((2 * env._current_episode.info['geodesic_distance'] - env.get_metrics()['distance_to_goal']) / (2*env._current_episode.info['geodesic_distance']), 3) 
                step_rewards.append(distance_diff_relu_relative)
                if step_count > 5 and env.get_metrics()['distance_to_goal'] < 1 and not env.episode_over:
                    print(env.get_metrics()['distance_to_goal'])
                    env.step("STOP")
                    print("STOP")
                if env.episode_over:
                    #print(step_rewards)
                    aim_sess.track(sum(step_rewards), name="end_rewards") 
                    # coverage
                    coverage = env._current_episode.info['geodesic_distance'] -  env.get_metrics()['distance_to_goal'] 
                    coverage = coverage / env._current_episode.info['geodesic_distance']
                    aim_sess.track(coverage, name="coverage")
                if episode % save_interval == 0 and episode > 5:  # draw every 10 episodes
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
                        time_str = str(ct.strftime("%c").replace(" ", "-").replace(".", "_").replace(":", "_"))
                        r_avg = str(sum(episode_rewards_lst) / len(episode_rewards_lst)).replace(".", "_")
                        #fname = os.path.basename(__file__) + "-" + runtime_name + "-" + str(episode) + "-"  + time_str
                        fname =  "episode=" + str(episode) + "___r_avg=" + r_avg + "___"  + time_str
                        fpath_video = f"{runtime_dir_name}/video_dir/"
                        make_video_cv2(np_all_obs, fpath_video + "video___" + fname)
                        fpath_rewards = f"{runtime_dir_name}/reward_lst/"
                        os.makedirs(fpath_rewards, exist_ok=True)
                        torch.save(episode_rewards_lst, fpath_rewards + "rewards_lst___" + fname + ".pt")
                        fpath_ckpt = f"{runtime_dir_name}/ckpt/"
                        os.makedirs(fpath_ckpt, exist_ok=True)
                        torch.save(model.state_dict(), fpath_ckpt + "ckpt___" + fname)
                step_count += 1
            #running_reward = 0.05*sum(step_rewards) + 0.95*running_reward
            #print(f"{episode} -> running_reward: {running_reward}")
            print(f"{episode} -> reward: {sum(step_rewards)} ")
            #print(f"{episode} -> coverage: {env._ccurent_ env._current_episode.info['geodesic_distance']} %")
            episode_rewards_lst.append(sum(step_rewards))
            discounted_rewards = []
            for t in range(len(step_rewards)):
                Gt = 0
                power = 0
                for future_reward in step_rewards[t:]:
                    Gt = Gt + gamma**power * future_reward
                    power += 1
                discounted_rewards.append(Gt)
            discounted_rewards = torch.tensor(discounted_rewards)
            # Normalize
            #discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()
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



