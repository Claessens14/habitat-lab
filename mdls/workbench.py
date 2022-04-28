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
This is a work bench script. Meant to be copied, and have models put into place. It includes..
    - aim compatibility
    - video logging
    - model state dict logging
    - environment initiallization and runtime (while loop) 
    - descriptive naming for aim tracking, video loggin, and state_dict (ids, time, params etc) 
'''

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


def model_runner(script_id, description, learning_rate=0.01, save_interval=100, training_episodes=1000, policy_depth=2, policy_width=16, gamma=0.95):    
    with habitat.Env(
        config=habitat.get_config(
            "configs/tasks/pointnav.yaml"
         )
        ) as env:

        ct = datetime.datetime.now()
        time_str = str(ct.strftime("%c").replace(" ", "-"))
        runtime_name = f"script_id-{script_id}___learning_rate-{learning_rate}___training_episodes-{training_episodes}___policy_width-{policy_width}___policy_depth-{policy_depth}___timestamp-{time_str}"
        runtime_name = runtime_name.replace(".", "_").replace(":", "_")
        runtime_dir_name = "./logs/" + runtime_name
        os.mkdir(runtime_dir_name)
        hparams = {
            "script_id": script_id,
            "description": description,
            "learning_rate": learning_rate,
            "training_episodes":training_episodes,
            "policy_width": policy_width,
            "policy_depth": policy_depth,
            "id": int(time.time())
            # TODO -- "batch_size": 32,
        }
        aim_sess = aim.Session(experiment=runtime_name)
        aim_sess.set_params(hparams, name="Hyper_Parameters")
        # environ setup complete
        
        action_space = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
        running_reward = 0
        episode_rewards_lst = []
        avg_coverage_lst = []
        for episode in range(training_episodes):
            # setup episode
            observations, log_prob_action_lst, step_rewards = env.reset(), [], []
            step_count = 0
            past_location = None
            past_distance_to_goal = env._current_episode.info['geodesic_distance']
            all_obs = []
            while not env.episode_over:
                # episode step 
               
                #input 
                input_tensor = torch.tensor([  observations['pointgoal_with_gps_compass'][0], 
                               observations['pointgoal_with_gps_compass'][1],
                               observations['heading'][0]
                              ], device=DEVICE) #observations['compass'][0]  

                # get and log action
                action, log_prob_action = model(inpu_tensor)                                 
                log_prob_action_lst.append(log_prob_action)
                
                # step in envronment
                observations = env.step(action_space[action.item()])
                
                # relative improvement to gaol (reward shaping
                distance_diff = past_distance_to_goal - env.get_metrics()['distance_to_goal']
                past_distance_to_goal = env.get_metrics()['distance_to_goal']
                distance_diff_relu = distance_diff if distance_diff > 0 else 0
                distance_diff_relu_relative  = round(distance_diff_relu / env._current_episode.info['geodesic_distance'], 3)
                step_rewards.append(distance_diff_relu_relative)
                
                
                if step_count > 5 and env.get_metrics()['distance_to_goal'] < 1 and not env.episode_over:
                    #print(env.get_metrics()['distance_to_goal'])
                    env.step("STOP")
                   # print("STOP")
                if env.episode_over:
                    aim_sess.track(sum(step_rewards), name="end_rewards") 
                    # coverage
                    coverage = env._current_episode.info['geodesic_distance'] -  env.get_metrics()['distance_to_goal'] 
                    coverage = coverage / env._current_episode.info['geodesic_distance']
                    aim_sess.track(coverage, name="coverage")
                    print(str(episode) + " --> coverage: " + str(round(coverage, 3)))
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
            
            #print(f"{episode} -> reward: {sum(step_rewards)} ")
            #print(f"{episode} -> coverage: {env._ccurent_ env._current_episode.info['geodesic_distance']} %")
            

        # reward declaring

        # Optimziing Policy



if __name__ == "__main__":
    model_runner()



