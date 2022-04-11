import habitat
import habitat_sim

import os
import cv2
import numpy as np
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.utils.render_wrapper import overlay_frame
from habitat.utils.visualizations.utils import images_to_video

'''
Heuristic-Aided Navigation
'''


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

def model_runner():
    
    with habitat.Env(
        config=habitat.get_config(
            "configs/tasks/pointnav.yaml"
         )
        ) as env:
        
        observations = env.reset()
        print("==================================")
        print("envronment setup complete")
        step_count = 0
        past_location = None
        past_distance_to_goal = env._current_episode.info['geodesic_distance']
        all_obs = []

        while not env.episode_over:
            print(step_count)
            print(past_distance_to_goal)
            print(env.get_metrics()['distance_to_goal'])
            import ipdb; ipdb.set_trace()

            print("---")
            # print(env.episode_over)
            # env._current_episode.start_position
            # env._current_episode.info['geodesic_distance']
            # env._current_episode.goals[0].position
            # env._current_episode.start_rotation

            # observations['pointgoal_with_gps_compass']
            #if step_count == 287:
            #    import ipdb; ipdb.set_trace()
            spiral_counter = 1
            if env.episode_over or env.get_metrics()['distance_to_goal'] < .2: # at the goal
                action = {'action': 'STOP', 'action_args': None} 
                observations = env.step(action)
            elif step_count == 0:
                action = {'action': 'MOVE_FORWARD', 'action_args': None} 
                observations = env.step(action)
            elif env.get_metrics()['distance_to_goal'] < past_distance_to_goal:   # diff is smaller
                # repeat
                spiral_counter = 0
                past_distance_to_goal = env.get_metrics()['distance_to_goal']
                action = {'action': 'MOVE_FORWARD', 'action_args': None} 
                observations = env.step(action)
            else:
                action = {'action': 'TURN_RIGHT', 'action_args': None} 
                observations = env.step(action)
                action = {'action': 'TURN_RIGHT', 'action_args': None} 
                observations = env.step(action)
                action = {'action': 'TURN_RIGHT', 'action_args': None} 
                observations = env.step(action)
                
                print(spiral_counter)
                for _ in range(spiral_counter):
                    past_distance_to_goal = env.get_metrics()['distance_to_goal']
                    action = {'action': 'MOVE_FORWARD', 'action_args': None} 
                    observations = env.step(action)
                spiral_counter = 1


                info = env.get_metrics()
                use_ob = observations_to_image(observations, info)
                use_ob = overlay_frame(use_ob, info)
                draw_ob = use_ob[:]
              
                if  True:
                    draw_ob = np.transpose(draw_ob, (1, 0, 2))
                   # draw_obuse_ob = pygame.surfarray.make_surface(draw_ob)
                   # screen.blit(draw_obuse_ob, (0, 0))
                   # pygame.display.update()
                if True:
                    all_obs.append(draw_ob)
                if step_count == 250: 
                    #import ipdb; ipdb.set_trace()
                    all_obs = np.array(all_obs)
                    all_obs = np.transpose(all_obs, (0, 2, 1, 3))
                    make_video_cv2(all_obs, "interactive_play")

                #action = env.action_space.sample()
            #action = {'action': 'MOVE_FORWARD', 'action_args': None}  
            step_count += 1
        print(env.episode_over)
        print("==================================")


if __name__ == "__main__":
    model_runner()



