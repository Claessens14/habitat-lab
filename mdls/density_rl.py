import habitat
import habitat_sim



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
            #action = env.action_space.sample()
            #action = {'action': 'MOVE_FORWARD', 'action_args': None}  
            step_count += 1
        print(env.episode_over)
        print("==================================")


if __name__ == "__main__":
    model_runner()



