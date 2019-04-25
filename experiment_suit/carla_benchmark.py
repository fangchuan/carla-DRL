import numpy as np
from collections import deque
from utils import logger
from gym.spaces import Box, Discrete, Tuple
from utils.common import DEBUG_PRINT

METRICS_PARAMETERS = {
    'intersection_offroad': {'frames_skip': 10,
                             'frames_recount': 20,
                             'threshold': 0.3
                             },
    'intersection_otherlane': {'frames_skip': 10,
                               'frames_recount': 20,
                               'threshold': 0.4
                               },
    'collision_other': {'frames_skip': 10,
                        'frames_recount': 20,
                        'threshold': 400
                        },
    'collision_vehicles': {'frames_skip': 10,
                           'frames_recount': 30,
                           'threshold': 400
                           },
    'collision_pedestrians': {'frames_skip': 5,
                              'frames_recount': 100,
                              'threshold': 300
                              },
}



def benchmark_summary(env, action_fn, num_test_episodes,logger):

    percen_of_success = 0
    episode_completion = deque()
    episode_offroad = deque()
    episode_otherlane = deque()
    episode_collision_vehicle = deque()
    episode_collision_pedestrian = deque()
    episode_collision_other = deque()

    obs = env.reset()
    obs = env.reset_env()

    if isinstance(env.action_space, Box):
        max_action = np.array(env.action_space.high)
    elif isinstance(env.action_space, Discrete):
        max_action = 1

    steps = 0
    for episode in range(num_test_episodes):
        while True:
            action = action_fn(obs)[0]
            print("action : ", action)

            obs, reward, done, info = env.step(max_action * action)
            done = done.any() if isinstance(done, np.ndarray) else done
            steps += 1

            DEBUG_PRINT("steps: ",steps)
            DEBUG_PRINT("rewards: ", reward)

            if done:
                experiment_id = info["experiment_id"]
                weather = info["weather"]
                start_point = info["start_point"]
                end_point = info["end_point"]
                result = info["is_complete"]
                init_distance = info["init_distance"]
                final_distance = info["distance_to_goal"]
                duration = info["game_timestamp"]  # in milliseconds

                collision_vehicles = info["collision_vehicles"] if info["collision_vehicles"] > \
                                                                METRICS_PARAMETERS["collision_vehicles"]["threshold"] \
                    else 0
                collision_pedestrians = info["collision_pedestrians"] if info["collision_pedestrians"] > \
                                                                        METRICS_PARAMETERS["collision_pedestrians"][
                                                                            "threshold"] \
                    else 0
                collision_others = info["collision_other"] if info["collision_other"] > \
                                                            METRICS_PARAMETERS["collision_other"][
                                                                "threshold"] \
                    else 0
                intersection_offraod = info["intersection_offroad"] if info["intersection_offroad"] > \
                                                                    METRICS_PARAMETERS["intersection_offroad"][
                                                                        "threshold"] \
                    else 0
                intersection_otherlane = info["intersection_otherlane"] if info["intersection_otherlane"] > \
                                                                        METRICS_PARAMETERS["intersection_otherlane"][
                                                                            "threshold"] \
                    else 0
                distance_completion = np.clip((init_distance - final_distance), 0, float("inf")) / init_distance

                episode_collision_vehicle.append(collision_vehicles)
                episode_collision_pedestrian.append(collision_pedestrians)
                episode_collision_other.append(collision_others)
                episode_offroad.append(intersection_offraod)
                episode_otherlane.append(intersection_otherlane)
                percen_of_success = (percen_of_success + 1) if result else percen_of_success
                episode_completion.append(distance_completion)

                logger.record_tabular("experiment_id", experiment_id)
                logger.record_tabular("weather", weather)
                logger.record_tabular("start_point", start_point)
                logger.record_tabular("end_point", end_point)
                logger.record_tabular("init_distance", init_distance)
                logger.record_tabular("result", result)
                logger.record_tabular("final_distance",final_distance)
                logger.record_tabular("duration",duration)
                logger.dump_tabular()

                obs = env.reset_env()

                break

    average_percen_of_success = float(percen_of_success/num_test_episodes)
    average_episode_completion = np.mean(episode_completion)
    average_episode_offroad = np.mean(episode_offroad)
    average_episode_otherlane = np.mean(episode_otherlane)
    average_episode_collvehicle = np.mean(episode_collision_vehicle)
    average_episode_collpedes = np.mean(episode_collision_pedestrian)
    average_episode_collother = np.mean(episode_collision_other)
    metrics = {
        'episodes_completion':average_episode_completion,
        'episodes_fully_completed':average_percen_of_success,
        'intersection_offroad':average_episode_offroad,
        'intersection_otherlane':average_episode_otherlane,
        'collision_pedestrians':average_episode_collpedes,
        'collision_vehicles':average_episode_collvehicle,
        'collision_other':average_episode_collother
    }
    return metrics