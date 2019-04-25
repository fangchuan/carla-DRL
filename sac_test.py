import os
import sys
import gym
import numpy as np
import baselines.common.tf_util as Util
from utils.common import common_arg_parser
from utils import logger
from utils.common import MaxAndSkipEnv
from datetime import datetime
from drl_algorithm.sac import sac
from environment import carla_gym
from utils.email_sender import email_sender

@email_sender
def main(argvs):
    '''
    sac_test.py
    '''
    argparser = common_arg_parser()
    args = argparser.parse_args()
    model_file_save_path = args.save_path
    model_file_load_path = args.load_path
    total_step_numbers = args.total_steps_num
    is_play = args.play

    SEED = 0
    MAX_SKIP_FRAMES = 4
    CRITIC_L2_REGULARIATION = 1e-2
    ENTROPY_ALPHA = 0.2
    LEARN_RATE = 1e-3
    REPLAY_BUFFER_SIZE = 1e6
    GAMMA = 0.99
    BATCH_SIZE = 64
    TARGET_UPDATE_POLYAK = 0.999
    START_STEPS = 1e2
    USE_PER = True
    USE_IMAGE_ONLY_OBSERVATION = False

    logger.configure()

    with Util.make_session(num_cpu=8) as session:
        model_file_name = datetime.now().strftime("carla-sac-model-%Y-%m-%d-%H")
        model_file = model_file_name + '.ckpt'

        env = gym.make('Carla-v0')
        # 将env设置为SkipEnv, 返回MaxObservation, total_reward
        env = MaxAndSkipEnv(env, skip=MAX_SKIP_FRAMES, use_image_only_observation=False)

        if not os.path.exists(model_file_save_path):
            os.makedirs(model_file_save_path, exist_ok=True)
        model_file = os.path.join(model_file_save_path, model_file)

        action_fn = sac.sac(env=env,
                              session=session,
                              seed=SEED,
                              replay_size=int(REPLAY_BUFFER_SIZE),
                              gamma=GAMMA,
                              alpha=ENTROPY_ALPHA,
                              total_steps=int(total_step_numbers),
                              polyak=TARGET_UPDATE_POLYAK,
                              lr=LEARN_RATE,
                              batch_size=BATCH_SIZE,
                              start_steps=START_STEPS,
                              use_prioritized_replay=USE_PER,
                              use_image_only_observations=USE_IMAGE_ONLY_OBSERVATION,
                              model_file_load_path=model_file_load_path,
                              model_file_save_path=model_file)

        if is_play:
            logger.log("Running SAC trained model")
            obs = env.reset()
            steps = 0
            while True:
                action = action_fn(obs, deterministic=False)
                print("action : ", action)

                obs, reward, done, info = env.step(action)
                done = done.any() if isinstance(done, np.ndarray) else done
                steps += 1

                logger.record_tabular("steps", steps)
                logger.record_tabular("rewards", reward)

                if done:
                    obs = env.reset_env()
                    logger.dump_tabular()

        return action_fn


if __name__ == '__main__':
    main(sys.argv)
