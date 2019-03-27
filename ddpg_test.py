import os
import sys
import gym
import tensorflow as tf
import baselines.common.tf_util as Util
from utils.common import common_arg_parser
from utils import logger
from utils.common import MaxAndSkipEnv
from datetime import datetime
from drl_algorithm.ddpg import ddpg


def main(argvs):
    argparser = common_arg_parser()
    args = argparser.parse_args()
    model_file_save_path = args.save_path
    model_file_load_path = args.load_path
    total_step_numbers = args.total_steps_num
    is_play = args.play

    SEED = 0
    MAX_SKIP_FRAMES = 4
    CRITIC_L2_REGULARIATION = 1e-2
    NOISE_STDDEV = 0.2
    ACTOR_LEARN_RATE = 1e-3
    CRITIC_LEARN_RATE = 1e-3
    REPLAY_BUFFER_SIZE = 5e4
    GAMMA = 0.99
    BATCH_SIZE = 100
    TARGET_UPDATE_POLYAK = 0.995
    START_STEPS = 1e3

    logger.configure()


    with Util.make_session(num_cpu=8) as session:
        model_file_name = datetime.now().strftime("carla-ddpg-model-%Y-%m-%d-%H")
        model_file = model_file_name + '.ckpt'

        env = gym.make('Carla-v0')
        # 将env设置为SkipEnv, 返回MaxObservation, total_reward
        # env = MaxAndSkipEnv(env, skip=MAX_SKIP_FRAMES)

        if not os.path.exists(model_file_save_path):
            os.makedirs(model_file_save_path, exist_ok=True)
        model_file = os.path.join(model_file_save_path, model_file)

        action_fn = ddpg.ddpg(env=env,
                             session=session,
                             seed=SEED,
                             use_action_noise=True,
                             use_param_noise=False,
                             noise_std=NOISE_STDDEV,
                             replay_size=int(REPLAY_BUFFER_SIZE),
                             gamma=GAMMA,
                             total_steps=int(total_step_numbers),
                             polyak=TARGET_UPDATE_POLYAK,
                             pi_lr=ACTOR_LEARN_RATE,
                             q_lr=CRITIC_LEARN_RATE,
                             batch_size=BATCH_SIZE,
                             start_steps=START_STEPS,
                             model_file_load_path=model_file_load_path,
                             model_file_save_path=model_file)

        if is_play:
            logger.log("Running DDPG trained model")
            obs = env.reset()
            steps = 0
            while True:
                action = action_fn(obs, apply_noise=False)
                print("action : ", action)
                action = action[0]

                max_action = env.action_space.high
                obs, reward, done, info = env.step(max_action * action)
                done = done.any() if isinstance(done, np.ndarray) else done
                steps += 1

                logger.record_tabular("steps", steps)
                logger.record_tabular("rewards", reward)

                if done:
                    obs = env.reset()
                    logger.dump_tabular()

        return action_fn



if __name__ == '__main__':
    main(sys.argv)
