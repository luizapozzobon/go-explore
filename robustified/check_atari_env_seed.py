import os
import sys
from copy import copy
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[3]
print(SRC_DIR)
sys.path.append(str(SRC_DIR))

import atari_py
import gym
import horovod.tensorflow as hvd
import tensorflow as tf
from atari_env_seed_wrapper.atari_seed_wrapper import patch_atari_seed
from baselines import bench
from baselines.common import set_global_seeds

from atari_reset.atari_reset.policies import CnnPolicy, GRUPolicy
from atari_reset.atari_reset.ppo import learn
from atari_reset.atari_reset.wrappers import (EpsGreedyEnv, NoopResetEnv, SubprocVecEnv,
                                  VecFrameStack, VideoWriter, my_wrapper)


def reboot_env(game, seed):
    global atari_py
    ale_py = patch_atari_seed(atari_py, seed, game)

    env = gym.make(game + "NoFrameskip-v4")
    return env


def test_env_seed(args):
    if "fetch" in args.game:
        raise ValueError("Only atari games supported.")

    hvd.init()
    print("initialized worker %d" % hvd.rank(), flush=True)

    set_global_seeds(hvd.rank())

    ncpu = 2
    config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=ncpu,
        inter_op_parallelism_threads=ncpu,
    )
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    tf.Session(config=config).__enter__()

    print("SAVE PATH", args.save_path)

    def make_env(rank, args, seed=None):
        def env_fn():
            if seed is not None:
                env = reboot_env(args.game, seed)
            else:
                env = gym.make(args.game + "NoFrameskip-v4")

            if args.seed_env:
                env.seed(0)

            # FIXME pass as an argument
            debug = False
            if debug:
                import time
                NUM_STEPS = 500
                env.reset()
                for step in range(NUM_STEPS):
                    # samples random action
                    action = env.action_space.sample()

                    # apply the action
                    obs, reward, done, info = env.step(action)

                    env.render()

                    # Wait a bit before the next frame unless you want to see a crazy fast video
                    time.sleep(0.01)

                    # If the epsiode is up, then start another one
                    if done:
                        env.reset()

                # Close the env
                env.close()

            env = bench.Monitor(
                env, "{}.monitor.json".format(rank), allow_early_resets=True
            )
            if False and rank % nenvs == 0 and hvd.local_rank() == 0:
                os.makedirs(args.save_path + "/vids/" + args.game, exist_ok=True)
                videofile_prefix = args.save_path + "/vids/" + args.game
                env = VideoWriter(env, videofile_prefix)
            if args.noops:
                os.makedirs(args.save_path, exist_ok=True)
                env = NoopResetEnv(
                    env,
                    30,
                    nenvs,
                    args.save_path,
                    num_per_noop=args.num_per_noop,
                    unlimited_score=args.unlimited_score,
                )
                env = my_wrapper(env, clip_rewards=True, sticky=args.sticky)

            if args.epsgreedy:
                env = EpsGreedyEnv(env)
            return env

        return env_fn

    test_seeds = args.env_seeds + args.test_seeds
    num_tests = len(test_seeds)
    original_save_path = copy(args.save_path)

    for i, test_seed in enumerate(test_seeds):
        print(f"Test {i+1}/{num_tests}, seed={test_seed}")

        args.save_path = f"{original_save_path}-seed_{test_seed}"

        nenvs = args.nenvs
        env = SubprocVecEnv(
            [make_env(i + nenvs * hvd.rank(), args, seed=test_seed) for i in range(nenvs)]
        )
        env = VecFrameStack(env, 4)

        args.policy = {"cnn": CnnPolicy, "gru": GRUPolicy}[args.policy]

        args.sil_pg_weight_by_value = False
        args.sil_vf_relu = False
        args.sil_vf_coef = 0
        args.sil_coef = 0
        args.sil_ent_coef = 0
        args.ent_coef = 0
        args.vf_coef = 0
        args.cliprange = 1
        args.l2_coef = 0
        args.adam_epsilon = 1e-8
        args.gamma = 0.99
        args.lam = 0.10
        args.scale_rewards = 1.0
        args.sil_weight_success_rate = True
        args.norm_adv = 1.0
        args.log_interval = 1
        args.save_interval = 100
        args.subtract_rew_avg = True
        args.clip_rewards = False

        print(args)

        learn(env, args, True)


if __name__ == "__main__":
    import time

    NUM_STEPS = 500
    seed = sys.argv[1]
    env = reboot_env(game="Riverraid", seed=seed)
    env.reset()
    for step in range(NUM_STEPS):
        # samples random action
        action = env.action_space.sample()

        # apply the action
        obs, reward, done, info = env.step(action)

        env.render()

        # Wait a bit before the next frame unless you want to see a crazy fast video
        time.sleep(0.01)

        # If the epsiode is up, then start another one
        if done:
            env.reset()

    # Close the env
    env.close()
