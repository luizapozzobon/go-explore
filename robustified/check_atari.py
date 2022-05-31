#!/usr/bin/env python
import argparse
import os
import sys
import gym
import numpy as np
from copy import copy
from pathlib import Path
ATARI_RESET_DIR = Path(__file__).resolve().parent / 'atari-reset'
sys.path.append(str(ATARI_RESET_DIR))

def reboot_env(game, seed):
    global atari_py
    ale_py = patch_atari_seed(atari_py, seed, game)

    env = gym.make(game + "NoFrameskip-v4")
    return env


def test_env_seed(args):
    if "fetch" in args.game:
        raise ValueError("Only atari games supported.")

    import filelock
    with filelock.FileLock('/tmp/robotstify.lock'):
        try:
            import goexplore_py.complex_fetch_env
        except Exception:
            print('Could not import complex_fetch_env, is goexplore_py in PYTHONPATH?')

    import tensorflow as tf
    import horovod.tensorflow as hvd
    hvd.init()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(hvd.local_rank())
    print('initialized worker %d' % hvd.rank())
    from baselines.common import set_global_seeds
    set_global_seeds(hvd.rank())
    from baselines import bench
    from baselines.common import set_global_seeds
    from atari_reset.wrappers import VecFrameStack, VideoWriter, my_wrapper,\
        EpsGreedyEnv, StickyActionEnv, NoopResetEnv, SubprocVecEnv, PreventSlugEnv, FetchSaveEnv, TanhWrap
    from atari_reset.ppo import learn
    from atari_reset.policies import CnnPolicy, GRUPolicy, FFPolicy

    ncpu = 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            # log_device_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    # config.gpu_options.allow_growth = True
    # config.gpu_options.visible_device_list = str(hvd.local_rank())
    tf.Session(config=config).__enter__()

    import atari_py
    global atari_py

    max_noops = 30 if args.noops else 0
    print('SAVE PATH', args.save_path)

    def make_env(rank, args, seed=None):
        def env_fn():
            if seed is not None:
                SRC_DIR = Path(__file__).resolve().parents[3]
                print(SRC_DIR)
                sys.path.append(str(SRC_DIR))
                from atari_env_seed_wrapper.atari_seed_wrapper import patch_atari_seed

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

            os.makedirs(args.save_path, exist_ok=True)
            env = bench.Monitor(env, args.save_path + "/{}.monitor.json".format(rank), allow_early_resets=True)
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

    print(3)
    test_seeds = args.env_seeds + args.test_seeds
    num_tests = len(test_seeds)
    original_save_path = copy(args.save_path)

    for i, test_seed in enumerate(test_seeds):
        print(4.1)
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

def test(args):
    import filelock

    import tensorflow as tf
    import horovod.tensorflow as hvd
    hvd.init()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(hvd.local_rank())
    print('initialized worker %d' % hvd.rank())
    from baselines.common import set_global_seeds
    set_global_seeds(hvd.rank())
    from baselines import bench
    from baselines.common import set_global_seeds
    from atari_reset.wrappers import VecFrameStack, VideoWriter, my_wrapper,\
        EpsGreedyEnv, StickyActionEnv, NoopResetEnv, SubprocVecEnv, PreventSlugEnv, FetchSaveEnv, TanhWrap
    from atari_reset.ppo import learn
    from atari_reset.policies import CnnPolicy, GRUPolicy, FFPolicy

    ncpu = 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            # log_device_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    # config.gpu_options.allow_growth = True
    # config.gpu_options.visible_device_list = str(hvd.local_rank())
    tf.Session(config=config).__enter__()

    max_noops = 30 if args.noops else 0
    print('SAVE PATH', args.save_path)

    def make_env(rank):
        def env_fn():
            if args.game == 'fetch':
                assert args.fetch_target_location is not None, 'For now, we require a target location for fetch'
                kwargs = {}
                dargs = vars(args)
                for attr in dargs:
                    if attr.startswith('fetch_'):
                        if attr == 'fetch_type':
                            kwargs['model_file'] = f"teleOp_{args.fetch_type}.xml"
                        elif attr != 'fetch_total_timestep':
                            kwargs[attr[len('fetch_'):]] = dargs[attr]

                env = goexplore_py.complex_fetch_env.ComplexFetchEnv(
                    **kwargs
                )
            elif args.game == 'fetch_dumb':
                env = goexplore_py.dumb_fetch_env.ComplexFetchEnv()
            else:
                env = gym.make(args.game + 'NoFrameskip-v4')
                if args.seed_env:
                    env.seed(0)
                # if args.unlimited_score:
                #     # This removes the TimeLimit wrapper around the env
                #     env = env.env
                # env = PreventSlugEnv(env)
            # change for long runs
            # env._max_episode_steps *= 1000

            os.makedirs(args.save_path, exist_ok=True)

            env = bench.Monitor(env, args.save_path + "/{}.monitor.json".format(rank), allow_early_resets=True)
            if False and rank%nenvs == 0 and hvd.local_rank()==0:
                os.makedirs(args.save_path + '/vids/' + args.game, exist_ok=True)
                videofile_prefix = args.save_path + '/vids/' + args.game
                env = VideoWriter(env, videofile_prefix)
            if 'fetch' not in args.game:
                if args.noops:
                    os.makedirs(args.save_path, exist_ok=True)
                    env = NoopResetEnv(env, 30, nenvs, args.save_path, num_per_noop=args.num_per_noop, unlimited_score=args.unlimited_score)
                    env = my_wrapper(env, clip_rewards=True, sticky=args.sticky)
                if args.epsgreedy:
                    env = EpsGreedyEnv(env)
            else:
                os.makedirs(f'{args.save_path}', exist_ok=True)
                env = FetchSaveEnv(env, rank=rank, n_ranks=nenvs, save_path=f'{args.save_path}/', demo_path=args.demo)
                env = TanhWrap(env)
            # def print_rec(e):
            #     print(e.__class__.__name__)
            #     if hasattr(e, 'env'):
            #         print_rec(e.env)
            # import time
            # import random
            # time.sleep(random.random() * 10)
            # print('\tSHOWING STUFF')
            # print_rec(env)
            # print('\n\n\n')
            return env
        return env_fn

    nenvs = args.nenvs
    env = SubprocVecEnv([make_env(i + nenvs * hvd.rank()) for i in range(nenvs)])
    env = VecFrameStack(env, 1 if 'fetch' in args.game else 4)

    if 'fetch' in args.game:
        print('Fetch environment, using the feedforward policy.')
        args.policy = FFPolicy
    else:
        args.policy = {'cnn': CnnPolicy, 'gru': GRUPolicy}[args.policy]

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

    learn(env, args, True)
    # learn(policy=policy, env=env, nsteps=256, log_interval=1, save_interval=100, total_timesteps=args.num_timesteps,
    #       load_path=args.load_path, save_path=args.save_path, game_name=args.game, test_mode=True, max_noops=max_noops)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='MontezumaRevenge')
    parser.add_argument('--num_timesteps', type=int, default=1e8)
    parser.add_argument('--num_per_noop', type=int, default=500)
    parser.add_argument('--policy', default='gru')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='', help='Where to save results to')
    parser.add_argument("--noops", help="Use 0 to 30 random noops at the start of each episode", action="store_true")
    parser.add_argument("--sticky", help="Use sticky actions", action="store_true")
    parser.add_argument("--seed_env", help="Seed the environment", action="store_true")
    parser.add_argument("--unlimited_score", help="Run with no time limit and fix the issue with score rollover", action="store_true")
    parser.add_argument('--nsubsteps', type=int, default=40)
    parser.add_argument("--epsgreedy", help="Take random action with probability 0.01", action="store_true")
    parser.add_argument('--demo', type=str, default=None)
    parser.add_argument('--nenvs', type=int, default=32)
    parser.add_argument("--ffshape", type=str, default='1x1024',
                        help="Shape of fully connected network: NLAYERxWIDTH")
    parser.add_argument('--fetch_nsubsteps', type=int, default=20)
    parser.add_argument('--fetch_timestep', type=float, default=0.002)
    parser.add_argument('--fetch_total_timestep', type=float, default=None)
    parser.add_argument("--inc_entropy_threshold", type=int, default=100,
                        help="Increase entropy when at this stage in the demo")
    parser.add_argument('--fetch_incl_extra_full_state', action='store_true', default=False)
    parser.add_argument('--fetch_state_is_pixels', action='store_true', default=False)
    parser.add_argument('--fetch_force_closed_doors', action='store_true', default=False)
    parser.add_argument('--fetch_include_proprioception', action='store_true', default=False)
    parser.add_argument('--fetch_state_azimuths', type=str, default='145_215')
    parser.add_argument('--fetch_type', type=str, default='boxes')
    parser.add_argument('--fetch_target_location', type=str, default=None)
    parser.add_argument('--fetch_state_wh', type=int, default=96)
    parser.add_argument('--nsteps', type=int, default=256)
    parser.add_argument('--ffmemsize', type=int, default=800)

    # Env seed experiments
    parser.add_argument(
        "--patch-env-seed",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, the seed of the environment will be changed to the one provided by the user",
    )
    parser.add_argument(
        "--env-seeds",
        nargs="+",
        default=[None],  # [1, 10, 42, 60],
        help="a list of seeds for the experiment's environment (applied if `patch-env-seed == True`)",
    )
    parser.add_argument(
        "--test-seeds",
        nargs="+",
        default=["9", "18", "50"],
        help="a list of seed for testing the agent",
    )
    parser.add_argument(
        "--test-episodes",
        type=int,
        default=20,
        help="the number of test episodes per `test_seed`",
    )

    args = parser.parse_args()

    if args.fetch_total_timestep is not None:
        args.fetch_timestep = args.fetch_total_timestep / args.fetch_nsubsteps

    args.im_cells = {}

    # assert not os.path.exists(args.save_path)

    import atari_reset.policies
    atari_reset.policies.FFSHAPE = args.ffshape
    atari_reset.policies.MEMSIZE = args.ffmemsize


    def check_done():
        import pickle, glob
        all_episodes = []
        for e in glob.glob(f'{args.save_path}/*.pickle'):
            all_episodes += pickle.load(open(e, 'rb'))
        all_episodes.sort(key=lambda x: x['start_step'])
        if len(all_episodes) < args.num_per_noop:
            return False
        n_to_consider = args.num_per_noop
        while n_to_consider < len(all_episodes) - 1 and all_episodes[n_to_consider - 1]['start_step'] == \
                all_episodes[n_to_consider]['start_step']:
            n_to_consider += 1
        return all([('score' in e) for e in all_episodes[:n_to_consider]])

    if args.patch_env_seed:
        # from check_atari_env_seed import test_env_seed
        print("Running environment seed patch testing.")
        if not check_done():
            test_env_seed(args)
    else:
        print("Running default go-explore testing.")
        if not check_done():
            test(args)
