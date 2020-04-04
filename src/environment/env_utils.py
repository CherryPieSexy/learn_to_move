import time
import gym
import numpy as np
from osim.env import L2M2019Env

from .multiprocessing_env import SubprocVecEnv
from .skeleton_wrapper import SkeletonWrapper


class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)

        return action

    def reverse_action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)

        return action


class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, frame_skip):
        gym.Wrapper.__init__(self, env)
        self.frame_skip = frame_skip

    def reset(self):
        return self.env.reset()

    def step(self, action):
        step_reward = 0.0
        for _ in range(self.frame_skip):
            time_before_step = time.time()
            obs, reward, done, info = self.env.step(action)
            # early stopping of episode if step takes too long
            if (time.time() - time_before_step) > 10:
                done = True
            step_reward += reward
            if done:
                break
        return obs, step_reward, done, info


class SegmentPadWrapper(gym.Wrapper):
    def __init__(self, env, segment_len):
        gym.Wrapper.__init__(self, env)
        self.segment_len = segment_len // 2
        self.episode_len = 0
        self.residual_len = 0
        self.episode_ended = False
        self.last_transaction = None

    def reset(self):
        self.episode_ended = False
        self.episode_len = 0
        self.residual_len = 0
        return self.env.reset()

    def check_reset(self):
        if (self.episode_len + self.residual_len) % self.segment_len == 0:
            self.last_transaction[3]['reset'] = True
        self.residual_len += 1

    def step(self, action):
        if self.episode_ended:
            self.check_reset()
        else:
            self.episode_len += 1
            # obs, reward, done, info
            self.last_transaction = self.env.step(action)
            self.last_transaction[3]['reset'] = False
            if self.last_transaction[2]:
                self.episode_ended = True
                self.check_reset()
        return self.last_transaction


def init_environment(env_num, env_name, frame_skip, segment_len):
    def make_env():
        def _thunk():
            env = NormalizedActions(
                SegmentPadWrapper(
                    FrameSkipWrapper(gym.make(env_name), frame_skip),
                    segment_len
                )
            )
            return env

        return _thunk

    train_env = SubprocVecEnv([make_env() for _ in range(env_num)])
    test_env = NormalizedActions(
        FrameSkipWrapper(gym.make(env_name), frame_skip)
    )
    return train_env, test_env


def init_skeleton_environment(
        env_num, segment_len, difficulty, accuracy,
        frame_skip, timestep_limit=1000,
        footstep_weight=10, effort_weight=1, v_tgt_weight=1,
        alive_bonus=1, death_penalty=0, task_bonus=1,
        vec_train=True,
):
    seeds = np.random.choice(1_000, size=env_num + 1, replace=False)

    def make_env(seed):
        def _thunk():
            env = L2M2019Env(
                visualize=False,
                integrator_accuracy=accuracy,
                difficulty=difficulty,
                seed=seed
            )
            env = NormalizedActions(
                SegmentPadWrapper(
                    FrameSkipWrapper(
                        env, frame_skip
                    ), segment_len
                )
            )
            env = SkeletonWrapper(
                env, vec_train,
                frame_skip, timestep_limit,
                footstep_weight, effort_weight, v_tgt_weight,
                alive_bonus, death_penalty, task_bonus
            )

            return env

        return _thunk

    train_env = SubprocVecEnv([make_env(seeds[i]) for i in range(env_num)])
    test_env = L2M2019Env(
        visualize=False,
        integrator_accuracy=accuracy,
        difficulty=difficulty,
        seed=seeds[-1]
    )
    test_env = NormalizedActions(
        FrameSkipWrapper(
            test_env, frame_skip
        )
    )
    test_env = SkeletonWrapper(
        test_env, False,
        frame_skip, 2500,
        10, 1, 1,
        1, 0, 1
    )
    return train_env, test_env


class FrameSkipWrapperMoreInfo(gym.Wrapper):
    def __init__(self, env, frame_skip):
        gym.Wrapper.__init__(self, env)
        self.id = np.random.randint(0, 100000000)
        self.frame_skip = frame_skip
        self.max = np.random.randint(80, 100)
        self.n = 0

    def reset(self):
        self.n = 0
        self.max = np.random.randint(80, 100)
        self.id = np.random.randint(0, 100000000)
        return self.env.reset()

    def step(self, action):
        step_reward = 0.0
        infos = {}
        for i in range(self.frame_skip):
            time_before_step = time.time()
            obs, reward, done, _ = self.env.step(action)
            self.n += 1
            # if self.n > self.max:
            #     done = True
            info = {'state_desc': self.env.get_state_desc(),
                    'vtgt_global': self.env.vtgt.vtgt_obj.vtgt,
                    'obs': obs,
                    'reward': reward,
                    'done': done,
                    'id': self.id,
                    'env_seed': self.env.seed}
            infos[i] = info
            # early stopping of episode if step takes too long
            if (time.time() - time_before_step) > 10:
                done = True
            step_reward += reward
            if done:
                break
        return obs, step_reward, done, infos


def init_skeleton_environment_more_info(
        env_num, segment_len, difficulty, accuracy,
        frame_skip, timestep_limit=1000,
        footstep_weight=10, effort_weight=1, v_tgt_weight=1,
        alive_bonus=1, death_penalty=0, task_bonus=1,
        vec_train=True,
):
    seeds = np.random.choice(1_000, size=env_num + 1, replace=False)

    def make_env(seed):
        def _thunk():
            env = L2M2019Env(
                visualize=False,
                integrator_accuracy=accuracy,
                difficulty=difficulty,
                seed=seed
            )
            env.seed = seed
            env = NormalizedActions(
                SegmentPadWrapper(
                    FrameSkipWrapperMoreInfo(
                        env, frame_skip
                    ), segment_len
                )
            )
            env = SkeletonWrapper(
                env, vec_train,
                frame_skip, timestep_limit,
                footstep_weight, effort_weight, v_tgt_weight,
                alive_bonus, death_penalty, task_bonus
            )

            return env

        return _thunk

    train_env = SubprocVecEnv([make_env(seeds[i]) for i in range(env_num)])
    return train_env
