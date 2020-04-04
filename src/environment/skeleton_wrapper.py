import gym
import numpy as np


class SkeletonWrapper(gym.Wrapper):
    def __init__(
            self, env, train_env,
            frame_skip, timestep_limit,
            footstep_weight, effort_weight, v_tgt_weight,
            alive_bonus, death_penalty, task_bonus
    ):
        gym.Wrapper.__init__(self, env)
        self.train_env = train_env

        self.frame_skip = frame_skip

        env.spec.timestep_limit = timestep_limit
        self.timestep_limit = timestep_limit
        self.episode_len = 0
        # reward weights from the environment
        self.footstep_weight = footstep_weight
        self.effort_weight = effort_weight
        self.v_tgt_weight = v_tgt_weight
        self.alive_bonus = alive_bonus
        self.death_penalty = death_penalty
        self.task_bonus = task_bonus

        # placeholders
        self.step_time = 0
        self.v_tgt_step_penalty = 0

    @staticmethod
    def pelvis_to_numpy(pelvis):
        array = [pelvis['height'], pelvis['pitch'], pelvis['roll']] + pelvis['vel']
        # pelvis_vel = [
        #     dx = + forward; -dy = + leftward, dz  = + upward,
        #     (angular velocity) - pitch, roll, yaw
        # ]
        return np.array(array)

    @staticmethod
    def dict_to_numpy(x):
        if type(x) is dict:
            return [v for k, v in x.items()]
        return x

    def leg_to_numpy(self, leg):
        observation = []
        for k, v in leg.items():
            observation += self.dict_to_numpy(v)
        return np.array(observation)

    def observation_to_numpy(self, observation):
        v_tgt_field = observation['v_tgt_field'].reshape(-1) / 10
        pelvis = self.pelvis_to_numpy(observation['pelvis'])  # 9
        r_leg = self.leg_to_numpy(observation['r_leg'])  # 44
        l_leg = self.leg_to_numpy(observation['l_leg'])  # 44
        flatten_observation = np.concatenate([
            v_tgt_field, pelvis, r_leg, l_leg
        ])
        return flatten_observation

    def reset(self, **kwargs):
        obs = self.env.reset()
        obs = self.observation_to_numpy(obs)
        self.env.env.env.env.d_reward['weight']['footstep'] = self.footstep_weight
        self.env.env.env.env.d_reward['weight']['effort'] = self.effort_weight
        self.env.env.env.env.d_reward['weight']['v_tgt'] = self.v_tgt_weight
        self.episode_len = 0
        return obs

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.observation_to_numpy(raw_obs)
        if self.train_env:
            reward = self.shape_reward(action, reward, done)
        self.episode_len += 1
        # if self.episode_len % 100 == 0 and self.train_env:
        #     # 0 -> 1; 1 -> 0
        #     self.move = (-1) * self.move + 1
        return obs, reward, done, info

    def footstep_reward(self):
        # TODO: нужно поменять v_tgt и пересчитать reward в соотвествии с self.move
        # TODO: сейчас работает с v_tgt_weight = 0
        pass

    @staticmethod
    def crossing_legs_penalty(state_desc):
        # stolen from Scitator
        pelvis_xyz = np.array(state_desc['body_pos']['pelvis'])
        left = np.array(state_desc['body_pos']['toes_l']) - pelvis_xyz
        right = np.array(state_desc['body_pos']['toes_r']) - pelvis_xyz
        axis = np.array(state_desc['body_pos']['head']) - pelvis_xyz
        cross_legs_penalty = np.cross(left, right).dot(axis)
        if cross_legs_penalty > 0:
            cross_legs_penalty = 0.0
        return 10 * cross_legs_penalty

    @staticmethod
    def bending_knees_bonus(state_desc):
        # stolen from Scitator
        r_knee_flexion = np.minimum(state_desc['joint_pos']['knee_r'][0], 0.)
        l_knee_flexion = np.minimum(state_desc['joint_pos']['knee_l'][0], 0.)
        # bonus only for one bended knee
        # bend_knees_bonus = np.abs(r_knee_flexion + l_knee_flexion)
        bend_knees_bonus = -np.minimum(r_knee_flexion, l_knee_flexion)
        # I believe 0.4 is optimal clip value
        bend_knees_bonus = np.clip(bend_knees_bonus, 0.0, 0.4)
        return bend_knees_bonus

    @staticmethod
    def get_v_body(state_desc):
        dx = state_desc['body_vel']['pelvis'][0]
        dy = state_desc['body_vel']['pelvis'][2]
        return np.asarray([dx, -dy])

    def get_v_tgt(self, state_desc):
        p_body = [state_desc['body_pos']['pelvis'][0], -state_desc['body_pos']['pelvis'][2]]
        v_tgt = self.env.env.env.env.vtgt.get_vtgt(p_body).T
        return v_tgt

    @staticmethod
    def pelvis_velocity_bonus(v_tgt, v_body):
        v_tgt_abs = (v_tgt ** 2).sum() ** 0.5
        v_body_abs = (v_body ** 2).sum() ** 0.5
        v_dp = np.dot(v_tgt, v_body)
        cos = v_dp / (v_tgt_abs * v_body_abs + 0.1)
        bonus = v_body_abs / 1.4
        return (np.sign(cos) * bonus)[0]

    @staticmethod
    def target_achieve_bonus(v_tgt):
        v_tgt_square = (v_tgt ** 2).sum()
        if 0.5 ** 2 < v_tgt_square <= 0.7 ** 2:
            return 0.1
        elif v_tgt_square <= 0.5 ** 2:
            return 1.0 - 3.5 * v_tgt_square
        else:
            return 0
        # elif 0.2 < v_tgt_abs <= 0.5:
        #     return 0.25
        # elif 0.1 < v_tgt_abs <= 0.2:
        #     return 1.0
        # elif v_tgt_abs <= 0.1:
        #     return 2.0

    @staticmethod
    def dense_effort_penalty(action):
        action = 0.5 * (action + 1)  # transform action from nn to environment range
        effort = 0.5 * (action ** 2).mean()
        return -effort

    def v_tgt_deviation_penalty(self, foot_step_reward, v_body, v_tgt):
        delta_v = v_body - v_tgt
        penalty = 0.5 * np.sqrt((delta_v ** 2).sum())
        self.v_tgt_step_penalty += penalty
        if foot_step_reward != 0.0:
            step_penalty = -self.v_tgt_step_penalty / max(1, self.step_time)
            self.v_tgt_step_penalty = 0
            self.step_time = 0
            return step_penalty
        else:
            self.step_time += 1
            return 0

    def shape_reward(self, action, foot_step_reward, done):
        state_desc = self.env.env.env.env.get_state_desc()
        if not self.alive_bonus:
            foot_step_reward -= 0.1 * self.frame_skip
        if not self.task_bonus:
            if foot_step_reward >= 450:
                foot_step_reward -= 500

        dead = (self.episode_len + 1) * self.frame_skip < self.timestep_limit
        if done and dead:
            foot_step_reward = self.death_penalty
        v_body = self.get_v_body(state_desc)
        v_tgt = self.get_v_tgt(state_desc)

        clp = self.crossing_legs_penalty(state_desc)
        # bkb = self.bending_knees_bonus(state_desc)
        pvb = self.pelvis_velocity_bonus(v_tgt, v_body)
        tab = self.target_achieve_bonus(v_tgt)
        vdp = self.v_tgt_deviation_penalty(foot_step_reward, v_body, v_tgt)
        dep = self.dense_effort_penalty(action)

        return [foot_step_reward, clp, vdp, pvb, dep, tab]

    def get_body_pos_vel(self):
        state_desc = self.env.env.env.env.get_state_desc()
        p_body = [state_desc['body_pos']['pelvis'][0], -state_desc['body_pos']['pelvis'][2]]
        v_body = [state_desc['body_vel']['pelvis'][0], -state_desc['body_vel']['pelvis'][2]]
        v_tgt = self.env.env.env.env.vtgt.get_vtgt(p_body).T
        return p_body, v_body, v_tgt
