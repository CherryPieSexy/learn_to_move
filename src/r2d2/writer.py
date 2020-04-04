import numpy as np
from tensorboardX import SummaryWriter


class Writer:
    def __init__(self, log_dir, n_agents):
        self.writer = SummaryWriter(log_dir)
        self.test_episodes = 0
        self.train_steps = 0
        self.time_steps = 0
        self.pretrain_steps = 0
        self.train_episodes = np.zeros(n_agents, dtype=int)

        self.q_head_names = [
            'footstep',
            'crossing_legs_penalty',
            'velocity_deviation_penalty',
            'pelvis_velocity_bonus',
            'effort_penalty',
            'target_achieve_bonus'
        ]

    def write_train_data(
            self,
            data, q_min, alpha, beta,
            train_time
    ):
        # alpha = priority exponent, beta = importance exponent
        policy_loss, alpha_loss, q_1_loss, q_2_loss, rewards = data
        self.writer.add_scalar('loss/policy', policy_loss, self.train_steps)
        self.writer.add_scalar('loss/sac_alpha', alpha_loss, self.train_steps)
        self.writer.add_scalar('loss/q_1', q_1_loss, self.train_steps)
        self.writer.add_scalar('loss/q_2', q_2_loss, self.train_steps)
        # self.writer.add_scalar('loss/std', std, self.train_steps)
        self.writer.add_scalar('loss/alpha', alpha, self.train_steps)
        self.writer.add_scalar('loss/beta', beta, self.train_steps)
        self.writer.add_scalar('reward/batch', rewards, self.train_steps)

        for i, q in enumerate(q_min):
            self.writer.add_scalar(
                'q_for_policy/{}'.format(self.q_head_names[i]), q, self.train_steps
            )

        sample_time, learn_time, upd_priority_time = train_time
        self.writer.add_scalar('time/exp_replay_sample', sample_time, self.train_steps)
        self.writer.add_scalar('time/learn', learn_time, self.train_steps)
        self.writer.add_scalar('time/upd_priority', upd_priority_time, self.train_steps)
        self.train_steps += 1

    def write_pretrain_data(self, data):
        policy_loss, alpha_loss, q_1_loss, q_2_loss, rewards = data
        self.writer.add_scalar('pretrain/policy', policy_loss, self.pretrain_steps)
        self.writer.add_scalar('pretrain/sac_alpha', alpha_loss, self.pretrain_steps)
        self.writer.add_scalar('pretrain/q_1', q_1_loss, self.pretrain_steps)
        self.writer.add_scalar('pretrain/q_2', q_2_loss, self.pretrain_steps)
        # self.writer.add_scalar('pretrain/std', std, self.pretrain_steps)
        self.pretrain_steps += 1

    def write_test_reward(self, reward, time):
        self.writer.add_scalar('reward/test', reward, self.test_episodes)
        self.writer.add_scalar('reward/test_time', time, self.test_episodes)
        self.test_episodes += 1

    def write_train_reward(self, done, episode_length, reward):
        for i, (d, r, ep_len) in enumerate(
                zip(done, reward, episode_length)
        ):
            if d and ep_len != 0:
                self.writer.add_scalars(
                    'agents/train_reward/', {
                        'agent_{}'.format(i): r,
                    }, self.train_episodes[i]
                )

                self.writer.add_scalars(
                    'agents/train_len/', {
                        'agent_{}'.format(i): ep_len
                    }, self.train_episodes[i]
                )

                self.writer.add_scalars(
                    'agents/reward_over_len', {
                        'agent_{}'.format(i): r / ep_len
                    }, self.train_episodes[i]
                )

                self.train_episodes[i] += 1

    def write_sample_time(self, segment_sample_time, exp_replay_time):
        env_time, nn_time, priority_time = segment_sample_time
        self.writer.add_scalar('time/env', env_time, self.time_steps)
        self.writer.add_scalar('time/nn_act', nn_time, self.time_steps)
        self.writer.add_scalar('time/priority', priority_time, self.time_steps)
        self.writer.add_scalar('time/exp_replay_push', exp_replay_time, self.time_steps)
        self.time_steps += 1

    def close(self):
        self.writer.close()
