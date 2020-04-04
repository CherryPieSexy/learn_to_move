import h5py
import numpy as np
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter

from off_policy.networks.run_skeleton_final import PolicyNet, QValueNet
from off_policy.networks.run_skeleton_final_5 import PolicyNet as FinalPolicy, QValueNet as FinalQValue


class DistillTrainer:
    """
    class for distillation knowledge from one agent (teacher) to another (student)
    agents must provide .act(observation, hidden) -> (mean, log_sigma) method

    """

    def __init__(
            self,
            device, data_loader,
            teacher_policy, teacher_critic_1, teacher_critic_2,
            student_policy, student_critic_1, student_critic_2,
            policy_optimizer, critic_optimizer_1, critic_optimizer_2,
            logdir, writer,
            action_dim=22):
        self.device = device
        self.data_loader = data_loader

        self.teacher_policy = teacher_policy
        self.teacher_critic_1 = teacher_critic_1
        self.teacher_critic_2 = teacher_critic_2

        self.student_policy = student_policy
        self.student_critic_1 = student_critic_1
        self.student_critic_2 = student_critic_2

        self.policy_optimizer = policy_optimizer
        self.critic_optimizer_1 = critic_optimizer_1
        self.critic_optimizer_2 = critic_optimizer_2

        self.logdir = logdir
        self.writer = writer
        self.action_dim = action_dim

    @staticmethod
    def _kl_loss(teacher_mu, teacher_log_sigma, student_mu, student_log_sigma):
        distribution_dim = teacher_mu.size(-1)
        mu_diff = teacher_mu - student_mu
        kl_div = 0.5 * (
                torch.exp(student_log_sigma - teacher_log_sigma).sum(-1) +
                (mu_diff ** 2 / torch.exp(teacher_log_sigma)).sum(-1) -
                distribution_dim +
                (teacher_log_sigma.sum(-1) - student_log_sigma.sum(-1))
        )
        return kl_div.mean()  # mean over batch and time

    def _optimize_kl(self, kl_loss):
        self.policy_optimizer.zero_grad()
        kl_loss.backward()
        self.policy_optimizer.step()

    def _optimize_critics(self, critic_1_loss, critic_2_loss):
        self.critic_optimizer_1.zero_grad()
        critic_1_loss.backward()
        self.critic_optimizer_1.step()

        self.critic_optimizer_2.zero_grad()
        critic_2_loss.backward()
        self.critic_optimizer_2.step()

    def _train_step(self, observation, hidden):
        batch, time, obs_size = observation.shape
        if obs_size != 339:
            # tgt_vel = np.full((batch, time, 2 * 11 * 11), 0.14, dtype=np.float)
            tgt_vel = np.random.normal(0, 0.1, size=(batch, time, 2 * 11 * 11))
            observation = np.concatenate([tgt_vel, observation], axis=-1)
        observation_t = torch.tensor(observation, dtype=torch.float32, device=self.device)
        policy_hidden, critic_hidden_1, critic_hidden_2 = hidden

        # optimize policy
        with torch.no_grad():
            teacher_mu, teacher_log_sigma, _ = self.teacher_policy(observation_t)
        student_mu, student_log_sigma, policy_hidden = self.student_policy(observation_t, policy_hidden)
        student_log_sigma = student_log_sigma - 0.25
        kl_loss = self._kl_loss(teacher_mu, teacher_log_sigma, student_mu, student_log_sigma)
        self._optimize_kl(kl_loss)

        # optimize critic
        with torch.no_grad():
            teacher_q_value_1, _ = self.teacher_critic_1(observation_t, teacher_mu)
            teacher_q_value_2, _ = self.teacher_critic_2(observation_t, teacher_mu)
            student_mu, student_log_sigma, _ = self.student_policy(observation_t)
        target_q_value = torch.min(teacher_q_value_1, teacher_q_value_2)
        student_q_value_1, critic_hidden_1 = self.student_critic_1(observation_t, student_mu, critic_hidden_1)
        student_q_value_2, critic_hidden_2 = self.student_critic_2(observation_t, student_mu, critic_hidden_2)
        critic_1_loss = ((target_q_value - student_q_value_1) ** 2).mean()
        critic_2_loss = ((target_q_value - student_q_value_2) ** 2).mean()
        self._optimize_critics(critic_1_loss, critic_2_loss)

        hidden = [policy_hidden, critic_hidden_1, critic_hidden_2]

        return kl_loss.item(), critic_1_loss.item(), critic_2_loss.item(), hidden

    def train(self, num_epochs, log_alpha):
        step = 0
        hidden = [None for _ in range(3)]
        for epoch in range(num_epochs):
            for batch in tqdm(self.data_loader):
                step += 1
                kl_loss, critic_1_loss, critic_2_loss, hidden = self._train_step(batch, hidden)
                # optimization done, hidden states must be detached
                hidden = [(h[0].detach(), h[1].detach()) for h in hidden]
                self.writer.add_scalar('kl_loss', kl_loss, step)
                self.writer.add_scalar('critic_1_loss', critic_1_loss, step)
                self.writer.add_scalar('critic_2_loss', critic_2_loss, step)

            torch.save(
                {
                    'policy_net': self.student_policy.state_dict(),
                    'q_net_1': self.student_critic_1.state_dict(),
                    'q_net_2': self.student_critic_2.state_dict(),

                    'policy_optim': self.policy_optimizer.state_dict(),
                    'q_net_1_optim': self.critic_optimizer_1.state_dict(),
                    'q_net_2_optim': self.critic_optimizer_2.state_dict(),

                    'log_alpha': log_alpha
                },
                self.logdir + 'distill_{}.pth'.format(epoch)
            )


class NumPyDataLoader:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __iter__(self):
        np.random.shuffle(self.data)
        for i in range(0, len(self.data) - 1, self.batch_size):
            batch = self.data[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                yield batch


class ExpReplayDataLoader:
    def __init__(self, exp_replay_file, batch_size):
        f = h5py.File(exp_replay_file, mode='r')
        data_group = f['experience_replay']
        self.data = data_group['_observations'][()]
        self.batch_size = batch_size

    def __iter__(self):
        np.random.shuffle(self.data)
        for i in range(0, len(self.data) - 1, self.batch_size):
            batch = self.data[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                yield batch


if __name__ == '__main__':
    logdir = 'distill_logs/d_five/'
    checkpoint_name = 'logs/learning_to_move/8b_3/epoch_39.pth'
    exp_replay_file = 'logs/learning_to_move/8b_3/exp_replay_40.h5'
    device = torch.device('cuda')
    checkpoint = torch.load(checkpoint_name, map_location='cpu')
    log_alpha_from_checkpoint = checkpoint['log_alpha']
    batch_size = 128
    # data = np.load('result.npy')
    # data_loader = DataLoader(data, batch_size)
    data_loader = ExpReplayDataLoader(exp_replay_file, batch_size)
    teacher_hidden = 1024
    student_hidden = 512
    q_dim = 6

    # teacher policy
    teacher_policy = PolicyNet(
        [2 * 11 * 11, 97], teacher_hidden, 22, False, True, 'elu', True, 0.1
    )
    teacher_policy.load_state_dict(checkpoint['policy_net'])
    teacher_policy.to(device)

    # teacher critic 1
    teacher_critic_1 = QValueNet(
        [2 * 11 * 11, 97], 22, teacher_hidden, q_dim, False, True, 'relu', True, 0.1
    )
    teacher_critic_1.load_state_dict(checkpoint['q_net_1'])
    teacher_critic_1.to(device)

    # teacher critic 2
    teacher_critic_2 = QValueNet(
        [2 * 11 * 11, 97], 22, teacher_hidden, q_dim, False, True, 'relu', True, 0.1
    )
    teacher_critic_2.load_state_dict(checkpoint['q_net_2'])
    teacher_critic_2.to(device)

    # student policy
    student_policy = FinalPolicy(
        [2 * 11 * 11, 97], student_hidden, 22, False, True, 'elu', True, 0.1
    )
    student_policy.to(device)

    # student critic 1
    student_critic_1 = FinalQValue(
        [2 * 11 * 11, 97], 22, student_hidden, q_dim, False, True, 'relu', True, 0.0
    )
    student_critic_1.to(device)

    # student critic 2
    student_critic_2 = FinalQValue(
        [2 * 11 * 11, 97], 22, student_hidden, q_dim, False, True, 'relu', True, 0.0
    )
    student_critic_2.to(device)

    # optimizers
    policy_optimizer = torch.optim.Adam(student_policy.parameters(), 1e-4)
    critic_optimizer_1 = torch.optim.Adam(student_critic_1.parameters(), 1e-4)
    critic_optimizer_2 = torch.optim.Adam(student_critic_2.parameters(), 1e-4)

    writer = SummaryWriter(logdir)
    trainer = DistillTrainer(
        device, data_loader,
        teacher_policy, teacher_critic_1, teacher_critic_2,
        student_policy, student_critic_1, student_critic_2,
        policy_optimizer, critic_optimizer_1, critic_optimizer_2,
        logdir, writer, 22
    )
    trainer.train(10, log_alpha_from_checkpoint)
