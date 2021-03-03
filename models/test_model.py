# standalone script to test checkpoints.
import argparse

import torch
import torch.nn as nn
import numpy as np

from osim.env import L2M2019Env


class Layer(nn.Module):
    def __init__(self, in_features, out_features, norm=True, afn=True, residual=True):
        # Normalization, Layer, Add
        super().__init__()
        layer = [nn.Linear(in_features, out_features)]
        if norm:
            layer.append(nn.LayerNorm(out_features))
        if afn:
            layer.append(nn.ELU())
            layer.append(nn.Dropout(0.1))
        self.layer = nn.Sequential(*layer)
        self.residual = residual

    def forward(self, layer_in):
        layer_out = self.layer(layer_in)
        if self.residual:
            layer_out = layer_out + layer_in
        return layer_out


class PolicyNet(nn.Module):
    def __init__(
            self,
            observation_dim, hidden_dim, action_dim,
    ):
        super().__init__()
        # 2 * 11 * 11 = v_tgt_field
        # 97 = observation shape, 22 = action shape
        tgt_dim, obs_dim = observation_dim
        self.feature_layers = nn.Sequential(
            Layer(obs_dim + tgt_dim, hidden_dim, residual=False),
            Layer(hidden_dim, hidden_dim),
            Layer(hidden_dim, hidden_dim),
        )

        self.mean_layer = Layer(
            hidden_dim, action_dim,
            norm=False, afn=False, residual=False
        )
        self.log_sigma_layer = Layer(
            hidden_dim, action_dim,
            norm=False, afn=False, residual=False
        )

    def forward(self, observation):
        features = self.feature_layers(observation)

        mean = self.mean_layer(features)
        log_sigma = self.log_sigma_layer(features)
        return mean, log_sigma


def dict_to_numpy(x):
    if type(x) is dict:
        return [v for k, v in x.items()]
    return x


def leg_to_numpy(leg):
    observation = []
    for k, v in leg.items():
        observation += dict_to_numpy(v)
    return np.array(observation)


def _obs(obs):
    v_tgt_field = obs['v_tgt_field'].reshape(-1) / 10
    pelvis = obs['pelvis']  # 9
    pelvis = np.array([pelvis['height'], pelvis['pitch'], pelvis['roll']] + pelvis['vel'])
    r_leg = leg_to_numpy(obs['r_leg'])  # 44
    l_leg = leg_to_numpy(obs['l_leg'])  # 44
    flatten_observation = np.concatenate([
        v_tgt_field, pelvis, r_leg, l_leg
    ])
    return flatten_observation


def play_ep(env, model):
    obs, done = env.reset(), False
    ep_reward = 0.0
    while not done:
        obs_t = torch.tensor(_obs(obs), dtype=torch.float32)
        with torch.no_grad():
            action = model(obs_t)[0].cpu().numpy()
        obs, r, done, _ = env.step(action)
        ep_reward += r
    print(f'done, ep_reward = {ep_reward}')


def play_ep_rand(env):
    _, done = env.reset(), False
    while not done:
        env.step(env.action_space.sample())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name', '-p',
        help='name of file with model checkpoint to load.',
        type=str
    )
    parser.add_argument(
        '--n_episodes', '-n',
        help='number of episodes to play, default 3.',
        default=3, type=int
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = PolicyNet((11 * 11 * 2, 97), 1024, 22)
    print(args.model_name)
    checkpoint = torch.load(args.model_name, map_location='cpu')
    policy_checkpoint = checkpoint['policy_net']
    model.load_state_dict(policy_checkpoint)
    model.eval()

    env = L2M2019Env(integrator_accuracy=1e-3)
    for _ in range(args.n_episodes):
        play_ep(env, model)
    env.close()


if __name__ == '__main__':
    main()
