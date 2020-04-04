import torch
import torch.nn as nn

from src.nn.layer import Layer


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class PolicyNet(nn.Module):
    def __init__(
            self,
            observation_dim, hidden_dim, action_dim,
            noisy, layer_norm, afn,
            residual=True, drop=0.0,
            *args, **kward
    ):
        super().__init__()
        # 2 * 11 * 11 = v_tgt_field
        # 97 = observation shape, 22 = action shape
        tgt_dim, obs_dim = observation_dim
        self.feature_layers = nn.Sequential(
            Layer(obs_dim + tgt_dim, hidden_dim, layer_norm, afn, residual, drop),
            Layer(hidden_dim, hidden_dim, layer_norm, afn, residual, drop),
            Layer(hidden_dim, hidden_dim, layer_norm, afn, residual, drop),
        )

        self.mean_layer = Layer(hidden_dim, action_dim, False, None)
        self.log_sigma_layer = Layer(hidden_dim, action_dim, False, None)
        self.fake_state = None

    @staticmethod
    def zero_state(state, done):
        state = (state * (1.0 - done).unsqueeze(-1))
        return state

    @staticmethod
    def state_to_numpy(state):
        return state.cpu().numpy()

    @staticmethod
    def state_from_numpy(state, device):
        if state is None:
            state = 0.0
        state_t = torch.tensor(
            state, dtype=torch.float32, device=device
        )
        return state_t

    def forward(self, observation, state=None):
        state = torch.zeros(
            (observation.size(0), 1),
            dtype=torch.float32,
            device=observation.device
        )
        features = self.feature_layers(observation)

        mean = self.mean_layer(features)
        log_sigma = self.log_sigma_layer(features)
        log_sigma = torch.clamp(log_sigma, LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_sigma, state


class QValueNet(nn.Module):
    def __init__(
            self,
            observation_dim, action_dim, hidden_dim, q_value_dim,
            noisy, layer_norm, afn,
            residual=True, drop=0.0,
            *args, **kward
    ):
        super().__init__()
        tgt_dim, obs_dim = observation_dim

        self.feature_layers = nn.Sequential(
            Layer(obs_dim + tgt_dim + action_dim, hidden_dim, layer_norm, afn, residual, drop),
            Layer(hidden_dim, hidden_dim, layer_norm, afn, residual, drop),
            Layer(hidden_dim, hidden_dim, layer_norm, afn, residual, drop),
        )

        self.q_value_layer = Layer(
            hidden_dim, q_value_dim, False, None
        )

    @staticmethod
    def zero_state(state, done):
        state = (state * (1.0 - done).unsqueeze(-1))
        return state

    @staticmethod
    def state_to_numpy(state):
        return state.cpu().numpy()

    @staticmethod
    def state_from_numpy(state, device):
        if state is None:
            state = 0.0
        state_t = torch.tensor(
            state, dtype=torch.float32, device=device
        )
        return state_t

    def forward(self, observation, action, state=None):
        state = torch.zeros(
            (observation.size(0), 1),
            dtype=torch.float32,
            device=observation.device
        )
        cat_input = torch.cat([observation, action], dim=-1)
        features = self.feature_layers(cat_input)
        q_value = self.q_value_layer(features)
        return q_value, state
