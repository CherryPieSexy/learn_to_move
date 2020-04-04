import torch
from .no_tgt import PolicyNet as NPolicyNet, QValueNet as NQValueNet  # N = no target
from .final import PolicyNet as FPolicyNet, QValueNet as FQValueNet  # F = final

models = {
    'skeleton_no_tgt': (NPolicyNet, NQValueNet),
    'skeleton_final': (FPolicyNet, FQValueNet),
}


def create_nets_sac(
        model_type,
        observation_dim, action_dim,
        hidden_dims_actor, noisy_actor, ln_actor, afn_actor, residual_actor, drop_actor, normal,
        hidden_dims_critic, noisy_critic, ln_critic, afn_critic, residual_critic, drop_critic,
        device,
        q_value_dim=1
):
    # init policy net
    policy_fn, q_value_fn = models[model_type]
    policy_net = policy_fn(
        observation_dim, hidden_dims_actor, action_dim,
        noisy_actor, ln_actor, afn_actor,
        residual=residual_actor, drop=drop_actor, stochastic=True, normal=normal
    )
    policy_net.to(device)
    # init q_net 1 and 2
    q_net_1 = q_value_fn(
        observation_dim, action_dim, hidden_dims_critic, q_value_dim,
        noisy_critic, ln_critic, afn_critic,
        residual=residual_critic, drop=drop_critic
    )
    q_net_2 = q_value_fn(
        observation_dim, action_dim, hidden_dims_critic, q_value_dim,
        noisy_critic, ln_critic, afn_critic,
        residual=residual_critic, drop=drop_critic
    )
    q_net_1.to(device)
    q_net_2.to(device)
    # init target q_net 1 and 2
    target_q_net_1 = q_value_fn(
        observation_dim, action_dim, hidden_dims_critic, q_value_dim,
        noisy_critic, ln_critic, afn_critic,
        residual=residual_critic, drop=drop_critic
    )
    target_q_net_2 = q_value_fn(
        observation_dim, action_dim, hidden_dims_critic, q_value_dim,
        noisy_critic, ln_critic, afn_critic,
        residual=residual_critic, drop=drop_critic
    )
    target_q_net_1.to(device)
    target_q_net_2.to(device)
    target_q_net_1.load_state_dict(q_net_1.state_dict())
    target_q_net_2.load_state_dict(q_net_2.state_dict())

    return policy_net, q_net_1, q_net_2, target_q_net_1, target_q_net_2


def create_optimizers(
        actor, critic, actor_lr, critic_lr
):
    actor_optim = torch.optim.Adam(actor.parameters(), actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), critic_lr)
    return actor_optim, critic_optim


def create_optimizers_sac(
        policy_net, q_net_1, q_net_2,
        policy_lr, q_lr
):
    policy_optim = torch.optim.Adam(policy_net.parameters(), policy_lr)
    q_optim_1 = torch.optim.Adam(q_net_1.parameters(), q_lr)
    q_optim_2 = torch.optim.Adam(q_net_2.parameters(), q_lr)
    return policy_optim, q_optim_1, q_optim_2
