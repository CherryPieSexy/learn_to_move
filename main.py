import gym
import json
import torch

from src.environment.env_utils import init_environment, init_skeleton_environment
from src.nn import create_nets_sac, create_optimizers_sac
from src.losses import QValueLoss, NStepQValueLoss, NStepQValueLossSeparateEntropy
from src.sac import SAC
from src.r2d2.segment_sampler import SegmentSampler
from src.r2d2.experience_replay import PrioritizedExperienceReplay, ExperienceReplay
from src.r2d2.writer import Writer
from src.r2d2.trainer import Trainer


def main(config_file):
    # read all parameters
    with open(config_file, 'r') as f:
        parameters = json.load(f)

    # ============================== init environment ==============================
    env_parameters = parameters['environment']
    env_num = env_parameters['env_num']
    segment_len = env_parameters['segment_len']
    difficulty = env_parameters['difficulty']
    accuracy = env_parameters['accuracy']
    frame_skip = env_parameters['frame_skip']
    timestep_limit = env_parameters['timestep_limit']

    weights = env_parameters['weights']
    footstep_weight, effort_weight, v_tgt_weight = weights['reward_weights']
    alive_bonus, death_penalty, task_bonus = weights['alive_death_task']

    # # other gym environments
    # train_env, test_env = init_environment(env_num, env_name, frame_skip, segment_len)
    # observation_space = train_env.observation_space.shape[0]
    # action_space = train_env.action_space.shape[0]

    train_env, test_env = init_skeleton_environment(
        env_num, segment_len, difficulty, accuracy,
        frame_skip, timestep_limit,
        footstep_weight, effort_weight, v_tgt_weight,
        alive_bonus, death_penalty
    )

    observation_space = [2 * 11 * 11, 97]
    action_space = 22

    # ================================= init nets ==================================
    # available model types: 'feed_forward', 'recurrent', 'attention', 'skeleton'
    network_parameters = parameters['networks']
    model_type = network_parameters['model_type']
    device = torch.device(network_parameters['device_str'])

    # actor parameters
    actor_parameters = network_parameters['actor_parameters']
    hidden_dims_actor = actor_parameters['hidden_dim']
    noisy_actor = actor_parameters['noisy'] == 'True'
    layer_norm_actor = actor_parameters['layer_norm'] == 'True'
    afn_actor = actor_parameters['afn']
    residual_actor = actor_parameters['residual'] == 'True'
    drop_actor = actor_parameters['dropout']
    actor_lr = actor_parameters['learning_rate']
    normal = actor_parameters['normal'] == 'True'

    # critic parameters
    critic_parameters = network_parameters['critic_parameters']
    hidden_dims_critic = critic_parameters['hidden_dim']
    noisy_critic = critic_parameters['noisy'] == 'True'
    layer_norm_critic = critic_parameters['layer_norm'] == 'True'
    afn_critic = critic_parameters['afn']
    residual_critic = critic_parameters['residual'] == 'True'
    drop_critic = critic_parameters['dropout']
    q_value_dim = critic_parameters['q_value_dim']
    critic_lr = critic_parameters['learning_rate']

    policy_net, q_net_1, q_net_2, target_q_net_1, target_q_net_2 = create_nets_sac(
        model_type, observation_space, action_space,
        hidden_dims_actor, noisy_actor, layer_norm_actor, afn_actor, residual_actor, drop_actor, normal,
        hidden_dims_critic, noisy_critic, layer_norm_critic, afn_critic, residual_critic, drop_critic,
        device, q_value_dim + 1  # WARNING: q_value_dim here is reward_dim + 1!
    )
    policy_optimizer, q_optim_1, q_optim_2 = create_optimizers_sac(
        policy_net, q_net_1, q_net_2, actor_lr, critic_lr
    )

    # ================================= init agent =================================
    agent_parameters = parameters['agent_parameters']
    gamma = agent_parameters['gamma']
    soft_tau = agent_parameters['soft_tau']
    n_step_loss = agent_parameters['n_step_loss']
    rescaling = agent_parameters['rescaling'] == 'True'
    q_weights = agent_parameters['q_weights']
    q_value_loss = NStepQValueLossSeparateEntropy(gamma, device, q_weights, n_steps=n_step_loss, rescaling=rescaling)
    n_steps = agent_parameters['n_step_train']  # number of steps from tail of segment to learn from
    # aka eta, priority = eta * max_t(delta) + (1 - eta) * mean_t(delta)
    priority_weight = agent_parameters['priority_weight']
    use_observation_normalization = agent_parameters['use_observation_normalization'] == 'True'

    agent = SAC(
        policy_net, q_net_1, q_net_2, target_q_net_1, target_q_net_2,
        q_value_loss,
        policy_optimizer, q_optim_1, q_optim_2,
        soft_tau, device, action_space, n_steps,
        priority_weight,
        q_value_dim, q_weights,
        use_observation_normalization
    )

    # ================== init segment sampler & experience replay ==================
    replay_parameters = parameters['replay_parameters']
    log_dir = replay_parameters['log_dir']
    writer = Writer(log_dir, env_num)

    segment_sampler = SegmentSampler(agent, train_env, segment_len, q_weights, writer)

    replay_capacity = replay_parameters['replay_capacity']  # R2D2 -> 100k
    actor_size = replay_parameters['actor_size']
    critic_size = replay_parameters['critic_size']
    prioritization = replay_parameters['prioritization'] == 'True'
    exp_replay_init_fn = PrioritizedExperienceReplay if prioritization else ExperienceReplay
    experience_replay = exp_replay_init_fn(
        replay_capacity, segment_len,
        observation_space, action_space, q_value_dim,
        # (1, 2) for feed forward net, (hidden_size, hidden_size * 2) for lstm of mhsa
        actor_size, critic_size
    )

    # ================================ init trainer ================================
    trainer_parameters = parameters['trainer_parameters']
    start_priority_exponent = trainer_parameters['start_priority_exponent']
    end_priority_exponent = trainer_parameters['end_priority_exponent']
    start_importance_exponent = trainer_parameters['start_importance_exponent']
    end_importance_exponent = trainer_parameters['end_importance_exponent']
    prioritization_steps = trainer_parameters['prioritization_steps']

    exp_replay_checkpoint = trainer_parameters['exp_replay_checkpoint']
    if exp_replay_checkpoint == 'None':
        exp_replay_checkpoint = None
    agent_checkpoint = trainer_parameters['agent_checkpoint']
    if agent_checkpoint == 'None':
        agent_checkpoint = None
    load_full = trainer_parameters['load_full'] == 'True'

    trainer = Trainer(
        env_num, test_env,
        segment_sampler, log_dir, writer,
        agent, experience_replay,
        start_priority_exponent, end_priority_exponent,
        start_importance_exponent, end_importance_exponent,
        q_value_dim
    )
    trainer.load_checkpoint(agent_checkpoint, load_full)

    # ================================ train agent  ================================
    training_parameters = parameters['training_parameters']
    min_experience_len = training_parameters['min_experience_len']
    num_epochs = training_parameters['num_epochs']
    epoch_size = training_parameters['epoch_size']
    batch_size = training_parameters['batch_size']
    train_steps = training_parameters['train_steps']
    test_n = training_parameters['test_n']
    render = training_parameters['render'] == 'True'

    segment_file = training_parameters['segment_file']
    pretrain_critic = training_parameters['pretrain_critic'] == 'True'
    num_pretrain_epoch = training_parameters['num_pretrain_epoch']

    if num_pretrain_epoch > 0:
        trainer.pretrain_from_segments(
            segment_file, num_pretrain_epoch, batch_size,
            actor_size, critic_size
        )

    trainer.train(
        min_experience_len,
        num_epochs, epoch_size,
        train_steps, batch_size,
        test_n, render,
        prioritization_steps,
        pretrain_critic,
        exp_replay_checkpoint
    )

    writer.close()
    train_env.close()
    test_env.close()


if __name__ == '__main__':
    experiment_config = 'src/experiments/exp_8c_2.json'
    main(experiment_config)
