import pickle
import numpy as np
from time import time


class SegmentTree:
    def __init__(
            self, capacity, segment_len,
            observation_shape, action_shape, reward_dim,
            actor_state_size, critic_state_size
    ):
        self._capacity = capacity
        self._index = 0
        self._full = False

        self._sum_tree = np.zeros((2 * capacity - 1,), dtype=np.float32)
        self._observations = np.zeros((capacity, segment_len + 1, sum(observation_shape)), dtype=np.float32)
        self._actions = np.zeros((capacity, segment_len, action_shape), dtype=np.float32)
        self._rewards = np.zeros((capacity, segment_len, reward_dim), dtype=np.float32)
        self._is_done = np.zeros((capacity, segment_len), dtype=np.float32)
        self._actor_states = np.zeros((capacity, actor_state_size), dtype=np.float32)
        self._critic_states = np.zeros((capacity, critic_state_size), dtype=np.float32)

    def _propagate(self, index):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self._sum_tree[parent] = self._sum_tree[left] + self._sum_tree[right]
        if parent != 0:
            self._propagate(parent)

    def update(self, index, value):
        self._sum_tree[index] = value
        self._propagate(index)

    def _append(self, data):
        (obs, action, reward, done), actor_state, critic_state = data
        self._observations[self._index] = obs
        self._actions[self._index] = action
        self._rewards[self._index] = reward
        self._is_done[self._index] = done
        self._actor_states[self._index] = actor_state
        self._critic_states[self._index] = critic_state

    def append(self, data, value):
        self._append(data)
        # self._data[self._index] = data
        self.update(self._index + self._capacity - 1, value)
        self._index = (self._index + 1) % self._capacity
        self._full = self._full or self._index == 0

    def _retrieve(self, index, value):
        left, right = 2 * index + 1, 2 * index + 2
        if left >= len(self._sum_tree):
            return index
        elif value <= self._sum_tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self._sum_tree[left])

    def _get_data_by_idx(self, idx):
        data = (
            (
                self._observations[idx],
                self._actions[idx],
                self._rewards[idx],
                self._is_done[idx]
            ),
            self._actor_states[idx],
            self._critic_states[idx]
        )
        return data

    def find(self, value):
        index = self._retrieve(0, value)
        data_index = index - self._capacity + 1
        # data, value, data_index, tree index
        result = (
            # self._data[data_index % self._capacity],
            self._get_data_by_idx(data_index % self._capacity),
            self._sum_tree[index],
            data_index, index
        )
        return result

    def get(self, data_index):
        data = self._get_data_by_idx(data_index % self._capacity)
        return data

    def total(self):
        return self._sum_tree[0]

    def save_raw_data(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(
                {
                    'observations': self._observations,
                    'actions': self._actions,
                    'rewards': self._rewards,
                    'is_done': self._is_done
                }, f
            )


class PrioritizedExperienceReplay:
    def __init__(
            self, capacity, segment_len,
            observation_shape, action_shape, reward_dim,
            actor_state_size, critic_state_size
    ):
        self.capacity = capacity
        self._data = SegmentTree(
            capacity, segment_len,
            observation_shape, action_shape, reward_dim,
            actor_state_size, critic_state_size
        )

    def push(self, sample, priority_loss, alpha):
        push_time_start = time()
        segment, actor_state, critic_state = sample
        obs, action, reward, done = segment
        for i in range(len(priority_loss)):
            # if not done[i][0]:  # prevent appending 'broken' segments
            self._data.append(
                (
                    (obs[i], action[i], reward[i], done[i]),  # segment itself
                    actor_state[i], critic_state[i]  # network states
                ),
                priority_loss[i] ** alpha
            )
        push_time = time() - push_time_start
        return push_time

    def _get_sample_from_segment(self, data_segment, i):
        valid = False
        while not valid:
            sample = np.random.uniform(i * data_segment, (i + 1) * data_segment)
            data, prob, idx, tree_idx = self._data.find(sample)
            if prob != 0:
                valid = True
        return data, prob, idx, tree_idx

    def sample(self, batch_size, beta):
        sample_time_start = time()
        p_total = self._data.total()
        data_segment = p_total / batch_size
        zip_data = [
            self._get_sample_from_segment(data_segment, i)
            for i in range(batch_size)
        ]
        data, prob, ids, tree_ids = zip(*zip_data)

        segment, actor_state, critic_state = zip(*data)
        segment = map(np.array, zip(*segment))
        actor_state = np.stack(actor_state, 0)
        critic_state = np.stack(critic_state, 0)
        sample = (segment, actor_state, critic_state)

        probs = np.array(prob) / p_total
        importance_weights = (self.capacity * probs) ** beta
        importance_weights /= importance_weights.max()
        result = (
            sample,
            tree_ids,
            importance_weights,
        )
        sample_time = time() - sample_time_start
        return result, sample_time

    def update_priorities(self, ids, priority_loss, alpha):
        update_time_start = time()
        priorities = np.power(priority_loss, alpha)
        [self._data.update(idx, priority) for idx, priority in zip(ids, priorities)]
        update_time = time() - update_time_start
        return update_time


class ExperienceReplay:
    # simple experience replay with the same methods and signatures as the prioritized version
    def __init__(
            self, capacity, segment_len,
            observation_shape, action_shape,
            actor_state_size, critic_state_size
    ):
        self.capacity = capacity
        self._index = 0
        self._full = False

        self._observations = np.zeros((capacity, segment_len + 1, sum(observation_shape)), dtype=np.float32)
        self._actions = np.zeros((capacity, segment_len, action_shape), dtype=np.float32)
        self._rewards = np.zeros((capacity, segment_len), dtype=np.float32)
        self._is_done = np.zeros((capacity, segment_len), dtype=np.float32)
        self._actor_states = np.zeros((capacity, actor_state_size), dtype=np.float32)
        self._critic_states = np.zeros((capacity, critic_state_size), dtype=np.float32)

    def push(self, sample, *args, **kwargs):
        push_time_start = time()
        segment, actor_state, critic_state = sample
        obs, action, reward, done = segment
        batch_size = obs.shape[0]
        for i in range(batch_size):
            # if not done[i][0]:  # prevent appending 'broken' segments
            # add data
            self._observations[self._index] = obs[i]
            self._actions[self._index] = action[i]
            self._rewards[self._index] = reward[i]
            self._is_done[self._index] = done[i]
            # add states
            self._actor_states[self._index] = actor_state[i]
            self._critic_states[self._index] = critic_state[i]
            # update index
            self._index = (self._index + 1) % self.capacity
            self._full = self._full or self._index == 0
        push_time = time() - push_time_start
        return push_time

    def sample(self, batch_size, *args, **kwargs):
        sample_time_start = time()
        max_id = self.capacity if self._full else self._index
        indices = np.random.randint(0, max_id, size=batch_size)
        segment = (
            self._observations[indices],
            self._actions[indices],
            self._rewards[indices],
            self._is_done[indices]
        )
        data = (
            segment,
            self._actor_states[indices],
            self._critic_states[indices]
        )
        result = (
            # data, tree_ids, importance_weights
            data, None, 1.0
        )
        sample_time = time() - sample_time_start
        return result, sample_time

    @staticmethod
    def update_priorities(*args, **kwargs):
        return 0
