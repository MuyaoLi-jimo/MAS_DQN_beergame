from collections import deque
import random
import numpy as np
import torch
import os

# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity, n_step=1, gamma=0.99,**kwargs):
        self.buffer = deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)
    
    def add(self, state, action, reward, next_state, done):
        """
        添加经验到缓冲区
        
        :param state: 当前状态
        :param action: 执行的动作
        :param reward: 获得的奖励
        :param next_state: 下一个状态
        :param done: 是否结束
        """
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < self.n_step:
            return
        
        state_n, action_n = self.n_step_buffer[0][:2]
        reward_n, next_state_n, done_n = self._get_n_step_info()
        self.buffer.append((state_n, action_n, reward_n, next_state_n, done_n))
        
        if done:
            self.n_step_buffer.clear()
    
    def _get_n_step_info(self):
        reward, next_state, done = 0.0, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        for idx, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            reward += (self.gamma ** idx) * r
            if d:
                break
        return reward, next_state, done
    
    def sample(self, batch_size, **kwargs):
        """
        从缓冲区采样一批经验
        
        :param batch_size: 批大小
        :return: 一批经验 (state, action, reward, next_state, done)
        """
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        """
        获取缓冲区当前大小
        
        :return: 缓冲区大小
        """
        return len(self.buffer)

# 定义优先经验回放缓冲区（支持 n-step）
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, n_step=1, gamma=0.99):
        """
        初始化缓冲区

        :param capacity: 缓冲区容量
        :param alpha: 控制优先级对采样概率的影响程度
        :param n_step: 多步回报步数
        :param gamma: 折扣因子
        """
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.alpha = alpha
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)

    def add(self, state, action, reward, next_state, done):
        """
        添加经验到缓冲区（支持 n-step）

        :param state: 当前状态
        :param action: 执行的动作
        :param reward: 获得的奖励
        :param next_state: 下一个状态
        :param done: 是否结束
        """
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.n_step:
            return

        state_n, action_n = self.n_step_buffer[0][:2]
        reward_n, next_state_n, done_n = self._get_n_step_info()

        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state_n, action_n, reward_n, next_state_n, done_n))
        else:
            self.buffer[self.pos] = (state_n, action_n, reward_n, next_state_n, done_n)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

        if done:
            self.n_step_buffer.clear()

    def _get_n_step_info(self):
        """
        计算 n 步累计奖励及终止状态
        """
        reward, next_state, done = 0.0, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        for idx, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            reward += (self.gamma ** idx) * r
            if d:
                break
        return reward, next_state, done

    def sample(self, batch_size, beta=0.4):
        """
        从缓冲区采样一批经验（按优先级概率）

        :param batch_size: 批大小
        :param beta: IS权重修正因子
        :return: (state, action, reward, next_state, done, indices, IS权重)
        """
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            states, actions, rewards, next_states, dones,
            indices, weights
        )

    def update_priorities(self, indices, priorities):
        """
        更新对应样本的优先级（通常用 TD误差）

        :param indices: 样本索引
        :param priorities: 新的优先级
        """
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        """
        获取缓冲区当前大小
        
        :return: 缓冲区大小
        """
        return len(self.buffer)

class WarmupIRScheduler:
    def __init__(self, optimizer, warmup_steps, base_lr):
        """
        optimizer: torch.optim.Optimizer 实例
        warmup_steps: 预热步数
        base_lr: 预热后要到达的学习率
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.last_step = 0

        # 将初始学习率设置为 0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 0.0

    def step(self):
        """每步调用一次，调整学习率"""
        self.last_step += 1
        if self.last_step <= self.warmup_steps:
            lr = self.base_lr * self.last_step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def get_lr(self):
        if self.last_step <= self.warmup_steps:
            return self.base_lr * self.last_step / self.warmup_steps
        else:
            return self.base_lr
        
def set_seed(seed, torch_deterministic=False):
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed