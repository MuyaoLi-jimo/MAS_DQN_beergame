import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import json
from pathlib import Path
from datetime import datetime
from env.env import Env
from beergame.utils.utils import ReplayBuffer
from beergame.model.mlp import QNetwork
from beergame.utils.visualize import plot_training_results,plot_test_results

# 定义DQN智能体
class DQNAgent:
    def __init__(self, state_size, action_size, firm_id, max_order=20, buffer_size=10000, batch_size=64, gamma=0.99, 
                 learning_rate=1e-3, tau=1e-3, update_every=4, dtype=torch.float32):
        """
        初始化DQN智能体
        
        :param state_size: 状态空间维度
        :param action_size: 动作空间维度
        :param firm_id: 企业ID，用于标识训练哪个企业
        :param max_order: 最大订单量，用于离散化动作空间
        :param buffer_size: 回放缓冲区大小
        :param batch_size: 批大小
        :param gamma: 折扣因子
        :param learning_rate: 学习率
        :param tau: 软更新参数
        :param update_every: 更新目标网络的频率
        """
        self.state_size = state_size
        self.action_size = action_size
        self.firm_id = firm_id
        self.max_order = max_order
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.learning_step = 0
        self.device = "cuda:0" if torch.cuda.is_available else "cpu"
        self.dtype = dtype
        
        # 创建Q网络和目标网络
        self.q_network = QNetwork(state_size, action_size).to(device=self.device , dtype=dtype)
        self.target_network = QNetwork(state_size, action_size).to(device=self.device , dtype=dtype)
        self.target_network.load_state_dict(self.q_network.state_dict()) #参数完全一致
        
        # 设置优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 创建经验回放缓冲区
        self.memory = ReplayBuffer(buffer_size,gamma=gamma)
        
        # 跟踪训练进度
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        """
        添加经验到回放缓冲区并按需学习
        
        :param state: 当前状态
        :param action: 执行的动作
        :param reward: 获得的奖励
        :param next_state: 下一个状态
        :param done: 是否结束
        """
        # 添加经验到回放缓冲区
        #state: np.ndarray(1,3)
        #next_state: np.ndarray(1,3)
        self.memory.add(state, action, reward, next_state, done)
        
        # 每隔一定步数学习
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
    
    def act(self, state, epsilon=0.0):
        """
        根据当前状态选择动作
        
        :param state: 当前状态
        :param epsilon: epsilon-贪婪策略参数
        :return: 选择的动作
        """
        # 从3维numpy数组转换为1维向量
        state = torch.from_numpy(state.flatten()).to(self.device,dtype=self.dtype).unsqueeze(0)
        
        # 切换到评估模式
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        # 切换回训练模式
        self.q_network.train()
        
        
        # epsilon-贪婪策略
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy()) + 1  # +1 因为我们的动作从1开始
        else:
            return random.randint(1, self.max_order)
    
    def learn(self, experiences):
        """
        从经验批次中学习
        
        :param experiences: (state, action, reward, next_state, done) 元组
        """
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        # 转换为torch张量
        states = torch.from_numpy(np.vstack([s.flatten() for s in states])).to(device=self.device,dtype=self.dtype)
        actions = torch.from_numpy(np.vstack([a-1 for a in actions])).long().to(device=self.device)  # -1 因为我们的动作从1开始，但索引从0开始
        rewards = torch.from_numpy(np.vstack(rewards)).to(device=self.device,dtype=self.dtype)
        next_states = torch.from_numpy(np.vstack([ns.flatten() for ns in next_states])).to(device=self.device,dtype=self.dtype)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).to(device=self.device,dtype=self.dtype)
        
        # 从目标网络获取下一个状态的最大预测Q值
        Q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        
        # 计算目标Q值
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # 获取当前Q值估计
        Q_expected = self.q_network(states).gather(1, actions)
        
        # 计算损失
        loss = nn.MSELoss()(Q_expected, Q_targets)
        
        # 最小化损失
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.learning_step += 1
        if self.learning_step % self.update_every == 0:
            self.soft_update()
        
        return loss.item()
    
    def soft_update(self):
        """
        软更新目标网络参数：θ_target = τ*θ_local + (1-τ)*θ_target
        """
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, filename):
        """
        保存模型参数
        
        :param filename: 文件名
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # 保存模型状态字典
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)
        print(f"模型已保存到 {filename}")
    
    def load(self, filename):
        """
        加载模型参数
        
        :param filename: 文件名
        """
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"从 {filename} 加载了模型")
            return True
        return False

def train_dqn(env, agent, checkpoint_dir, record_dir,num_episodes=1000, max_t=100, eps_start=1.0, eps_end=0.01, eps_decay=0.995, ):
    """
    训练DQN智能体
    
    :param env: 环境
    :param agent: DQN智能体
    :param num_episodes: 训练的episodes数量
    :param max_t: 每个episode的最大步数
    :param eps_start: 起始epsilon值
    :param eps_end: 最小epsilon值
    :param eps_decay: epsilon衰减率
    :return: 所有episode的奖励
    """
    scores = []  # 每个episode的总奖励
    eps = eps_start  # 初始epsilon值
    record_logger_file = record_dir / "train.json"
    
    for i_episode in range(1, num_episodes+1):
        state = env.reset()
        score = 0
        
        for t in range(max_t):
            # 对特定企业采取动作，其他企业随机决策
            actions = np.zeros((env.num_firms, 1))
            for firm_id in range(env.num_firms):
                if firm_id == agent.firm_id:
                    # 使用智能体策略
                    firm_state = state[firm_id].reshape(1, -1)
                    action = agent.act(firm_state, eps)
                    actions[firm_id] = action
                else:
                    # 对其他企业采取随机策略
                    actions[firm_id] = np.random.randint(1, 21)
            
            # 执行动作
            next_state, rewards, done = env.step(actions)
            
            # 该企业的奖励
            reward = rewards[agent.firm_id][0]
            
            # 保存经验并学习
            agent.step(state[agent.firm_id].reshape(1, -1), actions[agent.firm_id], reward, next_state[agent.firm_id].reshape(1, -1), done)
            
            # 更新状态和奖励
            state = next_state
            score += reward
            
            if done:
                break
        
        # 更新epsilon
        eps = max(eps_end, eps_decay * eps)
        
        # 记录分数
        scores.append(score)
        
        # 输出进度
        if i_episode % 100 == 0:
            print(f'Episode {i_episode}/{num_episodes} | Average Score: {np.mean(scores[-100:]):.2f} | Epsilon: {eps:.4f}')
        
        # 每隔一定episode保存模型
        if i_episode % 500 == 0:
            agent.save( checkpoint_dir / f'dqn_agent_firm_{agent.firm_id}_episode_{i_episode}.pth')
            with open(record_logger_file, 'w', encoding="utf-8") as json_file:
                json.dump(scores, json_file, indent=4, ensure_ascii=False)
    
    # 训练结束后保存最终模型
    agent.save(checkpoint_dir / f'dqn_agent_firm_{agent.firm_id}_final.pth')
    with open(record_logger_file, 'w', encoding="utf-8") as json_file:
        json.dump(scores, json_file, indent=4, ensure_ascii=False)
    
    return scores

def test_agent(env, agent, record_dir:Path, num_episodes=10):
    """
    测试训练好的DQN智能体
    
    :param env: 环境
    :param agent: 训练好的DQN智能体
    :param num_episodes: 测试的episodes数量
    :return: 所有episode的奖励和详细信息
    """
    record_dir = Path(record_dir)
    record_dir.mkdir(parents=True,exist_ok=True)
    record_logger_file = record_dir / "valid.json"
    
    scores = []
    inventory_history = []
    orders_history = []
    demand_history = []
    satisfied_demand_history = []
    
    for i_episode in range(1, num_episodes+1):
        state = env.reset()
        score = 0
        episode_inventory = []
        episode_orders = []
        episode_demand = []
        episode_satisfied_demand = []
        
        for t in range(env.max_steps):
            # 对特定企业采取动作，其他企业随机决策
            actions = np.zeros((env.num_firms, 1))
            for firm_id in range(env.num_firms):
                if firm_id == agent.firm_id:
                    # 使用智能体策略，不使用探索
                    firm_state = state[firm_id].reshape(1, -1)
                    action = agent.act(firm_state, epsilon=0.0)
                    actions[firm_id] = action
                else:
                    # 对其他企业采取随机策略
                    actions[firm_id] = np.random.randint(1, 21)
            
            # 执行动作
            next_state, rewards, done = env.step(actions)
            
            # 记录关键指标
            episode_inventory.append(env.inventory[agent.firm_id][0])
            episode_orders.append(actions[agent.firm_id][0])
            episode_demand.append(env.demand[agent.firm_id][0])
            episode_satisfied_demand.append(env.satisfied_demand[agent.firm_id][0])
            
            # 该企业的奖励
            reward = rewards[agent.firm_id][0]
            score += reward
            
            # 更新状态
            state = next_state
            
            if done:
                break
        
        # 记录分数和历史数据
        scores.append(score)
        inventory_history.append(episode_inventory)
        orders_history.append(episode_orders)
        demand_history.append(episode_demand)
        satisfied_demand_history.append(episode_satisfied_demand)
        
        print(f'Test Episode {i_episode}/{num_episodes} | Score: {score:.2f}')
    print(f"Average Score: {np.mean(scores)}")
    with open(record_logger_file, 'w', encoding="utf-8") as json_file:
        json.dump(scores, json_file, indent=4, ensure_ascii=False)
    
    return scores, inventory_history, orders_history, demand_history, satisfied_demand_history

if __name__ == "__main__":
    # 创建保存模型和图表的目录
    plt.rcParams['font.sans-serif'] = ['Source Han Sans SC']  # 指定中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 绘图显示负号
    
    # 初始化环境参数
    num_firms = 3  # 假设有3个企业
    p = [10, 9, 8]  # 价格列表
    h = 0.5  # 库存持有成本
    c = 2  # 损失销售成本
    initial_inventory = 100  # 初始库存
    poisson_lambda = 10  # 泊松分布的均值
    max_steps = 100  # 每个episode的最大步数
    seed = 1


    
    # 创建仿真环境
    env = Env(num_firms, p, h, c, initial_inventory, poisson_lambda, max_steps)
    
    # 为第二个企业创建DQN智能体
    firm_id = 0  # 选择第二个企业进行训练
    state_size = 3  # 每个企业的状态维度：订单、满足的需求和库存
    action_size = 20  # 假设最大订单量为20
    
    # 创建保存模型和图表的目录
    record_dir = Path(f"record/{firm_id}-dqn-{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}")
    checkpoint_dir = record_dir / 'checkpoints'
    record_dir = record_dir /  'visualize'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    
    agent = DQNAgent(state_size=state_size, action_size=action_size, firm_id=firm_id, max_order=action_size,)
    
    # 训练DQN智能体
    scores = train_dqn(env, agent, checkpoint_dir=checkpoint_dir,record_dir=record_dir, num_episodes=2000, max_t=max_steps, eps_start=1.0, eps_end=0.01, eps_decay=0.995,)

    # 绘制训练结果
    plot_training_results(scores,record_dir=record_dir)
    
    # 测试训练好的智能体
    test_scores, inventory_history, orders_history, demand_history, satisfied_demand_history = test_agent(env, agent, record_dir=record_dir,num_episodes=10)
    
    # 绘制测试结果
    plot_test_results(test_scores, record_dir,inventory_history, orders_history, demand_history, satisfied_demand_history)
