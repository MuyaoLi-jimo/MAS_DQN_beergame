import torch
import torch.nn as nn

# 定义Q网络模型
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, **kwargs):
        """
        初始化Q网络
        
        :param state_size: 状态空间维度
        :param action_size: 动作空间维度
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        """
        前向传播
        
        :param state: 输入状态
        :return: 各动作的Q值
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)



# 定义Q网络模型
class QNetworkPlus(nn.Module):
    def __init__(self, state_size, action_size, **kwargs):
        """
        初始化Q网络
        
        :param state_size: 状态空间维度
        :param action_size: 动作空间维度
        """
        super(QNetworkPlus, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        """
        前向传播
        
        :param state: 输入状态
        :return: 各动作的Q值
        """
        x = torch.nn.functional.silu(self.fc1(state))
        x = torch.nn.functional.silu(self.fc2(x))
        return self.fc3(x)
    

class MultiAgentQNetworkPlus(nn.Module):
    '''
    description: 
    return {*}
    '''
    def __init__(self, state_size, action_size,firm_num,decive,):
        """
        初始化Q网络
        
        :param state_size: 状态空间维度
        :param action_size: 动作空间维度
        """
        super(MultiAgentQNetworkPlus, self).__init__()
        self.decive = decive
        self.firm_emb = nn.Embedding(firm_num,state_size)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state, firm_id:int = 0):
        """
        前向传播
        
        :param state: 输入状态
        :return: 各动作的Q值
        """
        if isinstance(firm_id, int):
            firm_id_tensor = torch.tensor(firm_id, dtype=torch.long, device=self.device)
        elif isinstance(firm_id, torch.Tensor):
            firm_id_tensor = firm_id.to(dtype=torch.long, device=self.device)
        else:
            raise TypeError(f"firm_id must be an int or a torch.Tensor, but got {type(firm_id)}")

        firm_x = self.firm_emb(firm_id_tensor)
        x = firm_x + state
        x = torch.nn.functional.silu(self.fc1(x))
        x = torch.nn.functional.silu(self.fc2(x))
        return self.fc3(x)