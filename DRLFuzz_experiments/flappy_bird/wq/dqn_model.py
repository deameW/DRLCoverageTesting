import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

N_ACTIONS = 2  # env.action_space.n
N_STATES = 8  # env.observation_space.shape[0]

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.0001  # learning rate
EPSILON = 0.95  # greedy policy
GAMMA = 0.995  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 20000

class Net(nn.Module):

    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 128)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization

        self.fc2 = nn.Linear(128, 128)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization

        self.fc3 = nn.Linear(128, 64)
        self.fc3.weight.data.normal_(0, 0.1)  # initialization

        self.out = nn.Linear(64, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = self.fc3(x)
        x = F.tanh(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0
        self.f1 = "./model/wq_flappy_bird_dqn_1w.pkl"  # for storing memory
        # self.f1 = "../../result/flappy_bird/model/flappy_bird_model.pkl"  # for storing memory
        self.f2 = "./model/wq_flappy_bird_dqn_target_1w.pkl"
        self.f3 = "./data/train_data.xlsx"
        # self.f2 = "../../result/flappy_bird/model/flappy_bird_model.pkl"
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.load_model()

    def __del__(self):
        pass
        # self.save_model()

    def save_model(self):
        print('save model----')
        torch.save(self.eval_net.state_dict(), self.f1)
        torch.save(self.target_net.state_dict(), self.f2)
        print('save model')

    def load_model(self):
        print("load model----")
        if os.path.exists(self.f1):
            self.eval_net.load_state_dict(torch.load(self.f1))
            self.target_net.load_state_dict(torch.load(self.f2))
            print('load model')

    def choose_action(self, x, e=EPSILON):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < e:  # greedy
            actions_value = self.eval_net.forward(x)
            print(actions_value)
            action = torch.max(actions_value, 1)[1].data.numpy()
        else:  # random
            action = np.random.randint(0, N_ACTIONS)

        return action

    def store_transition(self, s, a, r, s_, isdone=False):
        self.store_train_data(s, a, r, s_, isdone)

        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def store_train_data(self, s, a, r, s_, isdone):
        if isinstance(a, np.ndarray):
            a = a.tolist()[0] if a.size > 1 else a.item()
        row = [s, a, r, s_, isdone]

        # 写入数据到 Excel 文件
        with open(self.f3, "a+") as file:
            row_data = "\t".join(map(str, row))  # 用制表符分隔每个变量
            file.write(row_data + "\n")  # 写入每行数据

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]

        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        # print('b_a',b_a)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()