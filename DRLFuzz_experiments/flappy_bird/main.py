import time

import torch
from torch import nn
import torch.nn.functional as F

from DRLFuzz_experiments.flappy_bird.test_flappy_bird import TestFlappyBird
from ple import PLE
import numpy as np
import random
import os
from scipy import spatial
import sys
from ple.games.flappybird import FlappyBird

sys.setrecursionlimit(10000000)
os.environ['SDL_VIDEODRIVER'] = "dummy"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class model(nn.Module):
    def __init__(self, ):
        super(model, self).__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization

        self.fc2 = nn.Linear(128, 128)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization

        self.fc3 = nn.Linear(128, 64)
        self.fc3.weight.data.normal_(0, 0.1)  # initialization

        self.out = nn.Linear(64, 2)
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


"""
用一个初始state 去跑一个episode, 返回这局游戏的总分数
"""


def test(arg):
    pipe1 = arg[0]
    pipe2 = arg[1]
    dist = arg[2]
    vel = arg[3]
    p.reset_game()
    testFlappybird._init(pipe1, pipe2, dist, vel)
    t = p.getGameState()  # TODO 这里的信息是输入初始状态环境返回的状态？    —— no，是init state的详细信息
    s = [
        t['player_y'],
        t['player_vel'],
        t['player_y'] - t['next_pipe_bottom_y'],
        t['player_y'] - t['next_pipe_top_y'],
        t['next_pipe_dist_to_player'],
        t['player_y'] - t['next_next_pipe_bottom_y'],
        t['player_y'] - t['next_next_pipe_top_y'],
        t['next_next_pipe_dist_to_player'],
    ]
    print(s)
    tmp = arg
    while True:
        dist = 0
        curr = ((int(t['player_y'] - s[2]), int(t['player_y'] - s[5]), int(s[3] - 309), int(s[0])))  # TODO 为啥要再算一遍
        # 计算新的State和初始state的距离， TODO 这个用算吗
        for i in range(len(tmp)):
            dist += (tmp[i] - curr[i]) ** 2
        dist = dist ** 0.5
        if dist > innerDelta:
            tmp = curr
            if getDistance(curr) > delta:  # 计算S' 到State Pool中状态的距离
                allStates.add(curr)
        s = torch.tensor(s, dtype=torch.float32, requires_grad=False).cuda()
        pred = net(s).cpu().detach().numpy()
        a = np.argmax(pred)
        p.act(p.getActionSet()[a])

        # time.sleep(0.02)
        if testFlappybird.game_over() or p.score() > 10:
            break
        t = p.getGameState()
        s = [
            t['player_y'],
            t['player_vel'],
            t['player_y'] - t['next_pipe_bottom_y'],
            t['player_y'] - t['next_pipe_top_y'],
            t['next_pipe_dist_to_player'],
            t['player_y'] - t['next_next_pipe_bottom_y'],
            t['player_y'] - t['next_next_pipe_top_y'],
            t['next_next_pipe_dist_to_player'],
        ]
    return p.score() + 5


def mutator(arg, l):
    pipe1 = arg[0]
    pipe2 = arg[1]
    dist = arg[2]
    vel = arg[3]
    p.reset_game()
    testFlappybird._init(pipe1, pipe2, dist, vel)
    t = p.getGameState()
    s = [
        t['player_y'],
        t['player_vel'],
        t['player_y'] - t['next_pipe_bottom_y'],
        t['player_y'] - t['next_pipe_top_y'],
        t['next_pipe_dist_to_player'],
        t['player_y'] - t['next_next_pipe_bottom_y'],
        t['player_y'] - t['next_next_pipe_top_y'],
        t['next_next_pipe_dist_to_player'],
    ]
    s = torch.tensor(s, dtype=torch.float32, requires_grad=False).cuda()
    pred = net.forward(s)
    a = torch.argmax(pred)
    reward = 0
    if s[1] < 0 and s[2] > 0:
        reward += 1
    else:
        reward += -2
    r = p.act(p.getActionSet()[a])
    if r == 0:
        reward += 1
    elif r > 0:
        reward += 5
    else:
        reward += -10

    t = p.getGameState()
    s_1 = [
        t['player_y'],
        t['player_vel'],
        t['player_y'] - t['next_pipe_bottom_y'],
        t['player_y'] - t['next_pipe_top_y'],
        t['next_pipe_dist_to_player'],
        t['player_y'] - t['next_next_pipe_bottom_y'],
        t['player_y'] - t['next_next_pipe_top_y'],
        t['next_next_pipe_dist_to_player'],
    ]
    s_1 = torch.tensor(s_1, dtype=torch.float32, requires_grad=False).cuda()
    q_1 = net(s_1)
    label = pred.clone().detach()  # 状态经过Q网络模型预测得到的结果。
    label[a] = r + 0.9 * torch.max(q_1)  # TD target

    s.requires_grad = True
    net.zero_grad()  # 清除梯度
    criterion = torch.nn.MSELoss()
    loss = criterion(net(s), label)
    loss.backward()
    grad = s.grad.cpu().numpy()
    pipe1 = pipe1 - grad[1] * l
    pipe2 = pipe2 - grad[4] * l
    dist = dist + grad[3] * l
    vel = vel + grad[0] * l
    pipe1 = max(min(pipe1, 192), 25)
    pipe2 = max(min(pipe2, 192), 25)
    dist = max(min(dist, -75), -120)
    vel = max(min(vel, 10), -56)
    return [pipe1, pipe2, dist, vel]


"""
    @params: num生成初始states个数、n迭代次数、l学习率、theta控制目标函数reward阈值、episode得分阈值、覆盖率
"""


def DRLFuzz(num, n, l, alpha, theta, coverage):
    global kdTree
    statePool = list()
    score = list()
    resultPool = set()
    for _ in range(num):  # 随机初始化num个初始states
        s = randFun(coverage)
        statePool.append(s)
        score.append(0)
    # 迭代n次
    for k in range(n):
        for i in range(num):
            # 每次迭代随机初始化states的episode reward
            score[i] = test(statePool[i])
            if score[i] < theta:
                # 把满足要求的初始化的State放入到resultPool中(分数)
                tmp = [int(x) for x in statePool[i]]
                if tuple(tmp) not in resultPool:
                    with open(savePath, 'a') as f:
                        for j in range(len(tmp)):
                            f.write(str(tmp[j]) + ' ')
                        f.write('\n')
                resultPool.add(tuple(tmp))
        # 在初始化的state跑episode的过程中，所有过程中满足覆盖率distance的states组成了kd树
        kdTree = spatial.KDTree(data=np.array(list(allStates)), leafsize=10000)
        print("iteration {} failed cases num:{}".format(k + 1, len(resultPool)))
        resultNum.c(len(resultPool))
        idx = sorted(range(len(score)), key=lambda x: score[x])  # 得到score中元素排序后的对应索引（从小到大）
        for i in range(num):
            if i < int(num * alpha):
                # 对reward最小的进行变异
                st = mutator(statePool[idx[i]], l)
                if st != statePool[idx[i]]:
                    statePool[idx[i]] = st
                else:
                    statePool[idx[i]] = randFun(coverage)
            else:
                statePool[idx[i]] = randFun(coverage)
    print("cresultPool len: ", len(resultPool))
    return resultPool


"""
随机初始化一个initial state
"""


def randFun(coverage):
    if coverage:
        global delta
        count = 0
        while True:
            pipe1 = random.randint(25, 192)  # 到下一个管道A的距离
            pipe2 = random.randint(25, 192)  # 下个管道A到下个管道B的距离
            dist = random.randint(-120, -75)  # 小鸟到管道A的距离
            vel = random.randint(-56, 10)  # 小鸟的垂直速度
            count += 1
            if count == 10000:
                delta *= 0.9
            if getDistance((pipe1, pipe2, dist, vel)) > delta:  # 最近距离， 超过10000个初始state还没有达到距离要求，距离会变小
                allStates.add((pipe1, pipe2, dist, vel))
                break
    else:
        pipe1 = random.randint(25, 192)
        pipe2 = random.randint(25, 192)
        dist = random.randint(-120, -75)
        vel = random.randint(-56, 10)
    return [pipe1, pipe2, dist, vel]


def getDistance(arg):
    if kdTree is None:
        return np.inf
    else:
        dist, _ = kdTree.query(np.array(list(arg)))
        return dist


MEMORY_CAPACITY = 20000
FlappyBird = FlappyBird()

testFlappybird = TestFlappyBird()
path = "./wq/model/wq_flappy_bird_dqn_1w.pkl"
net = model().cuda()
net.load_state_dict(torch.load(path))
savePath = "../result/flappy_bird/result_DRLFuzz.txt"
p = PLE(testFlappybird, fps=30, display_screen=True, force_fps=True)
p.init()
resultNum = list()
allStates = set()
random.seed(2003511)
kdTree = None
delta = 20
innerDelta = 20

if __name__ == '__main__':
    start_time = time.time()
    if os.path.exists(savePath):
        os.remove(savePath)
    result = DRLFuzz(100, 1000, 10, 0.1, 2, True)

    end_time = time.time()
    print("total; time: ", end_time - start_time )
    print("result num: ", len(result))
