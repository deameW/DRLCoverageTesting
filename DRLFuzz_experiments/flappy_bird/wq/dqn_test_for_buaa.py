import json
import os
import random
import time

import numpy as np
import torch
from pygame import K_w
from scipy import spatial
from torch import nn

from DRLFuzz_experiments.flappy_bird.wq.dqn_model import DQN
from DRLFuzz_experiments.flappy_bird.wq.env.flappy_bird_env import testFlappybird, env


# from dqn_model import DQN


def convert_to_float(obj):
    if isinstance(obj, list):
        return [float(item) for item in obj]
    return obj


def test_model(initial_state, mutate=False):
    score_all_episodes = 0
    # episode_data = []
    failedEpisode = 0

    dqn = DQN()

    env.init()
    env.reset_game()

    # for i_episode in range(iteration):
    ep_r = 0
    # episode_states = []
    env.reset_game()
    tmp = initial_state
    # Call TestFlappybird to initialize the game if initial states are given
    if initial_state is not None:
        s_ = initial_state
        testFlappybird._init(s_[0], s_[1], s_[2], s_[3])

        t_buaa = env.getGameState()
        s_buaa =  [
        t_buaa['player_y'],
        t_buaa['player_vel'],
        t_buaa['player_y'] - t_buaa['next_pipe_bottom_y'],
        t_buaa['player_y'] - t_buaa['next_pipe_top_y'],
        t_buaa['next_pipe_dist_to_player'],
        t_buaa['player_y'] - t_buaa['next_next_pipe_bottom_y'],
        t_buaa['player_y'] - t_buaa['next_next_pipe_top_y'],
        t_buaa['next_next_pipe_dist_to_player'],
    ]

        t = list(env.getGameState().values())
        s_ = t
        # episode_states.append(convert_to_float([t[3], t[6], t[2] - 309, t[1]]))
    # Let game picks initial state randomly
    else:
        s_ = list(env.getGameState().values())
        # episode_states.append(convert_to_float([s_[3], s_[6], s_[2] - 309, s_[1]]))

    while True:
        dist = 0
        curr = ((int(t_buaa['player_y'] - s_buaa[2]), int(t_buaa['player_y'] - s_buaa[5]), int(s_buaa[3] - 309), int(s_buaa[0])))
        for i in range(len(tmp)):
            dist += (tmp[i] - curr[i]) ** 2
        dist = dist ** 0.5
        if dist > innerDelta:
            tmp = curr
            if getDistance(curr) > delta:  # 计算S' 到State Pool中状态的距离
                allStates.add(curr)

        # Choose an action by trained model
        s = s_
        a = dqn.choose_action(s, 1)
        ac = K_w if a else None
        r = env.act(ac)

        # time.sleep(0.01)

        # Record the new state returned from environment
        s_ = list(env.getGameState().values())
        # episode_states.append(convert_to_float([s_[3], s_[6], s_[2] - 309, s_[1]]))

        # To tell if the game is over
        done = env.game_over()
        ep_r += r
        if done or  env.score() > 10:
            break
            # Record the information of the episode
            # score_all_episodes += ep_r
            # print('Ep: ', i_episode, '| Ep_r: ', round(
            #     ep_r, 2))
            # if ep_r <= 2:
            #     failedEpisode += 1
            # break
    # episode_data = {
    #     "initial_state": initial_state,
    #     "avg_award_over_episodes": score_all_episodes / iteration
    # }
    # print("Initial State [{}]avg reward over {} episodes: ".format(initial_state, iteration), score_all_episodes / iteration)
    # Return the average score under a given initial state
    return env.score() + 5

def test_model2(initial_state, mutate=False):
    score_all_episodes = 0
    # episode_data = []
    failedEpisode = 0

    dqn = DQN()

    env.init()
    env.reset_game()

    # for i_episode in range(iteration):
    ep_r = 0
    # episode_states = []
    env.reset_game()
    tmp = initial_state
    # Call TestFlappybird to initialize the game if initial states are given
    if initial_state is not None:
        s_ = initial_state
        testFlappybird._init(s_[0], s_[1], s_[2], s_[3])

        t_buaa = env.getGameState()
        s_buaa =  [
        t_buaa['player_y'],
        t_buaa['player_vel'],
        t_buaa['player_y'] - t_buaa['next_pipe_bottom_y'],
        t_buaa['player_y'] - t_buaa['next_pipe_top_y'],
        t_buaa['next_pipe_dist_to_player'],
        t_buaa['player_y'] - t_buaa['next_next_pipe_bottom_y'],
        t_buaa['player_y'] - t_buaa['next_next_pipe_top_y'],
        t_buaa['next_next_pipe_dist_to_player'],
    ]

        t = list(env.getGameState().values())
        s_ = t
        # episode_states.append(convert_to_float([t[3], t[6], t[2] - 309, t[1]]))
    # Let game picks initial state randomly
    else:
        s_ = list(env.getGameState().values())
        # episode_states.append(convert_to_float([s_[3], s_[6], s_[2] - 309, s_[1]]))

    while True:
        dist = 0
        curr = ((int(t_buaa['player_y'] - s_buaa[2]), int(t_buaa['player_y'] - s_buaa[5]), int(s_buaa[3] - 309), int(s_buaa[0])))
        for i in range(len(tmp)):
            dist += (tmp[i] - curr[i]) ** 2
        dist = dist ** 0.5
        if dist > innerDelta:
            tmp = curr
            if getDistance(curr) > delta:  # 计算S' 到State Pool中状态的距离
                allStates.add(curr)

        # Choose an action by trained model
        s = s_
        a = dqn.choose_action(s, 1)
        ac = K_w if a else None
        r = env.act(ac)

        # time.sleep(0.01)

        # Record the new state returned from environment
        s_ = list(env.getGameState().values())
        # episode_states.append(convert_to_float([s_[3], s_[6], s_[2] - 309, s_[1]]))

        # To tell if the game is over
        done = env.game_over()
        ep_r += r
        if done or  env.score() > 10:
            break
            # Record the information of the episode
            # score_all_episodes += ep_r
            # print('Ep: ', i_episode, '| Ep_r: ', round(
            #     ep_r, 2))
            # if ep_r <= 2:
            #     failedEpisode += 1
            # break
    episode_data = {
        "initial_state": initial_state,
        "avg_award_over_episodes": score_all_episodes
    }
    # print("Initial State [{}]avg reward over {} episodes: ".format(initial_state, iteration), score_all_episodes / iteration)
    # Return the average score under a given initial state
    return episode_data


""" 
    state的各个维度：
    'player_y': 256, 
    'player_vel': 26, 
    'next_pipe_dist_to_player': 272.0, 
    'next_pipe_top_y': 453,  1  下一个管子的top y
    'next_pipe_bottom_y': 553,  
    'next_next_pipe_dist_to_player': 416.0,   会决定小鸟的初始位置距离第一根管子多远
    'next_next_pipe_top_y': 309, 2
     'next_next_pipe_bottom_y': 409
"""

"""
    TODO
    8维度和4维度state如何转换？
    [152, 158, -110, -5] s
    [256, -5, 199.0, 152, 252, 343.0, 158, 258] t
    
    s = t[
        t[3],
        t[6],
        t[2] - 309
        t[1]
    ]
"""

save_path = "./test/episode_info_random_start.json"


def testInitialState(iteration, state):
    return test_model2(state, iteration)  # Random Initial State


def randomGenerate():
    pipe1 = random.randint(25, 192)
    pipe2 = random.randint(25, 192)
    dist = random.randint(-120, -75)
    vel = random.randint(-56, 10)
    return [pipe1, pipe2, dist, vel]


resultNum = list()
allStates = set()
kdTree = None
delta = 20
innerDelta = 20


import torch.nn.functional as F


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

path = "./model/wq_flappy_bird_dqn_1w.pkl"
net = model().cuda()
net.load_state_dict(torch.load(path))

def mutator(arg, l):
    pipe1 = arg[0]
    pipe2 = arg[1]
    dist = arg[2]
    vel = arg[3]
    env.reset_game()
    testFlappybird._init(pipe1, pipe2, dist, vel)
    t = env.getGameState()
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
    r = env.act(env.getActionSet()[a])
    if r == 0:
        reward += 1
    elif r > 0:
        reward += 5
    else:
        reward += -10

    t = env.getGameState()
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
            score[i] = test_model(statePool[i])
            if score[i] < theta:
                # 把满足要求的初始化的State放入到resultPool中(分数)
                tmp = [int(x) for x in statePool[i]]
                if tuple(tmp) not in resultPool:
                    with open("./buaa_result.txt", 'a') as f:
                        for j in range(len(tmp)):
                            f.write(str(tmp[j]) + ' ')
                        f.write('\n')
                resultPool.add(tuple(tmp))
        # 在初始化的state跑episode的过程中，所有过程中满足覆盖率distance的states组成了kd树
        kdTree = spatial.KDTree(data=np.array(list(allStates)), leafsize=10000)
        print("iteration {} failed cases num:{}".format(k + 1, len(resultPool)))
        # resultNum.c(len(resultPool))
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



if __name__ == '__main__':
    # dump_info = []
    # for i in range(100):
    #     episode_data = testInitialState(1, randomGenerate())
    #     dump_info.append(episode_data)
    #
    # if save_path != "":
    #     with open(save_path, 'w') as f:
    #         json.dump(dump_info, f, default=convert_to_float)

    # test_model(100, randomGenerate())
    start_time = time.time()
    if os.path.exists("./buaa_result.txt"):
        os.remove("./buaa_result.txt")
    result = DRLFuzz(100, 1000, 10, 0.1, 2, True)

    end_time = time.time()
    print("total; time: ", end_time - start_time)
    print("result num: ", len(result))
