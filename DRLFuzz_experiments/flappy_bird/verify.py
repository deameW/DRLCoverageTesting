import time

import torch
from pygame import K_w
from torch import nn
import torch.nn.functional as F

from DRLFuzz_experiments.flappy_bird.wq.dqn_test import test_model, randomGenerate
from DRLFuzz_experiments.flappy_bird.wq.env.flappy_bird_env import testFlappybird, env

# from DRLFuzz_experiments.flappy_bird.wq.dqn_test import test_model, randomGenerate
# from test_flappy_bird import TestFlappyBird
from ple import PLE
import numpy as np
import random
import os

# os.environ['SDL_VIDEODRIVER'] = "dummy"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


# class model(nn.Module):
#     def __init__(self, ):
#         super(model, self).__init__()
#         self.fc1 = nn.Linear(8, 128)
#         self.fc1.weight.data.normal_(0, 0.1)  # initialization
#
#         self.fc2 = nn.Linear(128, 128)
#         self.fc2.weight.data.normal_(0, 0.1)  # initialization
#
#         self.fc3 = nn.Linear(128, 64)
#         self.fc3.weight.data.normal_(0, 0.1)  # initialization
#
#         self.out = nn.Linear(64, 2)
#         self.out.weight.data.normal_(0, 0.1)  # initialization
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.tanh(x)
#         x = self.fc2(x)
#         x = F.tanh(x)
#         x = self.fc3(x)
#         x = F.tanh(x)
#         actions_value = self.out(x)
#         return actions_value
#
#
# def test(net, arg):
#     pipe1 = arg[0]
#     pipe2 = arg[1]
#     dist = arg[2]
#     vel = arg[3]
#     env.reset_game()
#     testFlappybird._init(pipe1, pipe2, dist, vel)
#     s_ = list(env.getGameState().values())
#     while True:
#         s = s_
#         s = torch.unsqueeze(torch.FloatTensor(s), 0)
#
#         # s = torch.tensor(s, dtype=torch.float32, requires_grad=False).cuda()
#         actions_value = net.forward(s)
#         print(actions_value)
#         a = torch.max(actions_value, 1)[1].data.numpy()
#
#         # a = actions_value.index(max(actions_value))
#         # pred = net(s).cpu().detach().numpy()
#         # pred = net(s).cpu().detach().numpy()
#         # a = np.argmax(pred)
#         print(a)
#         ac = K_w if a else None
#         env.act(ac)
#         if testFlappybird.game_over():
#             break
#         t = env.getGameState()
#         s = [
#             t['player_y'],
#             t['player_vel'],
#             t['player_y'] - t['next_pipe_bottom_y'],
#             t['player_y'] - t['next_pipe_top_y'],
#             t['next_pipe_dist_to_player'],
#             t['player_y'] - t['next_next_pipe_bottom_y'],
#             t['player_y'] - t['next_next_pipe_top_y'],
#             t['next_next_pipe_dist_to_player'],
#         ]
#         # s = t
#         time.sleep(0.01)
#
#     return env.score() + 5
#
#
# def verify(net, num):
#     # random.seed(2003511)
#     # scores = list()
#     # case = np.loadtxt(casePath)
#     # for c in case:
#     #     score = test(net, (c[0], c[1], c[2], c[3]))
#     #     scores.append(score)
#     # scores = np.array(scores)
#     # print([s for s in scores])
#     # print("Bad Cases mean:{} max{} min{} std{}".format(np.mean(scores), np.max(scores), np.min(scores), np.std(scores)))
#     random.seed(int(time.time()))
#     scores = list()
#     for i in range(num):
#         pipe1 = random.randint(25, 192)
#         pipe2 = random.randint(25, 192)
#         dist = random.randint(-120, -75)
#         vel = random.randint(-56, 10)
#         # score = test(net, [pipe1, pipe2, dist, vel])
#         # score = test(net, [83, 165, -117, -29])
#         score = test_model(1000, "xx.json", [83, 165, -117, -29], False)
#         scores.append(score)
#     scores = np.array(scores)
#     print([s for s in scores])
#     print("Random Cases mean:{} max{} min{} std{}".format(np.mean(scores), np.max(scores), np.min(scores),
#                                                           np.std(scores)))


# modelPath = "../result/flappy_bird/model/flappy_bird_model.pkl"
# fixedModelPath = "../result/flappy_bird/model/flappy_bird_model_repaired.pkl"
myDQNPath = "./wq/model/wq_flappy_bird_dqn_1w.pkl"
# casePath = "../result/flappy_bird/result_DRLFuzz.txt"
# testFlappybird = TestFlappyBird()
# p = PLE(testFlappybird, fps=30, display_screen=True)
# p.init()
# gamma = 0.9

if __name__ == '__main__':
    # print("Before retraining")
    # net = torch.load(myDQNPath).cuda().eval()
    # net = torch.load(myDQNPath)
    # eval_net = model()
    # eval_net.load_state_dict(torch.load(myDQNPath))
    # verify(eval_net, 1000)
    # print("After retraining")
    # net = torch.load(fixedModelPath).cuda().eval()
    # verify(net, 1000)
    # test(net, [74, 190, -102, -40])
    test_model(100, "./test/episode_info_random_start.json", randomGenerate())  # Random Initial State
