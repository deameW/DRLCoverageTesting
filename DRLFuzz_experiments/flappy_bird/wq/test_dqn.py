import json
import random
import time

import numpy as np
import torch
from pygame import K_w

from DRLFuzz_experiments.flappy_bird.test_flappy_bird import TestFlappyBird
from my_dqn import DQN
from ple import PLE

seed = int(time.time())
print(seed)

testFlappybird = TestFlappyBird()
env = PLE(testFlappybird, fps=30, display_screen=True, force_fps=True, rng= seed)

def convert_to_float(obj):
    if isinstance(obj, list):
        return [float(item) for item in obj]
    return obj


def test_model(iteration, save_path, initial_state, mutate=False):
    score_all_episodes = 0
    episode_data = []

    dqn = DQN()

    env.init()

    for i_episode in range(iteration):
        ep_r = 0
        episode_states = []
        env.reset_game()

        # 如果有初始状态，则模拟与环境的交互来设置初始状态
        if initial_state is not None:
            s_ = initial_state
            testFlappybird._init(int(s_[3]), int(s_[6]), int(s_[5] - 309), int(s_[1]))
            t = env.getGameState()
            episode_states.append(convert_to_float(s_))

        else:
            s_ = list(env.getGameState().values())
            episode_states.append(convert_to_float(s_))

        while True:
            s = s_

            # 根据当前状态选择动作
            a = dqn.choose_action(s, 1)
            ac = K_w if a else None

            # time.sleep(0.02)

            # 执行动作并观察奖励和下一个状态
            r = env.act(ac)
            s_ = list(env.getGameState().values())

            time.sleep(0.1)

            episode_states.append(convert_to_float(s_))

            # 判断是否结束当前游戏
            done = env.game_over()
            ep_r += r

            if done:
                # time.sleep(3)
                # print(len(episode_states),"        " ,episode_states[-15])
                # 记录本局游戏的数据
                episode_data.append({
                    "initial_state": episode_states[0],
                    "last_state_before_failure": episode_states[-1],
                    "total_reward": ep_r
                })
                score_all_episodes += ep_r
                print('Ep: ', i_episode, '| Ep_r: ', round(ep_r, 2))
                break

    # 将数据保存到 JSON 文件中
    if save_path != "":
        with open(save_path, 'w') as f:
            json.dump(episode_data, f, default=convert_to_float)

    print("Average score: ", score_all_episodes / iteration)

    if mutate:
        return score_all_episodes / iteration


"""
    next_next_pipe_bottom_y
    next_next_pipe_dist_to_player
    next_next_pipe_top_y 2
    next_pipe_bottom_y
    next_pipe_dist_to_player 3
    next_pipe_top_y 1
    player_vel 4
    player_y
    
    'player_y': 256, 
    'player_vel': 26, 
    'next_pipe_dist_to_player': 272.0, 
    'next_pipe_top_y': 453,  1  下一个管子的top y
    'next_pipe_bottom_y': 553,  
    'next_next_pipe_dist_to_player': 416.0,   会决定小鸟的初始位置距离第一根管子多远
    'next_next_pipe_top_y': 309, 2
     'next_next_pipe_bottom_y': 409
"""

if __name__ == '__main__':
    # test_model(10, "./test/states_generic.json", None)  # Random
    # test_model(100, "./test/states_generic.json",   [100.0, -1.0, 18.0, 150.0, 227.0, 100, 107.0, 207.0]) # TODO Corner Case 1
    test_model(100, "./test/states_generic.json",    [256.0, 0.0, 309.0, 300, 0.6949668825961136, 400, 200.0, 253.0]) #
    # test_model(100, "./test/states_generic.json",   [100.0, -2.0, 18.0, 127.0, 227.0, 95, 107.0, 207.0]) # TODO Corner Case 2
    # test_model(100, "./test/states_generic.json", [113.0, -1.0, 18.600000000000023, 115.0, 215.0, 182.60000000000002, 174.0, 274.0])  # for generic reward:91
    # test_model(100, "./test/states_generic.json", [256.0, -2.0, 309.0, 172.0, 272.0, 453.0, 26.0, 126.0])  # for generic reward:-4
    # test_model(100, "./test/states_generic.json", [0, 0, 0, 0, 0, 0, 0, 0])  #TODO for generic reward:-4 //和上面的episode reward一样？？
