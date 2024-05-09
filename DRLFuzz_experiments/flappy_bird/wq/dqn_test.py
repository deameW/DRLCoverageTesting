import json
import random
import time
from pygame import K_w

from DRLFuzz_experiments.flappy_bird.wq.dqn_model import DQN
from DRLFuzz_experiments.flappy_bird.wq.env.flappy_bird_env import testFlappybird, env


# from dqn_model import DQN


def convert_to_float(obj):
    if isinstance(obj, list):
        return [float(item) for item in obj]
    return obj


def test_model(iteration, initial_state, mutate=False):
    score_all_episodes = 0
    # episode_data = []
    failedEpisode = 0

    dqn = DQN()

    env.init()
    env.reset_game()

    for i_episode in range(iteration):
        ep_r = 0
        # episode_states = []
        env.reset_game()

        # Call TestFlappybird to initialize the game if initial states are given
        if initial_state is not None:
            s_ = initial_state
            testFlappybird._init(s_[0], s_[1], s_[2], s_[3])
            t = list(env.getGameState().values())
            s_ = t
            # episode_states.append(convert_to_float([t[3], t[6], t[2] - 309, t[1]]))
        # Let game picks initial state randomly
        else:
            s_ = list(env.getGameState().values())
            # episode_states.append(convert_to_float([s_[3], s_[6], s_[2] - 309, s_[1]]))

        while True:
            # Choose an action by trained model
            s = s_
            a = dqn.choose_action(s, 1)
            ac = K_w if a else None
            r = env.act(ac)

            time.sleep(0.02)

            # Record the new state returned from environment
            s_ = list(env.getGameState().values())
            # episode_states.append(convert_to_float([s_[3], s_[6], s_[2] - 309, s_[1]]))

            # To tell if the game is over
            done = env.game_over()
            ep_r += r
            if done:
                # Record the information of the episode
                score_all_episodes += ep_r
                # print('Ep: ', i_episode, '| Ep_r: ', round(
                #     ep_r, 2))
                if ep_r <= 2:
                    failedEpisode += 1
                break
    episode_data = {
        "initial_state": initial_state,
        "avg_award_over_episodes": score_all_episodes / iteration
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
    return test_model(iteration, state)  # Random Initial State


def randomGenerate():
    pipe1 = random.randint(25, 192)
    pipe2 = random.randint(25, 192)
    dist = random.randint(-120, -75)
    vel = random.randint(-56, 10)
    return [pipe1, pipe2, dist, vel]




if __name__ == '__main__':
    # dump_info = []
    # for i in range(100):
    #     episode_data = testInitialState(1, randomGenerate())
    #     dump_info.append(episode_data)
    #
    # if save_path != "":
    #     with open(save_path, 'w') as f:
    #         json.dump(dump_info, f, default=convert_to_float)

    test_model(100, randomGenerate())
