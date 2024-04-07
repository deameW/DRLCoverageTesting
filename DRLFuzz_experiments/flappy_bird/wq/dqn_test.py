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


def test_model(iteration, save_path, initial_state, mutate=False):
    score_all_episodes = 0
    episode_data = []
    failedEpisode = 0

    dqn = DQN()

    env.init()
    env.reset_game()

    for i_episode in range(iteration):
        ep_r = 0
        episode_states = []
        env.reset_game()

        # Call TestFlappybird to initialize the game if initial states are given
        if initial_state is not None:
            s_ = initial_state
            testFlappybird._init(s_[0], s_[1], s_[2], s_[3])
            t = list(env.getGameState().values())
            s_ = t
            episode_states.append(convert_to_float(t))
        # Let game picks initial state randomly
        else:
            s_ = list(env.getGameState().values())
            episode_states.append(convert_to_float(s_))

        while True:
            # Choose an action by trained model
            s = s_
            a = dqn.choose_action(s, 1)
            ac = K_w if a else None
            r = env.act(ac)

            time.sleep(0.01)

            # Record the new state returned from environment
            s_ = list(env.getGameState().values())
            episode_states.append(convert_to_float(s_))

            # To tell if the game is over
            done = env.game_over()
            ep_r += r

            if done:
                # Record the information of the episode
                episode_data.append({
                    "initial_state": episode_states[0],
                    "last_state_before_failure": episode_states[-1],
                    "total_reward": ep_r
                })
                score_all_episodes += ep_r
                print('Ep: ', i_episode, '| Ep_r: ', round(ep_r, 2))

                if ep_r <= 10:
                    failedEpisode += 1
                break

    # Dump the information of all episodes to json file
    average_score = score_all_episodes / iteration
    print("Average score: ", average_score)
    episode_data.append({
        "average_score": average_score,
        "failed_episode": failedEpisode,
    })
    if save_path != "":
        with open(save_path, 'w') as f:
            json.dump(episode_data, f, default=convert_to_float)

    # Return the average score under a given initial state
    if mutate:
        return score_all_episodes / iteration


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


def randomGenerate():
    pipe1 = random.randint(25, 192)
    pipe2 = random.randint(25, 192)
    dist = random.randint(-120, -75)
    vel = random.randint(-56, 10)
    return [pipe1, pipe2, dist, vel]


if __name__ == '__main__':
    # test_model(100, "./test/episode_info_random_start.json", randomGenerate())  # Random Initial State
    test_model(100, "./test/episode_info_random_start.json", [110, 87, -113, -23])  # Random Initial State
