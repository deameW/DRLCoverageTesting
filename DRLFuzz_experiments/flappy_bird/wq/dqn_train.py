from DRLFuzz_experiments.flappy_bird.wq.dqn_model import DQN
from pygame.constants import K_w

import numpy as np
import os

from DRLFuzz_experiments.flappy_bird.wq.env.flappy_bird_env import env

MEMORY_CAPACITY = 20000


os.environ['SDL_VIDEODRIVER'] = "dummy"


def convert_to_serializable(data):
    if isinstance(data, np.int32):
        return int(data)
    elif isinstance(data, list):
        return [convert_to_serializable(x) for x in data]
    else:
        return data


def train_model(iteration):
    dqn = DQN()
    env.init()
    reward = env.act(None)
    env.reset_game()

    print('\nCollecting experience...')

    t = 0
    ep_r = 0
    s = None
    s_ = list(env.getGameState().values())
    print(s_)
    for i_episode in range(iteration):
        # env.render()
        ep_r = 0

        while 1:
            # s = s_[1:len(s_)]
            s = s_
            a = dqn.choose_action(s, 0.95)
            t = t + 1
            ac = None
            if a:
                ac = K_w

            r = env.act(ac)
            s_ = list(env.getGameState().values())
            done = env.game_over()

            # time.sleep(0.01)

            dqn.store_transition(s, a, r, s_, done)
            ep_r += r
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
            if done:
                env.reset_game()
                print('t=', t, 'Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))
                break

    # 保存文件
    dqn.save_model()


if __name__ == '__main__':
    train_model(10000)
