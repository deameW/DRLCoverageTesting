import time

from DRLFuzz_experiments.flappy_bird.test_flappy_bird import TestFlappyBird
from ple import PLE

# time seed
seed = int(time.time())

testFlappybird = TestFlappyBird()
env = PLE(testFlappybird, fps=30, display_screen=True, force_fps=True, rng=seed)
