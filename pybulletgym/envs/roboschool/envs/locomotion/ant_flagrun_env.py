from typing import Tuple

import numpy as np
from pybullet_envs.gym_locomotion_envs import AntBulletEnv, WalkerBaseBulletEnv

from pybulletgym.envs import gym_utils as ObjectHelper


class AntFlagrunBulletEnv(AntBulletEnv):
    def __init__(self, size=5, tolerance=0.5, max_targets=10, manual_goal_creation=False, seed=123):
        super().__init__()
        self.size = size
        self.manual_goal_creation = manual_goal_creation
        self.tol = tolerance
        self.max_targets = max_targets

        # needed for parallel envs so that they create the same targets on each thread (np_random seems to not work)
        self.mpi_common_rand: np.random.RandomState = np.random.RandomState(seed)
        self.create_target()
        self.flag = None

        self.goals = []

    def create_single_player_scene(self, bullet_client):
        scene = super().create_single_player_scene(bullet_client)
        if self.isRender and not self.flag:
            self.flag = ObjectHelper.get_sphere(self._p, self.walk_target_x, self.walk_target_y, 0.7)
        return scene

    def create_target(self) -> Tuple[float, float]:
        g = (self.mpi_common_rand.uniform(-1, 1), self.mpi_common_rand.uniform(-1, 1))
        while np.linalg.norm(g) < 0.25:
            g = (self.mpi_common_rand.uniform(-1, 1), self.mpi_common_rand.uniform(-1, 1))

        return g[0] * self.size, g[1] * self.size

    def create_targets(self, n):
        """
        Used to create multiple goals such that all agents run in parallel will continue to generate the same values.
        Each generation these should be cleared and recreated
        """
        for _ in range(n):
            self.goals.append(self.create_target())

    def set_target(self, x, y):
        self.walk_target_x = x
        self.walk_target_y = y
        self.robot.walk_target_x = x
        self.robot.walk_target_y = y

        if self.isRender:
            self._p.resetBasePositionAndOrientation(self.flag.bodies[0],
                                                    [self.walk_target_x, self.walk_target_y, 0.7],
                                                    [0, 0, 0, 1])

    def reset(self):
        WalkerBaseBulletEnv.electricity_cost = 0  # -2.0
        WalkerBaseBulletEnv.stall_torque_cost = 0  # -0.1

        r = super().reset()

        self.goals.clear()
        if not self.manual_goal_creation:  # creating a new goal on every reset if not manually creating the goal
            self.create_targets(self.max_targets)
            self.set_target(*self.goals.pop())

        return r

    def step(self, a):
        s, r, d, i = super().step(a)

        # If close enough to target then give extra reward and move the target.
        if np.linalg.norm(self.robot.body_xyz[:2] - np.array([self.walk_target_x, self.walk_target_y])) < self.tol:
            r += 5000
            try:
                self.set_target(*self.goals.pop())
                i['target'] = [self.walk_target_x, self.walk_target_y]
            except IndexError:
                print('Reached all goals!')
                d = True
                r += 100000

        return s, r, d, i
