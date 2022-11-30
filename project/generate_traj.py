__author__ = "Filip Lemic, Jakob Struye, Jeroen Famaey"
__copyright__ = "Copyright 2021, Internet Technology and Data Science Lab (IDLab), University of Antwerp - imec"
__version__ = "1.0.0"
__maintainer__ = "Filip Lemic"
__email__ = "filip.lemic@uantwerpen.be"
__status__ = "Development"

"""
@summary: The original file is under pm4vr/examples/acm_mmsys_size_vs_users_full.py. We modify the example for our 
          setup. The dimensions of the XY table are 1.3 m x 1.3 m, and we consider only 1 user.
           
@author: Syed Hashim Ali Shah
		 Panagiotis Skrimponis

@contact: ps3857@nyu.edu
"""

import os
import sys

path = os.path.abspath('../pm4vr/')
if path not in sys.path:
    sys.path.append(path)

import environment
from user import User
import visualization
import algorithm
import prediction
from algorithm import rad
import numpy as np
from collections import defaultdict
import time
import random
from scipy.io import savemat


def main():
    # Parameters
    steps_per_second = 1  # Number of data points (both virtual and physical) per second
    duration = 1001  # Duration of the experiment in seconds
    gamma = 1.5  # Gamma causes the influence of users to fall off exponentially instead of linearly
    radius = 65  # r is the radius of the arc on which a walking user is being redirected

    env_size = 130
    num_trajs = 2

    for num_traj in range(num_trajs):
        print(f'Trajectory #{num_traj}')
        env = environment.define_square(env_size)

        rdw = algorithm.RedirectedWalker(duration=duration, steps_per_second=steps_per_second, gamma=gamma,
                                         base_rate=rad(1.5), max_move_rate=rad(15), max_head_rate=rad(30),
                                         velocity_thresh=0.1, ang_compress_scale=0.85, ang_amplify_scale=1.3,
                                         scale_multiplier=2.5, radius=radius, t_a_norm=15.0, env=env)

        # Define the users by defining their virtual movement trajectory.
        usr1 = User([random.uniform(-env_size / 2, env_size / 2), random.uniform(-env_size / 2, env_size / 2)],
                    1.0, 1)
        usr1.fill_virtual_path(rdw.steps, rdw.delta_t, 1)

        users = [usr1]

        num_resets_per_users = defaultdict(
            int)  # Definition of the performance metric entitled number of resets per user
        distance_between_resets_per_user = defaultdict(list)  # Storing all distances between resets per user

        for user in users:
            # Let the first step be taken without redirection to kick-start stuff
            user.phy_locations.append(user.virt_locations[1])
            distance_between_resets_per_user[user.identity] = []
            distance_between_resets_per_user[user.identity].append(0.0)

        for time_iter in range(0, rdw.steps - 2):

            # At each step of the evaluation, calculate force_vectors and moving_rates. Force vectors are used to define the
            # optimal physical movement direction for each user (i.e., to avoid hitting environmental obstacles and other users).
            # Moving rates provide constraints on how much the user can be steered in the physical world without noticing it
            # in the virtual one. Force vectors are (at the moment) calculated using the APF-RDW algorithm
            force_vectors, env_vectors, user_vectors = rdw.calculate_force_vectors(users)
            moving_rates = rdw.calculate_max_rotations(users)

            iter_temp = 0

            # Iterate through all users and calculate their next physical step based on the force vectors and moving rates.
            for user in users:

                # x_step and y_step define the user's physical offset from the current location
                step = rdw.calculate_next_physical_step(user, moving_rates[iter_temp], force_vectors[iter_temp])

                # (Jakob) Redirection was implemented with a 180 degree rotation. The paper however proposes to rotate towards
                # the force vector. This moves the user away from all obstacles (walls and other users) optimally meaning
                # there's no reason to check for users and walls separately.
                # The following method checks if a collision is about to happen (threshold selected arbitrarily for now)
                reset_step = rdw.reset_if_needed(force_vectors[iter_temp], env_vectors[iter_temp],
                                                 user_vectors[iter_temp], threshold=1000000)

                reset_occurred = reset_step is not None
                if reset_occurred:
                    step = reset_step
                    # Create a new instance in the list representing distances passed without a reset
                    distance_between_resets_per_user[user.identity].append(algorithm.norm(step))
                else:
                    # Add the step in latest instance of the list representing distances passed without a reset
                    distance_between_resets_per_user[user.identity][-1] += algorithm.norm(step)

                # This is just for storing the number of rotations per user
                num_resets_per_users[user.identity] += reset_occurred

                # Update the user's physical trajectory with the newest location
                user.phy_locations.append(user.get_phy_loc() + step)

                iter_temp += 1

        data = np.zeros((duration - 1, 3))
        arr1 = (np.array(user.phy_locations) + 65) * 10
        tmp = arr1[:-1] - arr1[1:]
        data[:, 0] = (arr1[:-1, 0] + arr1[1:, 0]) / 2
        data[:, 1] = (arr1[:-1, 1] + arr1[1:, 1]) / 2

        tmp = np.arctan(tmp[:, 1] / tmp[:, 0]) * 180 / np.pi
        angles = np.array([-45, -30, -15, 0, 15, 30, 45])
        data[:, 2] = angles[np.digitize(tmp, [-37.5, -22.5, -7.5, 7.5, 22.5, 37.5])]
        np.save(time.strftime("../data/%Y%m%d_trajectory_"+f"{num_traj}.npy"), data)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'ERROR: {e}')
