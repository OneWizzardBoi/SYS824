import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from NLinkArm import NLinkArm
from obstacle_avoidance import joint_angles_to_lines, Obstacle

# Simulation parameters (from the basic example)
Kp = 2
dt = 0.1
N_LINKS = 3
N_ITERATIONS = 10000

def animation():

    # initializing the obstacles
    obstacles = [Obstacle(1, 1), Obstacle(-1, 1)]

    # defining the variable containers and initializing the arm
    link_lengths = [1] * N_LINKS
    joint_angles = np.array([0] * N_LINKS)
    goal_pos = [0, -2]
    arm = NLinkArm(link_lengths, joint_angles, goal_pos, True, obstacles)

    joint_goal_angles, solution_found = inverse_kinematics(link_lengths, joint_angles, goal_pos)

    if solution_found:

        while True:
        
            old_goal = np.array(goal_pos)
            goal_pos = np.array(arm.goal)
            end_effector = arm.end_effector
            errors, distance = distance_to_goal(end_effector, goal_pos)

            if distance > 0.1 and all(old_goal == goal_pos):

                # getting the angular velocities (for going towards the goal)
                prev_joint_angles = joint_angles
                joint_angles = joint_angles + ang_diff(joint_goal_angles, joint_angles) * dt
                joint_velocities = (joint_angles - prev_joint_angles)/dt 
                
                # computing the repulsion velocities
                robot_segments = joint_angles_to_lines(joint_angles, link_lengths)
                for obstacle in obstacles:
                    obstacle.compute_closest_point(robot_segments)

            else: break

            arm.update_joints(joint_angles)


def get_random_goal():

    ''' Returns random coordinates '''

    from random import random
    SAREA = 3.0
    return [SAREA * random() - SAREA / 2.0,
            SAREA * random() - SAREA / 2.0]


def inverse_kinematics(link_lengths, joint_angles, goal_pos):
    """
    Calculates the inverse kinematics using the Jacobian inverse method.
    """
    for iteration in range(N_ITERATIONS):
        current_pos = forward_kinematics(link_lengths, joint_angles)
        errors, distance = distance_to_goal(current_pos, goal_pos)
        if distance < 0.1:
            print("Solution found in %d iterations." % iteration)
            return joint_angles, True
        J = jacobian_inverse(link_lengths, joint_angles)
        joint_angles = joint_angles + np.matmul(J, errors)
    return joint_angles, False


def forward_kinematics(link_lengths, joint_angles):
    x = y = 0
    for i in range(1, N_LINKS + 1):
        x += link_lengths[i - 1] * np.cos(np.sum(joint_angles[:i]))
        y += link_lengths[i - 1] * np.sin(np.sum(joint_angles[:i]))
    return np.array([x, y]).T


def jacobian_inverse(link_lengths, joint_angles):
    J = np.zeros((2, N_LINKS))
    for i in range(N_LINKS):
        J[0, i] = 0
        J[1, i] = 0
        for j in range(i, N_LINKS):
            J[0, i] -= link_lengths[j] * np.sin(np.sum(joint_angles[:j]))
            J[1, i] += link_lengths[j] * np.cos(np.sum(joint_angles[:j]))

    return np.linalg.pinv(J)


def distance_to_goal(current_pos, goal_pos):
    x_diff = goal_pos[0] - current_pos[0]
    y_diff = goal_pos[1] - current_pos[1]
    return np.array([x_diff, y_diff]).T, np.hypot(x_diff, y_diff)


def ang_diff(theta1, theta2):
    """
    Returns the difference between two angles in the range -pi to +pi
    """
    return (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi


if __name__ == '__main__':
    animation()