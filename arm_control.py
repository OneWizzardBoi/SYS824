import sys
import time
import copy
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from NLinkArm import NLinkArm
from obstacle_avoidance import ClosestPoint, Obstacle

# Simulation parameters (from the basic example)
Kp = 1
dt = 0.1 # 0.1 seconds
N_LINKS = 3
N_ITERATIONS = 10000
command_ang_velocity = 1 # rad/s

def animation():

    # initializing the obstacles
    #obstacles = [Obstacle(0.5, 2.3, [0.5, 0]), Obstacle(-1, 2.3, [0.5, 0])]
    obstacles = [Obstacle(-1, 2.3, [0.5, 0])]

    # defining the variable containers and initializing the arm
    link_lengths = [1] * N_LINKS
    joint_angles = np.array([0] * N_LINKS)
    goal_pos = [0, 2]
    arm = NLinkArm(link_lengths, joint_angles, goal_pos, True, obstacles)

    # getting the inverse kinematics solution (joint_goal_angles)
    joint_goal_angles, solution_found = inverse_kinematics(link_lengths, joint_angles, goal_pos)

    if solution_found:

        # velocity calculation vars
        prev_joint_angles = copy.deepcopy(joint_angles)
        prev_joint_positions = copy.deepcopy(arm.points)

        while True:
        
            old_goal = np.array(goal_pos)
            goal_pos = np.array(arm.goal)
            end_effector = arm.end_effector
            errors, distance = distance_to_goal(end_effector, goal_pos)

            if distance > 0.1 and all(old_goal == goal_pos):

                # getting the angular velocities + the relative velocity (obstacle - robot)
                joint_angular_velocities = (joint_angles - prev_joint_angles)/dt

                # computing the repulsion velocities
                repulsion_velocities = []
                for obstacle in obstacles:
                    obstacle.compute_closest_point(arm.points)
                    obstacle.compute_relative_velocity(prev_joint_positions, link_lengths, dt)
                    repulsion_velocities.append(obstacle.compute_repulsion_vector())

                for i, velocity in enumerate(repulsion_velocities):
                    print(f"obstacle #{i+1} 's repulsion velocity magnitude: {velocity.magnitude}")
                
                # storing the current arm positions before update
                prev_joint_angles = copy.deepcopy(joint_angles)
                prev_joint_positions = copy.deepcopy(arm.points)

                # updating the arm's joint angles + updating obstacle positions
                
                #TODO : this is where the repulsion contribution  needs to be computed
                q_rep = partial_jacobian_inverse(link_lengths, joint_angles[0:obstacle.closest_point.segment_index])@velocity.vector
                while len(q_rep) < N_LINKS:
                    q_rep = np.append(q_rep,0)

                print(q_rep)
                joint_angular_velocities = ang_diff(joint_goal_angles, joint_angles) + q_rep
                
                joint_angles = joint_angles + joint_angular_velocities*dt#Kp * ang_diff(joint_goal_angles, joint_angles) * dt#
                for obstacle in obstacles: obstacle.update_position(dt)

                arm.update_joints(joint_angles)

            else: break


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

def partial_jacobian_inverse(link_lengths, joint_angles):
    J = np.zeros((2, len(joint_angles)+1))
    for i in range(len(joint_angles)+1):
        J[0, i] = 0
        J[1, i] = 0
        for j in range(i, len(joint_angles)+1):
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