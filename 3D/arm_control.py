import math
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt

from NLinkArm3d import NLinkArm
from obstacle_avoidance import ClosestPoint, Obstacle

# defining control parameters
Kp = 1.2
dt = 0.1
target_min_distance = 0.05

def main():
    
    # defining the target pose (x,y,z)
    target_i = 0
    target_points = [[-1, 1.5, -1.5], [1.5, -1, 0.8]]
    ee_target_poses = [target_point + [0,0,0] for target_point in target_points]
    goal_joint_angles_saved = [None] * len(target_points)

    # initializing the obstacles
    obstacles = [Obstacle(-3, 1.5, -1.5, [0.7, 0, 0]), Obstacle(3, -1, 0.8, [-0.7, 0, 0])]

    # init NLinkArm with the (Denavit-Hartenberg parameters) and the target pose
    link_lengths = [0.7, 1, 1]
    n_link_arm = NLinkArm([[0.         , -math.pi / 2 , 0.7, 0.],
                           [math.pi / 2, math.pi / 2  , 0., 0.],
                           [0.         , -math.pi / 2 , 0., 1],
                           [0.         , math.pi / 2  , 0., 0.],
                           [0.         , -math.pi / 2 , 0., 1],
                           [0.         , math.pi / 2  , 0., 0.],
                           [0.         , 0.           , 0., 0.]], ee_target_poses, obstacles=obstacles)

    # perpetually alternating between the target points
    first_pass = True
    while True:

        # defining the target point for the effector
        n_link_arm.ee_target_pose = ee_target_poses[target_i]

        # computing / consulting the inverse kinematics solution (target angles) according to the target
        solution_found = False
        goal_joint_angles = None
        if goal_joint_angles_saved[target_i] is None:
            solution_found, goal_joint_angles = n_link_arm.inverse_kinematics(ee_target_poses[target_i])
            if not first_pass: goal_joint_angles_saved[target_i] = (solution_found, goal_joint_angles)
        else:
            solution_found = goal_joint_angles_saved[target_i][0]
            goal_joint_angles = goal_joint_angles_saved[target_i][1]
        first_pass = False

        # control to get the effector to the target point (if an inverse kinematic solution exists)
        if solution_found:

            target_reached = False
            prev_joint_positions = n_link_arm.get_joint_positions()

            while not target_reached:

                # getting the current joint angles and positions
                curr_joint_angles = n_link_arm.get_joint_angles()
                arm_joint_positions = n_link_arm.get_joint_positions()

                # computing the repulsion velocities + keeping track of joint positions
                repulsion_velocities = []
                for obstacle in obstacles:
                    obstacle.compute_closest_point(arm_joint_positions)
                    obstacle.compute_relative_velocity(prev_joint_positions, link_lengths, dt)
                    repulsion_velocities.append(obstacle.compute_repulsion_vector())
                prev_joint_positions = n_link_arm.get_joint_positions()

                # computing the joint velocities vector due to the repulsion vector
                # ...

                # computing the command angular velocities
                joint_angular_vels = Kp * ang_diff(goal_joint_angles, curr_joint_angles)

                # applying the joint velocities with a discrete time interval (arm update)
                # updating the obstacle positions
                n_link_arm.set_joint_angles(curr_joint_angles + joint_angular_vels * dt)
                for obstacle in obstacles: obstacle.update_position(dt)
                n_link_arm.update_display()

                # checking if the effector reached the destination
                target_ee_d = numpy.linalg.norm(np.array(arm_joint_positions[-1]) - np.array(target_points[target_i]))
                target_reached = target_ee_d <= target_min_distance

            # preparing the selection of the next target point
            target_i = (target_i + 1) % len(target_points)

        else: raise ValueError("No inverse kinematic soltion found")


def ang_diff(theta1, theta2):

    """ Returns the difference between two angles in the range -pi to +pi """
    
    theta1 = np.array(theta1)
    theta2 = np.array(theta2)
    return (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi


if __name__ == "__main__":
    main()
