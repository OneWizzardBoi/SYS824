import math
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt

from NLinkArm3d import NLinkArm

# defining control parameters
Kp = 1
dt = 0.1
target_min_distance = 0.1

def main():
    
    # defining the target pose (x,y,z)
    target_point = [0.9, 1.5, 0.8]
    ee_target_pose = target_point + [0,0,0]

    # init NLinkArm with the (Denavit-Hartenberg parameters) and the target pose
    n_link_arm = NLinkArm([[0.         , -math.pi / 2 , 0.7, 0.],
                           [math.pi / 2, math.pi / 2  , 0., 0.],
                           [0.         , -math.pi / 2 , 0., 1],
                           [0.         , math.pi / 2  , 0., 0.],
                           [0.         , -math.pi / 2 , 0., 1],
                           [0.         , math.pi / 2  , 0., 0.],
                           [0.         , 0.           , 0., 0.]], ee_target_pose)
    
    # computing the inverse kinematics solution (target angles)
    solution_found, goal_joint_angles = n_link_arm.inverse_kinematics(ee_target_pose)
    
    # control to get the effector to the target point
    if solution_found:

        target_reached = False
        while not target_reached:

            # getting the current joint angles and effector position
            ee_position = n_link_arm.forward_kinematics()[:3]
            curr_joint_angles = n_link_arm.get_joint_angles()
            
            # checking if the effector reached the destination
            target_ee_d = numpy.linalg.norm(np.array(ee_position) - np.array(target_point))
            target_reached = target_ee_d <= target_min_distance
            
            # computing the command angular velocities
            joint_angular_vels = Kp * ang_diff(goal_joint_angles, curr_joint_angles)

            # applying the joint velocities with a discrete time interval
            n_link_arm.set_joint_angles(curr_joint_angles + joint_angular_vels * dt) 

            # updating the graph
            n_link_arm.update_display()

    # notifying of impossible inverse kinematic problem
    else: print("No inverse kinematic soltion found")


def ang_diff(theta1, theta2):

    """ Returns the difference between two angles in the range -pi to +pi """
    
    theta1 = np.array(theta1)
    theta2 = np.array(theta2)
    return (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi


if __name__ == "__main__":
    main()
