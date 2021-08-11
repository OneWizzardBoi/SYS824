import math
import random
import matplotlib.pyplot as plt

from NLinkArm3d import NLinkArm

def random_val(min_val, max_val):
    return min_val + random.random() * (max_val - min_val)


def main():
    
    # defining the target pose (x,y,z)
    target_point = [0.9, 0.8, 0.8]
    ee_target_pose = target_point + [random_val(-0.5, 0.5), random_val(-0.5, 0.5), random_val(-0.5, 0.5)]

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
        pass
    

if __name__ == "__main__":
    main()
