import math
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


class Link:

    ''' Class implementing a slingle robot link (segment) in a multi segment robot '''

    def __init__(self, dh_params):

        ''' Getting the Denavit-Hartenberg parameters for the link '''
        self.dh_params_ = dh_params


    def transformation_matrix(self):

        ''' Computing the transformation matrix to go from the start to the end of the link '''

        theta = self.dh_params_[0]
        alpha = self.dh_params_[1]
        a = self.dh_params_[2]
        d = self.dh_params_[3]

        st = math.sin(theta)
        ct = math.cos(theta)
        sa = math.sin(alpha)
        ca = math.cos(alpha)
        trans = np.array([[ct, -st * ca, st * sa, a * ct],
                          [st, ct * ca, -ct * sa, a * st],
                          [0, sa, ca, d],
                          [0, 0, 0, 1]])

        return trans


    @staticmethod
    def basic_jacobian(trans_prev, ee_pos):

        # getting the previous position on the link's extremity
        # getting the previous z-axis rotational values 
        pos_prev = np.array([trans_prev[0, 3], trans_prev[1, 3], trans_prev[2, 3]])
        z_axis_prev = np.array([trans_prev[0, 2], trans_prev[1, 2], trans_prev[2, 2]])

        basic_jacobian = np.hstack((np.cross(z_axis_prev, ee_pos - pos_prev), z_axis_prev))
        return basic_jacobian


class NLinkArm:

    ''' Class implementing a multi-segment robot '''

    max_distance_error = 0.05

    def __init__(self, dh_params_list, ee_target_poses):

        '''
        Parameters
        ----------
        dh_params_list : denavit-Hartenberg parameters for the robot
        ee_target_poses : target poses wich the robot must reach sequentially
        '''

        self.ee_target_poses = ee_target_poses

        self.link_list = []
        for i in range(len(dh_params_list)):
            self.link_list.append(Link(dh_params_list[i]))

        # setting up the display and first display
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        self.fig.show()


    def transformation_matrix(self):

        # multiplying the link's transformation matrices to get T0-n
        trans = np.identity(4)
        for i in range(len(self.link_list)):
            trans = np.dot(trans, self.link_list[i].transformation_matrix())
        return trans


    def forward_kinematics(self):

        trans = self.transformation_matrix()

        # getting the end effector position
        x = trans[0, 3]
        y = trans[1, 3]
        z = trans[2, 3]
        alpha, beta, gamma = self.euler_angle()

        return [x, y, z, alpha, beta, gamma]


    def basic_jacobian(self):

        basic_jacobian_mat = []

        # getting the end effector position
        ee_pos = self.forward_kinematics()[0:3]
        
        trans = np.identity(4)
        for i in range(len(self.link_list)):
            basic_jacobian_mat.append(self.link_list[i].basic_jacobian(trans, ee_pos))
            trans = np.dot(trans, self.link_list[i].transformation_matrix())

        return np.array(basic_jacobian_mat).T


    def inverse_kinematics(self, ref_ee_pose):

        goal_joint_angles = None

        # saving the initial joint angle values
        initial_joint_angles = self.get_joint_angles() 

        # iterative solution to the inverse kinematics
        ee_pose = None
        for cnt in range(1000):

            # getting the current effector position
            # getting the (target - current) position and angles differences
            ee_pose = self.forward_kinematics()
            diff_pose = np.array(ref_ee_pose) - ee_pose

            # getting the current Jacobian and euler angles
            basic_jacobian_mat = self.basic_jacobian()
            alpha, beta, gamma = self.euler_angle()

            K_zyz = np.array(
                [[0, -math.sin(alpha), math.cos(alpha) * math.sin(beta)],
                 [0, math.cos(alpha), math.sin(alpha) * math.sin(beta)],
                 [1, 0, math.cos(beta)]])
            K_alpha = np.identity(6)
            K_alpha[3:, 3:] = K_zyz

            # updating the joint angles (converging towards the solution)
            theta_dot = np.dot(
                np.dot(np.linalg.pinv(basic_jacobian_mat), K_alpha),
                np.array(diff_pose))
            self.update_joint_angles(theta_dot / 100.)

        # checking the solution validity
        distance_error = numpy.linalg.norm(np.array(ee_pose[:3]) - np.array(ref_ee_pose[:3]))
        solution_found = distance_error <= self.max_distance_error
        if solution_found: goal_joint_angles = self.get_joint_angles()
        
        # setting the initial joint angle values
        self.set_joint_angles(initial_joint_angles)

        return solution_found, goal_joint_angles


    def euler_angle(self):

        trans = self.transformation_matrix()

        alpha = math.atan2(trans[1][2], trans[0][2])
        if not (-math.pi / 2 <= alpha <= math.pi / 2):
            alpha = math.atan2(trans[1][2], trans[0][2]) + math.pi
        if not (-math.pi / 2 <= alpha <= math.pi / 2):
            alpha = math.atan2(trans[1][2], trans[0][2]) - math.pi

        beta = math.atan2(
            trans[0][2] * math.cos(alpha) + trans[1][2] * math.sin(alpha),
            trans[2][2])

        gamma = math.atan2(
            -trans[0][0] * math.sin(alpha) + trans[1][0] * math.cos(alpha),
            -trans[0][1] * math.sin(alpha) + trans[1][1] * math.cos(alpha))

        return alpha, beta, gamma


    def get_joint_angles(self):
        ''' Getting the (theta) variable in the Denavit-Hartenberg parameters '''
        return [self.link_list[i].dh_params_[0] for i in range(len(self.link_list))]

    def set_joint_angles(self, joint_angle_list):
        ''' Setting the (theta) variable in the Denavit-Hartenberg parameters '''
        for i in range(len(self.link_list)):
            self.link_list[i].dh_params_[0] = joint_angle_list[i]

    def update_joint_angles(self, diff_joint_angle_list):
        ''' Updating the (theta) variable in the Denavit-Hartenberg parameters '''
        for i in range(len(self.link_list)):
            self.link_list[i].dh_params_[0] += diff_joint_angle_list[i]
    

    def update_display(self):

        ''' Displaying the robot's joints and segments '''

        plt.cla()

        # defining containers for the joint coordinates
        x_list = []
        y_list = []
        z_list = []

        # robot base coordinates [0, 0, 0]
        trans = np.identity(4)
        x_list.append(trans[0, 3])
        y_list.append(trans[1, 3])
        z_list.append(trans[2, 3])
        # getting the joint coordinates by going through the T0-n matrices
        for i in range(len(self.link_list)):
            trans = np.dot(trans, self.link_list[i].transformation_matrix())
            x_list.append(trans[0, 3])
            y_list.append(trans[1, 3])
            z_list.append(trans[2, 3])

        # plotting the robot's joints and segments
        self.ax.plot(x_list, y_list, z_list, "o-", color="#0331fc", ms=4, mew=0.5)
        self.ax.plot(x_list[0], y_list[0], z_list[0], 'o', color="#000000")
        self.ax.plot(x_list[-1], y_list[-1], z_list[-1], 'o', color="#ed0707")
        
        # plotting the target point
        for ee_target_pose in self.ee_target_poses: 
            self.ax.plot(ee_target_pose[0], ee_target_pose[1], ee_target_pose[2], 'gx', color="#03fc03")

        # setting limits
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ax.set_zlim(-3, 3)
        
        plt.draw()
        plt.pause(0.0001)