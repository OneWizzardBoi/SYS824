import math
import numpy
from mathutils.geometry import intersect_point_line


def joint_angles_to_lines(joint_angles, link_lengths):
    
    ''' 
    This function assumes we are working with a planar robot with a base situated a (0, 0).
    Computes an (x1, y1, x2, y2) representation for each robot segment.

    Returns
    -------
    (list of tuples) : [(x1, y1, x2, y2), (x1, y1, x2, y2), ...]
    '''

    robot_segments = []

    if not len(joint_angles) == len(link_lengths):
        raise ValueError("The quantity of joint angles and robot segments have to match")

    else: 
        
        start_x = 0 
        start_y = 0

        for joint_angle, link_length in zip(joint_angles, link_lengths):

            x1 = start_x
            y1 = start_y
            x2 = x1 + link_length * math.cos(joint_angle)
            y2 = y1 + link_length * math.sin(joint_angle)

            start_x = x2
            start_y = y2

            robot_segments.append(tuple((x1, y1, x2, y2)))

    return robot_segments


class Obstacle():

    ''' Class implementing the mathematical modelisation and display of 2D obstacles '''

    def __init__(self, x_pos, y_pos):

        self.x_pos = x_pos
        self.y_pos = y_pos
        self.closest_point = None


    def compute_closest_point(self, robot_segments):

        '''
        Identifies the point on the robot which is closest to the obstacle

        Parameters
        ----------
        robot_segments : [(x1, y1, x2, y2), ...] as produced by "joint_angles_to_lines"
        '''

        candidate_points = []

        for segment in robot_segments:

            distance = None
            closest_point = None

            intersect_info = intersect_point_line((self.x_pos, self.y_pos), (segment[0], segment[1]), (segment[2], segment[3]))
            
            # handling when the nearest point is outside the segment
            if intersect_info[1] < 0 :
                closest_point = (segment[0], segment[1])
            elif intersect_info[1] > 1:
                closest_point = (segment[2], segment[3])
            # handling when the nearest point is on the segment
            else: closest_point = intersect_info[0]

            # computing the distance between the closest point ant the obstacle
            distance = math.sqrt((closest_point[0] - self.x_pos)**2 + (closest_point[1] - self.y_pos)**2)

            candidate_points.append(tuple((closest_point, distance)))

        # selecting the closest point accross the whole robot
        min_d = 1000
        winner_i = 0
        for i, candidate in enumerate(candidate_points):
            if candidate[1] < min_d:
                min_d = candidate[1]
                winner_i = i

        self.closest_point = candidate_points[winner_i]