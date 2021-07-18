import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from mathutils.geometry import intersect_point_line


class RepulsionVelocity():

    ''' Class for the representation of repulsion velocities '''

    def __init__(self, vector, obs_coordinates, closest_point):

        '''
        Parameters
        ----------
        vector (np.array([float, float])): the velocity vector
        obs_coorinates ([float, float]) : coordinates of the obstacle from which the vector originates
        closest_point ([float, float]) : coordinates of the closest point on the robot which the vector affects
        '''

        self.vector = vector
        self.obs_coordinates = obs_coordinates
        self.closest_point = closest_point

        self.magnitude = math.sqrt(self.vector[0]**2 + self.vector[1]**2)


class ClosestPoint():

    ''' Class for the representation of the closest point on the robot to an obstacle '''

    def __init__(self, coordinates, distance, segment_index, segment_pos):

        '''
        Parameters
        ----------
        coordinates : tuple containing the (x, y) coordinates of the point
        distance : distane between the obstacle and the closest point
        segment_index : identifies the robot segment the point belongs to
        segment_pos : specifies the percentage [0, 1] of the length of the segment where the point is situated
        '''

        self.coordinates = coordinates
        self.distance = distance
        self.segment_index = segment_index
        self.segment_pos = segment_pos


class Obstacle():

    ''' Class implementing the mathematical modelisation and display of 2D obstacles '''

    # mathematical model constants
    dcr = 0.2      # critical distance (black circle)
    d1 = 0.4       # minimum radius for repulsion activation (orange circle)
    cv = 1         # repulsion radius sizing constant
    k1 = 4         # vrep1 amplitude constant
    k2 = 2         # vrep2 amplitude constant
    l1 = d1        # lower bound for the damping force (below this c=1)
    l2 = l1 + 0.2  # upper bound for damping force (above this c=0)
    dmax = l2      # maximum radius for repulsion activatio

    def __init__(self, x_pos, y_pos, velocity=[0,0]):

        '''
        Parameters
        ----------
        x_pos (float) : the obstacle's initial x coordinate 
        x_pos (float) : the obstacle's initial y coordinate
        velocity ([float, float]) : the obstacle's velocity vector i.e [x velocity, y velocity]
        '''

        # specifying the spawn position
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.velocity = velocity
        self.velocity_mod = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)

        self.vrel = 1
        self.prev_vrel = 1
        self.d0 = self.d1
        self.closest_point = None


    @classmethod
    def display_obstacles(cls, obstacles, plt):

        for obstacle in obstacles:

            # displaying the obstacle
            plt.scatter(obstacle.x_pos, obstacle.y_pos, s=20, c='black')
            l1_radius = plt.Circle((obstacle.x_pos, obstacle.y_pos), obstacle.l1, facecolor='none', edgecolor='blue')
            l2_radius = plt.Circle((obstacle.x_pos, obstacle.y_pos), obstacle.l2, facecolor='none', edgecolor='blue')
            do_radius = plt.Circle((obstacle.x_pos, obstacle.y_pos), obstacle.d0, facecolor='none', edgecolor='orange')
            dcr_radius = plt.Circle((obstacle.x_pos, obstacle.y_pos), obstacle.dcr, facecolor='none', edgecolor='black')
            plt.gca().add_patch(l1_radius)
            plt.gca().add_patch(l2_radius)
            plt.gca().add_patch(do_radius)
            plt.gca().add_patch(dcr_radius)
            
            # displaying shortest distance
            if obstacle.closest_point is not None:
                plt.plot([obstacle.x_pos, obstacle.closest_point.coordinates[0]], [obstacle.y_pos, obstacle.closest_point.coordinates[1]], c='orange')


    def update_position(self, dt):
        
        self.x_pos += self.velocity[0]*dt
        self.y_pos += self.velocity[1]*dt


    def compute_closest_point(self, arm_points):

        '''
        Identifies the point on the robot which is closest to the obstacle

        Parameters
        ----------
        arm_points : the coordinates of each of the arm's points ie. [[x1, y1], [x2, y2]] "

        '''

        robot_segments = []
        candidate_points = []

        # assembling the coordinates for each segment of the robot [(x1, y1, x2, y2), (x2, y2, x3, y3), ...] 
        for i in range(len(arm_points)-1):
            robot_segments.append((arm_points[i][0], arm_points[i][1], arm_points[i+1][0], arm_points[i+1][1]))

        # computing the shortest distance between the obstacle and every segment of the robot
        for segment_i, segment in enumerate(robot_segments):

            seg_pos = None
            c_point = None
            distance = None
            
            intersect_info = intersect_point_line((self.x_pos, self.y_pos), (segment[0], segment[1]), (segment[2], segment[3]))
            
            # handling when the nearest point is outside the segment
            if intersect_info[1] < 0 :
                c_point = (segment[0], segment[1])
                seg_pos = 0
            elif intersect_info[1] > 1:
                c_point = (segment[2], segment[3])
                seg_pos = 1
            # handling when the nearest point is on the segment
            else:
                c_point = intersect_info[0]
                seg_pos = intersect_info[1]

            # computing the distance between the closest point ant the obstacle + adding to the list
            distance = math.sqrt((c_point[0] - self.x_pos)**2 + (c_point[1] - self.y_pos)**2)
            candidate_points.append(ClosestPoint(c_point, distance, segment_i, seg_pos))

        # selecting the closest point accross the whole robot
        min_d = 1000
        winner_i = 0
        for i, candidate in enumerate(candidate_points):
            if candidate.distance < min_d:
                min_d = candidate.distance
                winner_i = i

        self.closest_point = candidate_points[winner_i]


    def compute_relative_velocity(self, prev_joint_positions, link_lengths, dt):
        
        ''' Computes the relative velocity between the obstacle and the closest point (obstacle - closest point) '''

        # getting the previous position of the current closest point
        segment_i = self.closest_point.segment_index
        l1 = (prev_joint_positions[segment_i + 1][0] - prev_joint_positions[segment_i][0])
        l2 = (prev_joint_positions[segment_i + 1][1] - prev_joint_positions[segment_i][1])
        theta = math.atan2(l2, l1)
        x_diff = self.closest_point.segment_pos * link_lengths[segment_i] * math.cos(theta)
        y_diff = self.closest_point.segment_pos * link_lengths[segment_i] * math.sin(theta)
        x_prev = prev_joint_positions[segment_i][0] + x_diff
        y_prev = prev_joint_positions[segment_i][1] + y_diff
       
        # calculating the closest point velocity + calculating the relative velocity
        cp_velocity = math.sqrt((self.closest_point.coordinates[0] - x_prev)**2 + (self.closest_point.coordinates[1] - y_prev)**2) / dt
        self.vrel = self.velocity_mod - cp_velocity

        # updating d0, since it depends on vrel
        self.d0 = self.d1
        if self.vrel < 0: 
            self.d0 = self.d1 - (self.vrel * self.cv)
            if self.d0 > self.dmax: self.d0 = self.dmax


    def compute_repulsion_vector(self):

        dmin = self.closest_point.distance

        # calculating vrep1
        vrep1 = 0
        if (dmin - self.dcr) < self.d0:
            vrep1 = self.k1 * ((self.d0/(dmin-self.dcr)) - 1)

        # calculating vrep2
        c = 0
        vrep2 = 0
        if self.vrel < 0:
            if dmin <= self.l1: 
                c = 1
            elif (self.l1 < dmin) and (self.l2 >= dmin):
                c = (1 + math.cos(math.pi*(dmin - self.l1) / (self.l2 - self.l1))) / 2 
            
            vrep2 = -1 * c * self.k2 * self.vrel

        # calculating vrep
        v_x = self.closest_point.coordinates[0] - self.x_pos
        v_y = self.closest_point.coordinates[1] - self.y_pos
        unit_v = np.array([v_x, v_y]) / math.sqrt(v_x**2 + v_y**2)
        vrep = (vrep1 + vrep2) * unit_v

        return RepulsionVelocity(vrep, [self.x_pos, self.y_pos], copy.deepcopy(self.closest_point))