import math
import copy
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
from mathutils.geometry import intersect_point_line


class RepulsionVelocity():

    ''' Class for the representation of repulsion velocities '''

    def __init__(self, vector, obs_coordinates, closest_point):

        '''
        Parameters
        ----------
        vector (np.array([float, float, float])): the velocity vector
        obs_coorinates ([float, float, float]) : coordinates of the obstacle from which the vector originates
        closest_point ([float, float, float]) : coordinates of the closest point on the robot which the vector affects
        '''

        self.vector = vector
        self.obs_coordinates = obs_coordinates
        self.closest_point = closest_point
        self.magnitude = np.linalg.norm(self.vector)


class ClosestPoint():

    ''' Class for the representation of the closest point on the robot to an obstacle '''

    def __init__(self, coordinates, distance, segment_index, segment_pos):

        '''
        Parameters
        ----------
        coordinates : tuple containing the (x, y, z) coordinates of the point
        distance : distane between the obstacle and the closest point
        segment_index : identifies the robot segment the point belongs to
        segment_pos : specifies the percentage [0, 1] of the length of the segment where the point is situated
        '''

        self.coordinates = coordinates
        self.distance = distance
        self.segment_index = segment_index
        self.segment_pos = segment_pos


class Obstacle():

    ''' Class implementing the mathematical modelisation and display of 3D obstacles '''

    # mathematical model constants
    dcr = 0.3      # critical distance (black circle)
    d1 = 0.5       # minimum radius for repulsion activation (orange circle)
    cv = 1         # repulsion radius sizing constant
    k1 = 5         # vrep1 amplitude constant
    k2 = 3         # vrep2 amplitude constant
    l1 = d1        # lower bound for the damping force (below this c=1)
    l2 = l1 + 0.3  # upper bound for damping force (above this c=0)
    dmax = l2      # maximum radius for repulsion activatio

    # graphic display boudaries
    display_max = 3
    display_min = -3


    def __init__(self, x_pos, y_pos, z_pos, velocity=[0,0,0]):

        '''
        Parameters
        ----------
        x_pos (float) : the obstacle's initial x coordinate 
        y_pos (float) : the obstacle's initial y coordinate
        z_pos (float) : the obstacle's initial y coordinate
        velocity ([float, float, float]) : the obstacle's velocity vector i.e [x velocity, y velocity, z velocity]
        '''

        # specifying the spawn position
        self.init_x = self.x_pos = x_pos
        self.init_y = self.y_pos = y_pos
        self.init_z = self.z_pos = z_pos
        self.velocity = np.array(velocity)
        self.velocity_norm = np.linalg.norm(self.velocity)

        self.vrel = 1
        self.prev_vrel = 1
        self.d0 = self.d1
        self.closest_point = None
    

    @classmethod
    def display_obstacles(cls, obstacles, ax):

        for obstacle in obstacles:

            # displaying the obstacle

            ax.plot(obstacle.x_pos, obstacle.y_pos, obstacle.z_pos, 'o', color="#000000")
            dcr_radius_1 = plt.Circle((obstacle.x_pos, obstacle.y_pos), obstacle.dcr, facecolor='none', edgecolor='black')
            dcr_radius_2 = plt.Circle((obstacle.y_pos, obstacle.z_pos), obstacle.dcr, facecolor='none', edgecolor='black')
            dcr_radius_3 = plt.Circle((obstacle.x_pos, obstacle.z_pos), obstacle.dcr, facecolor='none', edgecolor='black')
            do_radius_1 = plt.Circle((obstacle.x_pos, obstacle.y_pos), obstacle.d0, facecolor='none', edgecolor='orange')
            do_radius_2 = plt.Circle((obstacle.y_pos, obstacle.z_pos), obstacle.d0, facecolor='none', edgecolor='orange')
            do_radius_3 = plt.Circle((obstacle.x_pos, obstacle.z_pos), obstacle.d0, facecolor='none', edgecolor='orange')

            ax.add_patch(dcr_radius_1)
            ax.add_patch(dcr_radius_2)
            ax.add_patch(dcr_radius_3)
            ax.add_patch(do_radius_1)
            ax.add_patch(do_radius_2)
            ax.add_patch(do_radius_3)
            
            art3d.pathpatch_2d_to_3d(dcr_radius_1, z=obstacle.z_pos, zdir="z")
            art3d.pathpatch_2d_to_3d(dcr_radius_2, z=obstacle.x_pos, zdir="x")
            art3d.pathpatch_2d_to_3d(dcr_radius_3, z=obstacle.y_pos, zdir="y")
            art3d.pathpatch_2d_to_3d(do_radius_1, z=obstacle.z_pos, zdir="z")
            art3d.pathpatch_2d_to_3d(do_radius_2, z=obstacle.x_pos, zdir="x")
            art3d.pathpatch_2d_to_3d(do_radius_3, z=obstacle.y_pos, zdir="y")

            # displaying shortest distance
            if obstacle.closest_point is not None:
                ax.plot([obstacle.x_pos, obstacle.closest_point.coordinates[0]], [obstacle.y_pos, obstacle.closest_point.coordinates[1]],\
                        [obstacle.z_pos, obstacle.closest_point.coordinates[2]], c='#a434eb')


    def update_position(self, dt):
        
        # flipping the velocities when the display dboundary is reached
        if not (self.x_pos <= self.display_max and self.x_pos >= self.display_min) and\
               (self.y_pos <= self.display_max and self.y_pos >= self.display_min) and\
               (self.z_pos <= self.display_max and self.z_pos >= self.display_min):
            self.velocity *= -1
        
        # applying the velocities
        self.x_pos += self.velocity[0]*dt
        self.y_pos += self.velocity[1]*dt
        self.z_pos += self.velocity[2]*dt


    def compute_closest_point(self, joint_pos):

        '''
        Identifies the point on the robot which is closest to the obstacle

        Parameters
        ----------
        joint_pos : the coordinates of each of the arm's joints ie. [(x1, y1, z1), (x2, y2, z2), ...]
        '''

        candidate_points = []

        # computing the shortest distance between the obstacle and every segment of the robot
        for segment_i in range(len(joint_pos)-1):

            seg_pos = None
            c_point = None
            distance = None
            
            intersect_info = intersect_point_line((self.x_pos, self.y_pos, self.z_pos), 
                                                  (joint_pos[segment_i][0], joint_pos[segment_i][1], joint_pos[segment_i][2]), 
                                                  (joint_pos[segment_i + 1][0], joint_pos[segment_i + 1][1], joint_pos[segment_i + 1][2]))
            
            # handling when the nearest point is outside the segment
            if intersect_info[1] < 0 :
                c_point = (joint_pos[segment_i][0], joint_pos[segment_i][1], joint_pos[segment_i][2])
                seg_pos = 0
            elif intersect_info[1] > 1:
                c_point = (joint_pos[segment_i + 1][0], joint_pos[segment_i + 1][1], joint_pos[segment_i + 1][2])
                seg_pos = 1
            # handling when the nearest point is on the segment
            else:
                c_point = intersect_info[0]
                seg_pos = intersect_info[1]

            # computing the distance between the closest point ant the obstacle + adding to the list
            distance = math.sqrt((c_point[0] - self.x_pos)**2 + (c_point[1] - self.y_pos)**2 + (c_point[2]-self.z_pos)**2)
            candidate_points.append(ClosestPoint(c_point, distance, segment_i, seg_pos))

        # finding the closest point accross the whole robot arm
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
        first_joint_vector = np.array(prev_joint_positions[segment_i])
        second_joint_vector = np.array(prev_joint_positions[segment_i + 1])
        segment_vector = np.array([second_joint_vector[i] - first_joint_vector[i] for i in range(3)])
        cp_prev_vector = (segment_vector * self.closest_point.segment_pos) + first_joint_vector
       
        # calculating the closest point velocity + calculating the relative velocity
        cp_velocity = np.linalg.norm(np.array(self.closest_point.coordinates) - cp_prev_vector) / dt
        self.vrel = self.velocity_norm - cp_velocity

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
            vrep1 = self.k1 * ((self.d0 / (dmin - self.dcr)) - 1)

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
        v_z = self.closest_point.coordinates[2] - self.z_pos
        unit_v = np.array([v_x, v_y, v_z]) / math.sqrt(v_x**2 + v_y**2 + v_z**2)
        vrep = (vrep1 + vrep2) * unit_v

        return RepulsionVelocity(vrep, [self.x_pos, self.y_pos, self.z_pos], copy.deepcopy(self.closest_point))


    @staticmethod
    def seperate_jacobian_segments(dhp_matrix):

        ''' Returns the row indicies associated whith each robot segment '''

        row_i = 0
        segment_i = 0
        segment_indicies = {}

        # getting a list of the DHP (d) parameters
        d_params = [dhp_row[-1] for dhp_row in dhp_matrix]

        for row_i in range(len(d_params)):

            if row_i == 0: segment_indicies[str(segment_i)] = []

            if (not d_params[row_i] == 0) and (not row_i == 0) :
                segment_i += 1
                segment_indicies[str(segment_i)] = []
                segment_indicies[str(segment_i)].append(row_i)

            else: segment_indicies[str(segment_i)].append(row_i)

        return segment_indicies