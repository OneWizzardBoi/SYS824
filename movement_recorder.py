import json
import os.path
import numpy as np
import matplotlib.pyplot as plt

class MovementRecorder:

    ''' Class for the recording and analysis of point series '''

    scatter_colors = ['red', 'green', 'blue']

    def __init__(self, data_file_path):
        
        self.point_data = {}
        self.data_file_path = data_file_path

        # if the specified file already exists, load it
        if os.path.isfile(self.data_file_path):
            self.load_positions_file()


    def add_series(self, key):
        
        if not key in self.point_data:
            self.point_data[key] = [] 
        else: 
            raise ValueError("Specified series already exists")

    def remove_series(self, key):
        
        if key in self.point_data:
            del self.point_data[key]
        else:
            raise ValueError("Specified series does not exist")


    def add_point(self, key, coordinates):
        
        if key in self.point_data:
            self.point_data[key].append(coordinates)
        else:
            raise ValueError("Specified series does not exist")


    def write_positions_file(self):
        with open(self.data_file_path, "w") as data_f:
            json.dump(self.point_data, data_f)

    def load_positions_file(self):
        with open(self.data_file_path) as data_f:
            self.point_data = json.load(data_f)


    def trace_positions_graph(self):
        
        c_i = 0

        for key, series in self.point_data.items():

            scatter_color = self.scatter_colors[c_i]
            X = [coordinates[0] for coordinates in self.point_data[key]]
            Y = [coordinates[1] for coordinates in self.point_data[key]]
            
            plt.plot(X, Y, color=scatter_color, label=key)
            plt.plot(X, Y, 'o', color=scatter_color)
            plt.legend()

            c_i += 1

        plt.show()
        

    def count_velocity_switches(self, dt):
        
        # going through the series
        for key, series in self.point_data.items():

            X_vel = []
            Y_vel = []
            X = [coordinates[0] for coordinates in self.point_data[key]]
            Y = [coordinates[1] for coordinates in self.point_data[key]]

            # computing the point velocities
            for i in range(len(series)-1):
                X_vel.append((X[i+1] - X[i])/dt)
                Y_vel.append((Y[i+1] - Y[i])/dt)

            # counting velocity sign changes
            switch_count = 0
            for i in range(len(X_vel)-1):
                if (X_vel[i] * X_vel[i+1]) < 0: switch_count += 1
                elif (Y_vel[i] * Y_vel[i+1]) < 0: switch_count += 1

            print(f"Number of velocity sign switches for {key} : {switch_count}")


if __name__ == "__main__":

    point_data_file_path = "/home/one_wizard_boi/Documents/Projects/SYS824/SYS824/point_data.json" 

    m_recorder = MovementRecorder(point_data_file_path)
    m_recorder.trace_positions_graph()
    m_recorder.count_velocity_switches(0.1)