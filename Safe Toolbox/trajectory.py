import numpy as np
import numpy.linalg as LA


class Trajectory:
    def __init__(self):
        self.heading = None
        self.dt = 0.01
    
    def tj_from_line(self, start_pos, end_pos, time_ttl, t_c):
        v_max = (end_pos - start_pos) * 2 / time_ttl
        if (t_c >= 0 and t_c < time_ttl/2):
            vel = v_max*t_c/(time_ttl/2)
            pos = start_pos + t_c*vel/2
            acc = v_max/(time_ttl/2)
        elif (t_c >= time_ttl/2 and t_c <= time_ttl):
            vel = v_max * (time_ttl - t_c) / (time_ttl / 2)
            pos = end_pos - (time_ttl - t_c) * vel / 2
            acc = - v_max/(time_ttl / 2)
        else:
            if (type(start_pos) == int) or (type(start_pos) == float):
                pos, vel, acc = 0, 0, 0
            else:
                pos, vel, acc = np.zeros(start_pos.shape), np.zeros(
                    start_pos.shape), np.zeros(start_pos.shape)
        return pos, vel, acc
    
    def oneline(self, t):
        T1 = 5
        points = []
        points.append(np.array([-2, 0, 0]))
        points.append(np.array([2, 0, 0]))
        if (0 < t) and (t <= T1):
            pos, vel, acc = self.tj_from_line(points[0], points[1], T1, t)
        else:
            pos, vel, acc = points[-1], np.zeros((3, 1)), np.zeros((3, 1))
        ang = 0
        omega = 0
        return pos, vel, acc, ang, omega
    
    def half_circle(self, t):
        T = 12
        radius = 2
        angle, _, _ = self.tj_from_line(0, np.pi, T, t)
        angle2, _, _ = self.tj_from_line(0, np.pi, T, t+self.dt)
        angle3, _, _ = self.tj_from_line(0, np.pi, T, t+2*self.dt)
        pos = np.array([radius*(np.cos(angle)), radius*np.sin(angle) - 1.2, 0])
        pos2 = np.array([radius*(np.cos(angle2)-0),
                        radius*np.sin(angle2) - 1.2, 0])
        pos3 = np.array([radius*(np.cos(angle3)-0),
                        radius*np.sin(angle3) - 1.2, 0])
        vel = (pos2 - pos)/self.dt
        vel2 = (pos3 - pos2)/self.dt
        acc = (vel2 - vel)/self.dt
        try:
            if self.heading == None:
                self.heading = vel/LA.norm(vel)
        except:
            pass
        ang = angle + np.pi/2
        omega = self.get_omega(vel)
        return pos, vel, acc, ang, omega

    def get_omega(self, vel):
        if np.allclose(vel, np.zeros((3, 1))):
            vel += 1e-5
        curr_heading = vel/LA.norm(vel)
        prev_heading = self.heading
        curr_heading = curr_heading.flatten()
        prev_heading = prev_heading.flatten()
        cross_prod = np.cross(prev_heading, curr_heading)
        sine = cross_prod[2]
        omega = np.arcsin(sine) / self.dt
        self.heading = curr_heading
        return omega

if __name__ == "__main__":
    trajectory = Trajectory()
    pos, vel, acc = trajectory.oneline(t=0.1)
    print(pos)
