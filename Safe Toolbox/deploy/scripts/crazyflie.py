import time
from threading import Event
import sys
import rospy
from nav_msgs.msg import Odometry
import numpy as np
from scipy.spatial.transform import Rotation
import numpy.linalg as LA
import torch
import eva_agents

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.mem import MemoryElement
from cflib.crazyflie.mem import Poly4D
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.positioning.motion_commander import MotionCommander
from cflib.positioning.position_hl_commander import PositionHlCommander
from cflib.utils import uri_helper

state = np.zeros(13, )
opti_state = np.zeros(13, )
opti_state_euler = np.zeros(9, )
cf = None

# file_opti = open('./opti.csv', 'w')
# file_loco = open('./loco.csv', 'w')
# file_action = open('./action.csv', 'w')


class CrazyFlie:
    def __init__(self, uri) -> None:
        self.uri = uri_helper.uri_from_env(default=uri)
        self.log_pos = None
        self.log_ang = None
        self.log_ang_rate = None
        self.set_log()
        self.deck_attached_event = Event()
        self.file = open('./drone.csv', 'w')
        self.file.write("Time || Input(vx, vz) || State(x, z, vx, vz) \n")

    def set_log(self):
        self.log_pos = LogConfig(name='Position', period_in_ms=10)
        self.log_pos.add_variable('stateEstimate.x', 'float')
        self.log_pos.add_variable('stateEstimate.y', 'float')
        self.log_pos.add_variable('stateEstimate.z', 'float')
        self.log_pos.add_variable('stateEstimate.vx', 'float')
        self.log_pos.add_variable('stateEstimate.vy', 'float')
        self.log_pos.add_variable('stateEstimate.vz', 'float')
        self.log_pos.data_received_cb.add_callback(self.log_pos_callback)

        self.log_ang = LogConfig(name='Angle', period_in_ms=10)
        self.log_ang.add_variable('stateEstimate.qx', 'float')
        self.log_ang.add_variable('stateEstimate.qy', 'float')
        self.log_ang.add_variable('stateEstimate.qz', 'float')
        self.log_ang.add_variable('stateEstimate.qw', 'float')
        self.log_ang.data_received_cb.add_callback(self.log_ang_callback)

        self.log_ang_rate = LogConfig(name='Angle_rate', period_in_ms=10)
        self.log_ang_rate.add_variable('stateEstimateZ.rateRoll', 'float')
        self.log_ang_rate.add_variable('stateEstimateZ.ratePitch', 'float')
        self.log_ang_rate.add_variable('stateEstimateZ.rateYaw', 'float')
        self.log_ang_rate.data_received_cb.add_callback(
            self.log_ang_rate_callback)

    def log_pos_callback(self, timestamp, data, logconf):
        global state
        state[0] = data['stateEstimate.x']
        state[1] = data['stateEstimate.y']
        state[2] = data['stateEstimate.z']
        state[3] = data['stateEstimate.vx']
        state[4] = data['stateEstimate.vy']
        state[5] = data['stateEstimate.vz']

    def log_ang_callback(self, timestamp, data, logconf):
        global state
        state[6] = data['stateEstimate.qx']
        state[7] = data['stateEstimate.qy']
        state[8] = data['stateEstimate.qz']
        state[9] = data['stateEstimate.qw']

    def log_ang_rate_callback(self, timestamp, data, logconf):
        global state
        state[10] = data['stateEstimateZ.rateRoll'] / 1000
        state[11] = data['stateEstimateZ.ratePitch'] / 1000
        state[12] = data['stateEstimateZ.rateYaw'] / 1000

    def simple_connect(self):
        print("Yeah, I'm connected! :D")
        time.sleep(5)
        print("Now I will disconnect :'(")

    def param_deck_flow(self, _, value_str):
        value = int(value_str)
        if value:
            self.deck_attached_event.set()
            print('Deck is attached!')
        else:
            print('Deck is NOT attached!')

    def simple_task(self, scf, DEFAULT_HEIGHT=0.3):
        with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
            print("Take off!")
            time.sleep(3)
            mc.stop()

    def hover(self, scf):
        with PositionHlCommander(
                scf,
                x=0, y=0.0, z=0.0,
                default_velocity=0.3,
                default_height=0.5,
                controller=PositionHlCommander.CONTROLLER_PID) as pc:
            # Go to a coordinate
            pc.go_to(0, 0, 0.5)
            # pc.go_to(1, 0, 0.5)
            # pc.go_to(0, 0, 0.5)

    def clip(self, input, thres=0.5):
        if input > thres:
            input = thres
        elif input < -thres:
            input = -thres
        return input

    def PD_control(self, input_state):
        x_dot = -input_state[0]
        y_dot = -input_state[1]
        z_dot = -input_state[2] - input_state[5]
        yaw_dot = 0
        return x_dot, y_dot, z_dot, yaw_dot

    def rl_mode(self, scf, DEFAULT_HEIGHT=0.5, duration=25):
        global state, cf, opti_state, file_action
        with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
            print("Take off!")
            time.sleep(0.5)
            start_time = time.time()
            rl_controller = eva_agents.RL_Agent('models/policy_drone_xz_LBACv1_epi100_seed55.pkl')
            print("Start experiment!")
            control_mode = "PD_start"
            x_dot, y_dot, z_dot, yaw_dot = 0, 0, 0, 0
            while True:
                tracking_time = time.time() - start_time
                if control_mode == "PD_start":
                    input_state = opti_state.copy()
                    input_state[0] -= (-1.5)
                    input_state[2] -= 0.35
                    x_dot, y_dot, z_dot, yaw_dot = self.PD_control(input_state)
                    x_dot = self.clip(x_dot, thres=0.25)
                    y_dot = self.clip(y_dot, thres=0.25)
                    z_dot = self.clip(z_dot, thres=0.25)
                    if tracking_time > 8:
                        print("Start RL!")
                        control_mode = "RL"
                
                dist = np.sqrt(np.square(opti_state[0]) + np.square(opti_state[2]-0.5))
                # print("dist: ", dist)
                if (dist < 0.4) and control_mode == "RL":
                    print("Start PD control!")
                    control_mode = "PD_final"

                if control_mode == "PD_final":
                    input_state = opti_state.copy()
                    input_state[2] -= 0.5
                    x_dot, y_dot, z_dot, yaw_dot = self.PD_control(input_state)
                    x_dot = self.clip(x_dot, thres=0.25)
                    y_dot = self.clip(y_dot, thres=0.25)
                    z_dot = self.clip(z_dot, thres=0.25)
                    
                if control_mode == "RL":
                    '''x-z collision avoidance'''
                    input_state = np.array([opti_state[0], opti_state[2], opti_state[3], opti_state[5]])
                    x_dot, z_dot = rl_controller.get_action(input_state)
                    x_dot = self.clip(x_dot, thres=0.25)
                    z_dot = self.clip(z_dot, thres=0.25)
                    y_dot = -1.5*opti_state[1]
                    y_dot = self.clip(y_dot, thres=0.25)
                    yaw_dot = 0
                
                print(opti_state[0], opti_state[2])
                cf.commander.send_velocity_world_setpoint(x_dot, y_dot, z_dot, yaw_dot)
                self.file.write('{},{},{},{},{},{},{}\n'.format(tracking_time,x_dot, z_dot, opti_state[0], opti_state[2], opti_state[3], opti_state[5]))
                if opti_state[2] > 2 or opti_state[0] < -2:
                    break
                time.sleep(0.02)
                if tracking_time > duration:
                    break
            mc.stop()
            self.file.close()
                
            # '''hover'''
            # opti_state_euler[2] -= 1
            # x_dot, y_dot, z_dot = rl_controller.control(opti_state_euler[0:6])
            # yaw_dot = 0
            # x_dot = self.clip(x_dot, thres=1)
            # y_dot = self.clip(y_dot, thres=1)
            # z_dot = self.clip(z_dot, thres=1)
           # yaw_dot = self.clip(yaw_dot, thres=1)
            # x_dot /= 4
            # y_dot /= 4
            # z_dot /= 4
            # yaw_dot /= 4
            # file_action.write('{},{},{},{}\n'.format(
            #     x_dot, y_dot, z_dot, yaw_dot))
            # cf.commander.send_velocity_world_setpoint(x_dot, y_dot, z_dot, yaw_dot)

            # '''hover'''
            # x_dot, y_dot, z_dot, yaw_dot = self.PD_control(opti_state)
            # x_dot = self.clip(x_dot, thres=8)
            # y_dot = self.clip(y_dot, thres=8)
            # z_dot = self.clip(z_dot, thres=8)
            # yaw_dot = self.clip(yaw_dot, thres=8)
            # x_dot /= 4
            # y_dot /= 4
            # z_dot /= 4
            # yaw_dot /= 4
            # cf.commander.send_velocity_world_setpoint(x_dot, y_dot, z_dot, yaw_dot)
            # file_action.write('{},{},{},{}\n'.format(
            #     x_dot, y_dot, z_dot, yaw_dot))

            # x_dot, y_dot, z_dot, yaw_dot = self.to_body_frame(
            #     rl_controller.control(opti_state))
            # mc._set_vel_setpoint(x_dot, y_dot, z_dot, yaw_dot)
            # print(self.clip(x_dot)/4, self.clip(y_dot)/4, self.clip(z_dot)/4)



    def to_body_frame(self, action):
        vel = action[0:3].reshape(3, 1)
        ang_vel = np.array([0, 0, action[-1]]).reshape(3, 1)
        r = Rotation.from_quat(state[6:10])
        tf_matrix = r.as_matrix()
        new_action = np.zeros(4, )
        new_action[0:3] = tf_matrix @ vel
        new_action[3] = (tf_matrix @ ang_vel)[-1]
        return new_action

    def run(self):
        global cf
        cflib.crtp.init_drivers()
        with SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            cf = scf.cf
            # init_optitrack_listener()
            scf.cf.log.add_config(self.log_pos)
            scf.cf.log.add_config(self.log_ang)
            self.log_pos.start()
            self.log_ang.start()
            scf.cf.param.add_update_callback(group='deck', name='bcFlow2',
                                             cb=self.param_deck_flow)
            if not self.deck_attached_event.wait(timeout=5):
                print('No flow deck detected!')
                sys.exit(1)

            print('starting rl mode')
            input()
            self.rl_mode(scf)  # to be changed

            self.log_pos.stop()
            self.log_ang.stop()


# def init_optitrack_listener():
#     rospy.init_node('crayflie', anonymous=True)
#     sub_robot_state_gazebo = rospy.Subscriber("/Bebop1/position_velocity_orientation_estimation", Odometry,
#                                               receive_robot_state)


def receive_robot_state(odom_robot):
    global opti_state
    opti_state = np.zeros(13, )
    opti_state[0:3] = [odom_robot.pose.pose.position.x,
                       odom_robot.pose.pose.position.y,
                       odom_robot.pose.pose.position.z, ]
    opti_state[3:6] = [odom_robot.twist.twist.linear.x,
                       odom_robot.twist.twist.linear.y,
                       odom_robot.twist.twist.linear.z]
    opti_state[6:10] = [odom_robot.pose.pose.orientation.x,
                        odom_robot.pose.pose.orientation.y,
                        odom_robot.pose.pose.orientation.z,
                        odom_robot.pose.pose.orientation.w]
    opti_state[10:13] = [odom_robot.twist.twist.angular.x,
                         -odom_robot.twist.twist.angular.y,
                         odom_robot.twist.twist.angular.z]
    # global file_opti
    # file_opti.write('{},{},{} || {},{},{} || {},{},{},{} || {},{},{} ||\n'.format(
    #     opti_state[0], opti_state[1], opti_state[2], opti_state[3], opti_state[4], opti_state[5], opti_state[6], opti_state[7], opti_state[8], opti_state[9], opti_state[10], opti_state[11], opti_state[12]))

    opti_state_euler[0:6] = opti_state[0:6]
    r = Rotation.from_quat(opti_state[6:10])
    opti_state_euler[6:9] = r.as_euler('xyz', degrees=False)

    # global file_loco
    # file_loco.write('{},{},{} || {},{},{} || {},{},{},{} || {},{},{} ||\n'.format(
    #     state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], state[8], state[9], state[10], state[11], state[12]))

    # cf.extpos.send_extpose(state[0], state[1], state[2], state[6], state[7], state[8], state[9])


if __name__ == '__main__':
    uri = "radio://0/82/2M/E7E7E7E7E7"
    flie = CrazyFlie(uri)
    flie.run()
    # file_opti.close()
    # file_loco.close()
    # file_action.close()
