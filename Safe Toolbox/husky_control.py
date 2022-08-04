#!/usr/bin/env python3
import rospy
from trajectory import *
import torch
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from scipy.spatial.transform import Rotation as R
import time
import eva_agents


class Husky_Control(object):
    def __init__(self):
        rospy.init_node("husky_control")
        self.traj = Trajectory().half_circle
        # self.agent = eva_agents.RL_Agent(PATH='models/policy_kine_car_RSPO_epi2000.pkl')
        self.agent = eva_agents.Car_NMPC_Agent(control_freq = 10)
        self.file = open('./half_circle_MPC_fast_traj.csv', 'w')
        # self.husky_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.husky_pub = rospy.Publisher('husky_velocity_controller/cmd_vel', Twist, queue_size=1)
        # self.husky_sub = rospy.Subscriber("/Bebop1/pose", PoseStamped, self.send_cmd)
        self.husky_sub = rospy.Subscriber("husky_velocity_controller/odom", Odometry, self.send_cmd)
        self.start_time = time.time()

    def step(self, action):
        tracking_time = time.time() - self.start_time

        vel_x_e = action[0]
        omega_e = action[1]
        vel_x_e = np.clip(vel_x_e, -0.5, 0.5)
        omega_e = np.clip(omega_e, -np.pi / 6, np.pi / 6)

        twist = Twist()

        vel_des = 0.5
        omega_des = 0

        twist.linear.x = vel_x_e + LA.norm(vel_des)
        twist.angular.z = omega_e + omega_des
        # print("===========================")
        # print(twist.linear.x, twist.angular.z)

        if tracking_time > 20:
            twist.linear.x = 0
            twist.angular.z = 0

        # print('ready to publish')
        # input()
        self.husky_pub.publish(twist)



    def send_cmd(self, msg):
        real_x = msg.pose.pose.position.x
        real_y = msg.pose.pose.position.y
        rot = R.from_quat([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        real_theta = rot.as_euler('xyz')[2]
        if (real_theta < 0):
            real_theta += 2*np.pi
        tracking_time = time.time() - self.start_time
        
        pos_des, vel_des, acc_des, theta_des, omega_des = self.traj(tracking_time)
  
        x_des, y_des, _ = pos_des
        
        # transform to husky frame
        x_err = (real_x - x_des) * np.cos(theta_des) + (real_y - y_des) * np.sin(theta_des)
        y_err = (real_y - y_des) * np.cos(theta_des) - (real_x - x_des) * np.sin(theta_des)
        theta_err =  real_theta - theta_des

        state = np.array([x_err, y_err, theta_err])

        self.file.write('{},{},{},{},{},{},{}\n'.format(tracking_time, real_x, real_y, real_theta, x_des, y_des, theta_des))
        
        vel_x_e, omega_e = self.agent.get_action(state)
        vel_x_e = np.clip(vel_x_e, -0.5, 0.5)
        omega_e = np.clip(omega_e, -np.pi/6, np.pi/6)
        
        
        twist = Twist()
        
        vel_des = 0.5
        omega_des = 0
        
        twist.linear.x = vel_x_e + LA.norm(vel_des)
        twist.angular.z = omega_e + omega_des
        # print("===========================")
        # print(twist.linear.x, twist.angular.z)
        
        if tracking_time > 20:
            twist.linear.x = 0
            twist.angular.z = 0
        
        self.husky_pub.publish(twist)

    def reset(self):
        pass
        
if __name__ == "__main__":
    controller  = Husky_Control()
    rospy.spin()
        