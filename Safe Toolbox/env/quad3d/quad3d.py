import numpy as np
from gym import core, spaces
import gym
from numpy.linalg import inv, norm
import scipy.integrate
from scipy.spatial.transform import Rotation


def quat_dot(quat, omega):
    """
    Parameters:
        quat, [i,j,k,w]
        omega, angular velocity of body in body axes

    Returns
        quat_dot, [i,j,k,w]

    """
    # Adapted from "Quaternions And Dynamics" by Basile Graf.
    (q0, q1, q2, q3) = (quat[0], quat[1], quat[2], quat[3])
    G = np.array([[q3, q2, -q1, -q0], [-q2, q3, q0, -q1], [q1, -q0, q3, -q2]])
    quat_dot = 0.5 * G.T @ omega
    # Augment to maintain unit quaternion.
    quat_err = np.sum(quat ** 2) - 1
    quat_err_grad = 2 * quat
    quat_dot = quat_dot - quat_err * quat_err_grad
    return quat_dot


class Quad3D(gym.Env):
    """
    Quadrotor forward dynamics model.
    """

    def __init__(
        self, seed=None, clipped_action=True,
    ):
        self._clipped_action = clipped_action
        self._max_episode_steps = 3000

        self._state_low = np.array(
            [-10, -10, -10, -5, -5, -5, -1, -1, -1, -1, -1, -1, -1,]
        )
        self._state_high = np.array([10, 10, 10, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1,])
        self.force_max = np.array([0.2, 0.2, 0.2, 0.1])
        self.action_space = spaces.Box(
            low=-self.force_max, high=self.force_max, dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-self._state_high, high=self._state_high, dtype=np.float32
        )
        self.seed(seed)

        self.mass = 0.03  # kg
        self.Ixx = 1.43e-5  # kg*m^2
        self.Iyy = 1.43e-5  # kg*m^2
        self.Izz = 2.89e-5  # kg*m^2
        self.arm_length = 0.046  # meters
        self.rotor_speed_min = 0  # rad/s
        self.rotor_speed_max = 2500  # rad/s
        self.k_thrust = 2.3e-08  # N/(rad/s)**2
        self.k_drag = 7.8e-11  # Nm/(rad/s)**2
        I = np.array(
            [[1.43e-5, 0, 0], [0, 1.43e-5, 0], [0, 0, 2.89e-5]]
        )  # inertial tensor in m^2 kg
        self.rotor_speed_min = 0
        self.rotor_speed_max = 2500
        self.inertia = I
        self.invI = np.linalg.inv(I)
        self.g = 9.81
        self.arm_length = 0.046  # arm length in m
        self.k_thrust = 2.3e-08
        self.k_drag = 7.8e-11

        # Precomputes
        k = self.k_drag / self.k_thrust
        L = self.arm_length
        self.to_TM = np.array(
            [[1, 1, 1, 1], [0, L, 0, -L], [-L, 0, L, 0], [k, -k, k, -k]]
        )
        self.inv_inertia = inv(self.inertia)
        self.weight = np.array([0, 0, -self.mass * self.g])
        self.t = 0
        self.dt_step = 0.01

    def reset(self, yaw=0, pitch=0, roll=0):
        position = [0, 0, 0]
        self.t = 0
        s = np.zeros(13)
        flag_reset = True
        while flag_reset:
            # s[0] = -1 + 2 * np.random.rand()
            # s[1] = -1 + 2 * np.random.rand()
            # s[2] = 2 * np.random.rand()
            s[0] = 0
            s[1] = 0
            s[2] = 0.5
            r = Rotation.from_euler("zxy", [yaw, roll, pitch], degrees=True)
            quat = r.as_quat()
            s[6] = quat[0]
            s[7] = quat[1]
            s[8] = quat[2]
            s[9] = quat[3]
            flag_reset = self.safe(s)
        self.state = self._unpack_state(s)
        return s

    def safe(self, s):
        done = True
        if s[2] > 0 and (np.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2) < 3.5):
            done = False
        return done

    def unsafe(self, s):
        done = False
        if s[2] < -0.3:
            done = True
        if np.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2) > 3.5:
            done = True
        return done

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        target = circle_trajectory(action)
        state = self.state
        control_variable = self.control(target, state)
        cmd_rotor_speeds = control_variable["cmd_motor_speeds"]
        rotor_speeds = np.clip(
            cmd_rotor_speeds, self.rotor_speed_min, self.rotor_speed_max
        )
        rotor_thrusts = self.k_thrust * rotor_speeds ** 2
        TM = self.to_TM @ rotor_thrusts
        T = TM[0]  # u1
        M = TM[1:4]  # u2
        self.t = self.t + 1
        # Form autonomous ODE for constant inputs and integrate one time step.
        def s_dot_fn(t, s):
            return self._s_dot_fn(s, T, M)

        # turn state into list
        s = Quad3D._pack_state(self.state)
        sol = scipy.integrate.solve_ivp(
            s_dot_fn, (0, self.dt_step), s, first_step=self.dt_step
        )
        s = sol["y"][:, -1]
        # turn state back to dict
        self.state = Quad3D._unpack_state(s)
        # s = self._pack_state(self.state)
        # Re-normalize unit quaternion.
        self.state["q"] = self.state["q"] / norm(self.state["q"])
        reward = -norm(s[0:3])
        done = False

        if self.t == 1000:
            done = True
        if self.unsafe(s):
            reward = -4000
            done = True
        out = False
        return s, reward, done, out

    def _s_dot_fn(self, s, u1, u2):
        """
        Compute derivative of state for quadrotor given fixed control inputs as
        an autonomous ODE.
        """
        state = Quad3D._unpack_state(s)
        # Position derivative.
        x_dot = state["v"]
        # Velocity derivative.
        F = u1 * Quad3D.rotate_k(state["q"])
        v_dot = (self.weight + F) / self.mass
        # Orientation derivative.
        q_dot = quat_dot(state["q"], state["w"])
        # Angular velocity derivative.
        omega = state["w"]
        omega_hat = Quad3D.hat_map(omega)
        w_dot = self.inv_inertia @ (u2 - omega_hat @ (self.inertia @ omega))
        # Pack into vector of derivatives.
        s_dot = np.zeros((13,))
        s_dot[0:3] = x_dot
        s_dot[3:6] = v_dot
        s_dot[6:10] = q_dot
        s_dot[10:13] = w_dot
        return s_dot

    def control(self, flat_output, state):
        """
        :param desired_state: pos, vel, acc, yaw, yaw_dot
        :param current_state: pos, vel, euler, omega
        :return:
        """
        # self.Kp = np.diag([15, 15, 30])  # 8.5
        self.Kp = np.diag([0, 0, 0])  # 8.5
        self.Kd = np.diag([12, 12, 10])  # 4.5
        # altitude control gains
        self.K_R = np.diag([3000, 3000, 3000])  # 2500 400
        self.K_w = np.diag([300, 300, 300])  # 60 50
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))
        error_pos = state.get("x") - flat_output.get("x")
        error_vel = state.get("v") - flat_output.get("x_dot")
        error_vel = np.array(error_vel).reshape(3, 1)
        error_pos = np.array(error_pos).reshape(3, 1)

        # Equation 26
        rdd_des = (
            np.array(flat_output.get("x_ddot")).reshape(3, 1)
            - np.matmul(self.Kd, error_vel)
            - np.matmul(self.Kp, error_pos)
        )
        # Equation 28
        F_des = (self.mass * rdd_des) + np.array([0, 0, self.mass * self.g]).reshape(
            3, 1
        )  # (3 * 1)

        # Find Rotation matrix
        R = Rotation.as_matrix(
            Rotation.from_quat(state.get("q"))
        )  # Quaternions to Rotation Matrix
        # print(R.shape)
        # Equation 29, Find u1
        b3 = R[0:3, 2:3]

        # print(b3)
        u1 = np.matmul(b3.T, F_des)  # u1[0,0] to access value
        # print(np.transpose(b3))

        # ----------------------- the following is to  find u2 ---------------------------------------------------------
        # Equation 30
        b3_des = F_des / np.linalg.norm(F_des)  # 3 * 1
        a_Psi = np.array(
            [np.cos(flat_output.get("yaw")), np.sin(flat_output.get("yaw")), 0]
        ).reshape(
            3, 1
        )  # 3 * 1
        b2_des = np.cross(b3_des, a_Psi, axis=0) / np.linalg.norm(
            np.cross(b3_des, a_Psi, axis=0)
        )
        b1_des = np.cross(b2_des, b3_des, axis=0)

        # Equation 33
        R_des = np.hstack((b1_des, b2_des, b3_des))
        # print(R_des)

        # Equation 34
        # R_temp = 0.5 * (np.matmul(np.transpose(R_des), R) - np.matmul(np.transpose(R), R_des))
        temp = R_des.T @ R - R.T @ R_des
        R_temp = 0.5 * temp
        # orientation error vector
        e_R = 0.5 * np.array([-R_temp[1, 2], R_temp[0, 2], -R_temp[0, 1]]).reshape(3, 1)
        # Equation 35
        u2 = self.inertia @ (
            -self.K_R @ e_R - self.K_w @ (np.array(state.get("w")).reshape(3, 1))
        )

        gama = self.k_drag / self.k_thrust
        Len = self.arm_length
        cof_temp = np.array(
            [1, 1, 1, 1, 0, Len, 0, -Len, -Len, 0, Len, 0, gama, -gama, gama, -gama]
        ).reshape(4, 4)

        u = np.vstack((u1, u2))

        F_i = np.matmul(np.linalg.inv(cof_temp), u)

        for i in range(4):
            if F_i[i, 0] < 0:
                F_i[i, 0] = 0
                cmd_motor_speeds[i] = self.rotor_speed_max
            cmd_motor_speeds[i] = np.sqrt(F_i[i, 0] / self.k_thrust)
            if cmd_motor_speeds[i] > self.rotor_speed_max:
                cmd_motor_speeds[i] = self.rotor_speed_max

        cmd_thrust = u1[0, 0]
        cmd_moment[0] = u2[0, 0]
        cmd_moment[1] = u2[1, 0]
        cmd_moment[2] = u2[2, 0]

        cmd_q = Rotation.as_quat(Rotation.from_matrix(R_des))

        control_input = {
            "cmd_motor_speeds": cmd_motor_speeds,
            "cmd_thrust": cmd_thrust,
            "cmd_moment": cmd_moment,
            "cmd_q": cmd_q,
        }
        return control_input

    @classmethod
    def rotate_k(cls, q):
        """
        Rotate the unit vector k by quaternion q. This is the third column of
        the rotation matrix associated with a rotation by q.
        """
        return np.array(
            [
                2 * (q[0] * q[2] + q[1] * q[3]),
                2 * (q[1] * q[2] - q[0] * q[3]),
                1 - 2 * (q[0] ** 2 + q[1] ** 2),
            ]
        )

    @classmethod
    def hat_map(cls, s):
        """
        Given vector s in R^3, return associate skew symmetric matrix S in R^3x3
        """
        return np.array([[0, -s[2], s[1]], [s[2], 0, -s[0]], [-s[1], s[0], 0]])

    @classmethod
    def _pack_state(cls, state):
        """
        Convert a state dict to Quadrotor's private internal vector representation.
        """
        s = np.zeros((13,))
        s[0:3] = state["x"]
        s[3:6] = state["v"]
        s[6:10] = state["q"]
        s[10:13] = state["w"]
        return s

    @classmethod
    def _unpack_state(cls, s):
        """
        Convert Quadrotor's private internal vector representation to a state dict.
        """
        state = {"x": s[0:3], "v": s[3:6], "q": s[6:10], "w": s[10:13]}
        return state


def circle_trajectory(action):

    x = np.zeros((3,))
    x_dot = np.zeros((3,))
    x_ddot = np.zeros((3,))
    x_dddot = np.zeros((3,))
    x_ddddot = np.zeros((3,))
    pos = [0, 0, 0]
    vel = [action[0], action[1], action[2]]
    acc = [0, 0, 0]
    yaw = action[3]
    yaw_dot = 0
    flat_output = {
        "x": np.array(pos),
        "x_dot": np.array(vel),
        "x_ddot": np.array(acc),
        "x_dddot": x_dddot,
        "x_ddddot": x_ddddot,
        "yaw": yaw,
        "yaw_dot": yaw_dot,
    }
    return flat_output
