import numpy as np
import cvxpy as cp
import torch
from kine_car_nmpc import *

class Drone_MPC_Agent:
    def __init__(self, control_freq) -> None:
        self.horizon = 10
        self.control_freq = control_freq
        self.dt = 1 / control_freq
        ## target state
        self.x_ref = np.array([0, 1.1]) 
        ## model parameters
        self.A = np.array([[1, 0], 
                           [0, 1]])
        self.B = np.array([[self.dt, 0],
                           [0, self.dt]])
        ## control parameters
        self.P = np.eye(2)
        self.Q = np.eye(2)
        
        ## constraint parameters
        self.D = np.block([[np.eye(2)],
                           [-np.eye(2)]])
        ## dynamics contr.
        self.u_lim = np.ones(4) * 0.25
        ## obstacle constr.
        self.area_left = np.array([-1, 1.8, 4, -0.2])
        self.area_middle = np.array([0, 1.8, 1, -1])
        self.area_right = np.array([1, 1.3, 0.5, -0.2])
    
    def get_action(self, state):
        x_init = state[:2]
        
        area_constr = None
        for area in [self.area_left, self.area_middle, self.area_right]:
            if (self.D @ x_init <= area).all():
                area_constr = area
                break
        
        # create the optimization problem
        x = cp.Variable((2, self.horizon + 1))
        u = cp.Variable((2, self.horizon))
        cost = 0
        constr = []
        for k in range(self.horizon):
            cost += cp.quad_form(x[:, k] - self.x_ref, self.P)
            cost += cp.quad_form(u[:, k], self.Q)
            constr.append(x[:, k+1] == self.A @ x[:, k] + self.B @ u[:, k])
            constr.append(self.D @ u[:, k] <= self.u_lim)
            constr.append(self.D @ x[:, k] <= area_constr)
        constr.append(x[:, 0] == x_init)
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve()
        
        u = u[:, 0].value
        return u


class Car_NMPC_Agent(object):
    def __init__(self, control_freq) -> None:
        self.dt = 1 / control_freq
        self.horizon = 50
        self.param = MPC_Formulation_Param()
        self.param.set_horizon(dt=self.dt, N=self.horizon)
        self.solver = acados_mpc_solver_generation(
            self.param, collision_avoidance=False)
    
    def get_action(self, state):
        x_init = state.copy()
        self.solver.set(0, "lbx", x_init)
        self.solver.set(0, "ubx", x_init)
        status = self.solver.solve()
        if status != 0:
            print("acados returned status {} in closed loop iteration {}.".format(
                status))
        u = self.solver.get(0, "u")
        return u
    
    
class RL_Agent():
    def __init__(self, PATH) -> None:
        self.policy = torch.load(PATH, map_location=torch.device('cpu'))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        # print(self.device)
        # input()
        action, _, _ = self.policy.sample(state)
        action = action.detach().cpu().numpy()[0]
        return action



if __name__ == "__main__":
    PATH = "models/policy_drone_xz_SQRL_epi3000_seed1.pkl"
    agent = RL_Agent(PATH)
    print(agent.policy.eval())
        
        
        
        
            
        
        
        