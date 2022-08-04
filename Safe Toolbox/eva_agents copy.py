import numpy as np
import cvxpy as cp
import torch

class MPC_agent:
    def __init__(self, control_freq) -> None:
        self.horizon = 10
        self.control_freq = control_freq
        self.dt = 1 / control_freq
        ## target state
        self.x_ref = np.array([0.5, 0, 0, 0])
        ## model parameters
        self.A = np.array([[1, 0, self.dt, 0], 
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.B = np.array([[0.5*self.dt**2, 0],
                           [0, 0.5*self.dt**2],
                           [self.dt, 0],
                           [0, self.dt]])
        ## control parameters
        self.P = np.eye(4)
        self.Q = np.eye(2)
        
        ## constraint parameters
        self.D = np.block([[np.eye(4)],
                           [-np.eye(4)]])
        ## dynamics contr.
        self.v_lim = np.ones(4) * 0.25
        ## obstacle constr.
        self.area_left = np.array([-1, 1.8, 4, -0.2])
        self.area_middle = np.array([0, 1.8, 1, -1])
        self.area_right = np.array([1, 1.3, 0.5, -0.2])
    
    def get_action(self, state):
        x_init = state.copy()
        
        area_constr = None
        for area in [self.area_left, self.area_middle, self.area_right]:
            trans = np.block([[np.eye(2)],
                           [-np.eye(2)]])
            if (trans @ x_init[:2] <= area).all():
                area_constr = area
                break
        dyn_constr = np.zeros(8,)
        dyn_constr[:2] = area_constr[:2]
        dyn_constr[2:4] = self.v_lim[:2]
        dyn_constr[4:6] = area_constr[2:]
        dyn_constr[6:8] = self.v_lim[2:]
        
        # create the optimization problem
        x = cp.Variable((4, self.horizon + 1))
        u = cp.Variable((2, self.horizon))
        cost = 0
        constr = []
        for k in range(self.horizon):
            cost += cp.quad_form(x[:, k] - self.x_ref, self.P)
            cost += cp.quad_form(u[:, k], self.Q)
            constr.append(x[:, k+1] == self.A @ x[:, k] + self.B @ u[:, k])
            constr.append(self.D @ x[:, k] <= dyn_constr)
        constr.append(x[:, 0] == x_init)
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve()
        
        acc = u[:, 0].value
        tar_vel = x[2:, 1].value
        return tar_vel


class RL_Agent():
    def __init__(self) -> None:
        self.rl_policy = torch.load('saved_model/policy_drone_xz_LBAC_1.pkl'),
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _, _ = self.policy.sample(state)
        action = action.detach().cpu().numpy()[0]
        return action



if __name__ == "__main__":
    agent = MPC_agent(control_freq=10)
    state = np.array([0, 0])
    action = agent.get_action(state)
    print(action)
        
        
        
        
            
        
        
        