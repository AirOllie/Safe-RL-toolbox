import numpy as np
from dataclasses import dataclass
import casadi as cd
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver



@dataclass
class MPC_Formulation_Param:
    v_ref = 0.5
    omega_ref = 0

    # control bound
    v_e_lim = 0.5
    omega_e_ref = np.pi/3
    
    # state cost weights
    q_x_e = 100
    q_y_e = 100
    q_theta_e = 100

    r_v_e = 50
    r_omega_e = 50
    
    # terminal cost weights
    q_x_e_ter = 100
    q_y_e_ter = 100
    q_theta_e_ter = 100
    
    def set_horizon(self, dt, N):
        self.N = N
        self.dt = dt
        self.Tf = N * dt
    
    


def acados_mpc_solver_generation(mpc_form_param, collision_avoidance = False):
    # Acados model
    model = AcadosModel()
    model.name = "mav_nmpc_tracker_model"

    # state
    x_e = cd.MX.sym('x_e')
    y_e = cd.MX.sym('y_e')
    theta_e = cd.MX.sym('theta_e')
    x = cd.vertcat(x_e, y_e, theta_e)

    # control
    v_e = cd.MX.sym('v_e')
    omega_e = cd.MX.sym('omega_e')
    u = cd.vertcat(v_e, omega_e)

    # state derivative
    x_e_dot = cd.MX.sym('x_e_dot')
    y_e_dot = cd.MX.sym('y_e_dot')
    theta_e_dot = cd.MX.sym('theta_e_dot')
    x_dot = cd.vertcat(x_e_dot, y_e_dot, theta_e_dot)

    # dynamics
    dyn_f_expl = cd.vertcat(
        (mpc_form_param.v_ref + v_e) * cd.cos(theta_e) - mpc_form_param.v_ref + mpc_form_param.omega_ref * y_e,
        (mpc_form_param.v_ref + v_e) * cd.sin(theta_e) - mpc_form_param.omega_ref * x_e,
        omega_e
    )
    dyn_f_impl = x_dot - dyn_f_expl

    # acados mpc model
    model.x = x
    model.u = u
    model.xdot = x_dot
    model.f_expl_expr = dyn_f_expl
    model.f_impl_expr = dyn_f_impl

    # Acados ocp
    ocp = AcadosOcp()
    ocp.model = model

    # ocp dimension
    ocp.dims.N = mpc_form_param.N
    nx = 3
    nu = 2
    ny = nx + nu  # TODO
    ny_e = 3  # terminal cost, penalize on pos, pos_dot

    # initial condition, can be changed in real time
    ocp.constraints.x0 = np.zeros(nx)

    # cost terms
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    Vx = np.block([[np.eye(nx)],
                   [np.zeros((nu, nx))]])
    ocp.cost.Vx = Vx
    Vu = np.block([[np.zeros((nx, nu))],
                   [np.eye(nu)]]) 
    ocp.cost.Vu = Vu
    Vx_e = np.eye(3)
    ocp.cost.Vx_e = Vx_e
    # weights, changed in real time
    ocp.cost.W = np.diag([mpc_form_param.q_x_e, mpc_form_param.q_y_e, mpc_form_param.q_theta_e,
                          mpc_form_param.r_v_e, mpc_form_param.r_omega_e])
    ocp.cost.W_e =  np.diag([mpc_form_param.q_x_e_ter, mpc_form_param.q_y_e_ter, mpc_form_param.q_theta_e_ter])
    
    
    # reference for tracking, can be changed in real time; remains zeros in our case
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # set control bound
    ocp.constraints.lbu = np.array(
        [-mpc_form_param.r_v_e, -mpc_form_param.r_omega_e])
    ocp.constraints.ubu = np.array(
        [mpc_form_param.r_v_e, mpc_form_param.r_omega_e])
    ocp.constraints.idxbu = np.array(range(nu))

    # solver options
    # horizon
    ocp.solver_options.tf = mpc_form_param.Tf
    # qp solver
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'    # PARTIAL_CONDENSING_HPIPM FULL_CONDENSING_QPOASES
    ocp.solver_options.qp_solver_cond_N = 5
    ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.qp_solver_warm_start = 1
    # nlp solver
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.nlp_solver_max_iter = 100
    ocp.solver_options.nlp_solver_tol_eq = 1E-3
    ocp.solver_options.nlp_solver_tol_ineq = 1E-3
    ocp.solver_options.nlp_solver_tol_comp = 1E-3
    ocp.solver_options.nlp_solver_tol_stat = 1E-3
    # hessian
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    # integrator
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 3
    # print
    ocp.solver_options.print_level = 0

    # Acados solver
    print("Starting solver generation...")
    solver = AcadosOcpSolver(ocp, json_file='ACADOS_nmpc_tracker_solver.json')
    print("Solver generated.")
    return solver


if __name__ == "__main__":
    param = MPC_Formulation_Param()
    param.set_horizon(0.01, 10)
    acados_mpc_solver_generation(param)
