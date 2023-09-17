import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time

def vector_normalize(vec):
    vec_norms = np.maximum(np.linalg.norm(vec, axis=1), 1e-6)
    vec = vec / vec_norms[:, np.newaxis]
    return vec

def getQuaternionFromEuler(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

def RotationwithQuaternion(q, p):
    x,y,z,w = q
    R_q = np.array([[w,z,-y,x],[-z,w,x,y],[y,-x,w,z],[-x,-y,-z,w]])
    L_q = np.array([[w,z,-y,-x],[-z,w,x,-y],[y,-x,w,-z],[x,y,z,w]])
    return np.matmul(p, np.matmul(R_q, L_q))

def MPC(self_state, goal_state, obstacles, psi_prev=0, k_q=20, q_coeff_max=10, init_Qv=0.2):
    opti = ca.Opti()
    ## parameters for optimization
    T = 0.2
    N = 10  # MPC horizon
    v_max = 0.5
    omega_max = 0.6
    safe_distance = 0.55
    Q = np.array([[1.2, 0.0, 0.0, 0.0],[0.0, 1.2, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, init_Qv]])
    R = np.array([[0.2, 0.0], [0.0, 0.15]])

    goal = goal_state[:,:3]

    tra = goal_state[:,3]
    conf = goal_state[:,4]
    e_z = goal_state[:-1,5:]

    tra_mean = np.mean(tra)
    conf_mean = np.mean(conf)
    # lamb = np.power((1 - tra_mean)*(1 - conf_mean), -2)
    lamb = np.power((1 - tra_mean), -2)

    opt_x0 = opti.parameter(4)
    opt_controls = opti.variable(N, 2)
    v = opt_controls[:, 0]
    omega = opt_controls[:, 1]

    ## dynamic parameters
    ps = np.diff(goal, axis=0)
    ps = vector_normalize(ps)
    ks = np.sum(ps * e_z, axis=-1, keepdims=True)
    e_x = ps - ks * e_z
    e_x = vector_normalize(e_x)
    e_y = np.cross(e_z, e_x)
    cur_e_x = e_x[0]
    cur_e_z = e_z[0]
    cur_e_y = e_y[0]

    cur_yaw, cur_roll, cur_pitch = self_state[0, 2], self_state[0, 4], self_state[0, 5]
    quat = getQuaternionFromEuler(cur_roll, cur_pitch, cur_yaw)
    cur_gear = 1.0 if self_state[0,3] >= 0 else -1.0
    cur_direction = RotationwithQuaternion(quat, np.array([1,0,0,1]))[:3] * cur_gear

    cur_q = np.dot(cur_direction, cur_e_x) * cur_e_x + np.dot(cur_direction, cur_e_z) * cur_e_z
    cur_p = ps[0]
    sin_psi = np.dot(np.cross(cur_q, cur_p), cur_e_y)
    psi = np.arcsin(sin_psi) if not np.isnan(sin_psi) else psi_prev
    q_coeff = min(k_q * max(0, np.tan(psi)) + 1, q_coeff_max)
    Q[3, 3] = q_coeff * init_Qv

    ## state variables
    target_velocities = np.ones(shape=(len(goal_state), 1)) * v_max
    goal = np.hstack([goal, target_velocities])

    opt_states = opti.variable(N+1, 4)
    x = opt_states[:, 0]
    y = opt_states[:, 1]
    theta = opt_states[:, 2]
    vel = opt_states[:, 3]

    ## create funciont for F(x)
    theta_x = self_state[0][5]*np.cos(self_state[0][2]) - self_state[0][4]*np.sin(self_state[0][2])
    theta_y = self_state[0][5]*np.sin(self_state[0][2]) + self_state[0][4]*np.cos(self_state[0][2])
    f = lambda x_, u_: ca.vertcat(*[u_[0]*ca.cos(x_[2])*np.cos(theta_x), u_[0]*ca.sin(x_[2])*np.cos(theta_y), u_[1], u_[0]*ca.cos(x_[2])*np.cos(theta_x)])

    ## init_condition
    opti.subject_to(opt_states[0, :] == opt_x0.T)

    # Position Boundaries
    # Here you can customize the avoidance of local obstacles

    # Admissable Control constraints
    opti.subject_to(opti.bounded(-v_max, v, v_max))
    opti.subject_to(opti.bounded(-omega_max, omega, omega_max))

    # System Model constraints
    for i in range(N):
        x_next = opt_states[i, :] + T*f(opt_states[i, :], opt_controls[i, :]).T
        opti.subject_to(opt_states[i+1, :]==x_next)

    #### cost function
    obj = 0
    for i in range(N):
        obj = obj + 0.1*ca.mtimes([(opt_states[i, :] - goal[[i]]), Q, (opt_states[i, :]- goal[[i]]).T]) + lamb * ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T])
    obj = obj + 2*ca.mtimes([(opt_states[N-1, :] - goal[[N-1]]), Q, (opt_states[N-1, :]- goal[[N-1]]).T])

    opti.minimize(obj)
    opts_setting = {'ipopt.max_iter':80, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-3, 'ipopt.acceptable_obj_change_tol':1e-3}
    opti.solver('ipopt',opts_setting)
    opti.set_value(opt_x0, self_state[:,:4])

    try:
        sol = opti.solve()
        u_res = sol.value(opt_controls)
        state_res = sol.value(opt_states)

        print("\tpsi : %.2f, q_coeff : %.2f, cur_vel : %.2f\n" %(psi, q_coeff, self_state[0, 3]))
    except:
        state_res = np.repeat(self_state[:4],N+1,axis=0)
        u_res = np.zeros([N,2])

    return state_res, u_res
