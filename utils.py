import numpy as np

def sigma_measure(l):
    '''
        Non-linear sigma measure to model distance variant sensor noise
    '''
    return 0.0001 + l**2 * 0.01

def kalman_predict(x_old, P_old, Q):
    '''
        Apply the kalman filter predict with an unknown state transitions (no velocities known from moving targets)
    '''
    x_est = x_old
    P_est = P_old + Q
    return x_est, P_est

def kalman_update(x_est, z, P_est, R):
    '''
        Apply the kalman filter update with noisy observation model that returns the position from every target: 
        z = I x + w -> H = I
    '''
    H = np.eye(len(x_est))
    y = z - x_est

    K = P_est @ np.linalg.inv(R + H @ P_est @ H.T)
    x_new = x_est + K @ y
    tmp = K @ H
    P_new = (np.eye(tmp.shape[0]) - tmp) @ P_est
    
    return x_new, P_new