import numpy as np
from scipy.spatial.transform import Rotation


def solvePnP_DLT(points3D, points2D, K):
    N = len(points3D)
    
    K_inv = np.linalg.inv(K)
    points2D_hom = np.hstack([points2D, np.ones((N, 1))])
    points2D_norm = (K_inv @ points2D_hom.T).T
    x_prime = points2D_norm[:, 0]
    y_prime = points2D_norm[:, 1]

    A = np.zeros((2 * N, 12))
    for i in range(N):
        u, v, w = points3D[i]
        xp, yp = x_prime[i], y_prime[i]
        A[2*i, 0:4] = [u, v, w, 1]
        A[2*i, 4:8] = [0, 0, 0, 0]
        A[2*i, 8:12] = [-xp*u, -xp*v, -xp*w, -xp]
        A[2*i+1, 0:4] = [0, 0, 0, 0]
        A[2*i+1, 4:8] = [u, v, w, 1]
        A[2*i+1, 8:12] = [-yp*u, -yp*v, -yp*w, -yp]

    U, L, Vt = np.linalg.svd(A)
    theta = Vt[-1, :]
    
    R_raw = np.array([
        [theta[0], theta[1], theta[2]],
        [theta[4], theta[5], theta[6]],
        [theta[8], theta[9], theta[10]]
    ])
    t_raw = np.array([theta[3], theta[7], theta[11]])
    
    U_r, L_r, Vt_r = np.linalg.svd(R_raw)
    R = U_r @ Vt_r
    t = t_raw / L_r[0]

    if np.linalg.det(R) < 0:
        R = -R
        t = -t
    omega = Rotation.from_matrix(R).as_rotvec()
    
    return omega, t


def triangulate_DLT(rotation_vectors, translations, camera_points2D, K):
    J = len(rotation_vectors)
    N = len(camera_points2D[0])
    
    K_inv = np.linalg.inv(K)
    points3D = np.zeros((N, 3))
    
    for point_idx in range(N):
        A = np.zeros((2 * J, 3))
        b = np.zeros(2 * J)
        
        for cam_idx in range(J):
            omega = rotation_vectors[cam_idx]
            t = translations[cam_idx]
            R = Rotation.from_rotvec(omega).as_matrix()
            
            x, y = camera_points2D[cam_idx][point_idx]
            point2D_hom = np.array([x, y, 1.0])
            point2D_norm = K_inv @ point2D_hom
            x_prime, y_prime = point2D_norm[0], point2D_norm[1]
            
            r11, r12, r13 = R[0, :]
            r21, r22, r23 = R[1, :]
            r31, r32, r33 = R[2, :]
            t1, t2, t3 = t
            
            A[2*cam_idx, 0] = r31 * x_prime - r11
            A[2*cam_idx, 1] = r32 * x_prime - r12
            A[2*cam_idx, 2] = r33 * x_prime - r13
            b[2*cam_idx] = t1 - t3 * x_prime
            
            A[2*cam_idx+1, 0] = r31 * y_prime - r21
            A[2*cam_idx+1, 1] = r32 * y_prime - r22
            A[2*cam_idx+1, 2] = r33 * y_prime - r23
            b[2*cam_idx+1] = t2 - t3 * y_prime
        
        w = np.linalg.lstsq(A, b, rcond=None)[0]
        points3D[point_idx] = w
    
    return points3D