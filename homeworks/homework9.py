import numpy as np
import cv2


def compute_homography_dlt(points1, points2):
    N = len(points1)
    
    Omega = np.zeros((2 * N, 9))
    
    for i in range(N):
        x, y = points1[i]
        x_prime, y_prime = points2[i]
        
        Omega[2*i, 0:3] = [0, 0, 0]
        Omega[2*i, 3:6] = [-x, -y, -1]
        Omega[2*i, 6:9] = [y_prime * x, y_prime * y, y_prime]
        
        Omega[2*i+1, 0:3] = [x, y, 1]
        Omega[2*i+1, 3:6] = [0, 0, 0]
        Omega[2*i+1, 6:9] = [-x_prime * x, -x_prime * y, -x_prime]
    
    U, L, Vt = np.linalg.svd(Omega)
    theta = Vt[-1, :]
    
    H = theta.reshape(3, 3)
    H = H / H[2, 2]
    
    return H


def stabilize_video_with_homography(frames):
    num_frames = len(frames)
    height, width = frames[0].shape[:2]
    
    stabilized_frames = np.zeros_like(frames)
    stabilized_frames[0] = frames[0].copy()
    
    sift = cv2.SIFT_create(
        nfeatures = 1000,
        edgeThreshold = 6,
        contrastThreshold = 0.04,
        nOctaveLayers = 3,
        sigma = 1.6
    )
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    gray_ref = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    keypoints_ref, descriptors_ref = sift.detectAndCompute(gray_ref, None)
    
    for i in range(1, num_frames):
        gray_curr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        
        keypoints_curr, descriptors_curr = sift.detectAndCompute(gray_curr, None)
        
        if descriptors_ref is None or descriptors_curr is None:
            stabilized_frames[i] = frames[i].copy()
            continue
        
        matches = bf.match(descriptors_ref, descriptors_curr)
        
        if len(matches) < 4:
            stabilized_frames[i] = frames[i].copy()
            continue
        
        points_ref = np.float32([keypoints_ref[m.queryIdx].pt for m in matches])
        points_curr = np.float32([keypoints_curr[m.trainIdx].pt for m in matches])
        
        H, inlier_mask = cv2.findHomography(points_curr, points_ref, cv2.RANSAC)
        
        if H is None:
            stabilized_frames[i] = frames[i].copy()
            continue
        
        stabilized_frames[i] = cv2.warpPerspective(frames[i], H, (width, height))
    
    return stabilized_frames


def get_affine_transform(points1, points2):
    N = len(points1)
    
    Omega = np.zeros((2 * N, 6))
    beta = np.zeros(2 * N)
    
    for i in range(N):
        x, y = points1[i]
        x_prime, y_prime = points2[i]
        
        Omega[2*i, 0:3] = [x, y, 1]
        Omega[2*i, 3:6] = [0, 0, 0]
        beta[2*i] = x_prime
        
        Omega[2*i+1, 0:3] = [0, 0, 0]
        Omega[2*i+1, 3:6] = [x, y, 1]
        beta[2*i+1] = y_prime
    
    theta = np.linalg.lstsq(Omega, beta, rcond=None)[0]
    
    A = np.array([
        [theta[0], theta[1], theta[2]],
        [theta[3], theta[4], theta[5]]
    ])
    
    hat_points = np.zeros((N, 2))
    for i in range(N):
        x, y = points1[i]
        hat_points[i] = A @ np.array([x, y, 1])
    
    proj_error = np.linalg.norm(points2 - hat_points, axis=1).mean()
    
    return A, proj_error