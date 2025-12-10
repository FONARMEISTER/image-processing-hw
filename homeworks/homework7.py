import numpy as np

def transform_points(points, R, t):
  points2 = (points - t) @ R
  return points2


def rotation_matrix_from_rotvec(omega):
  theta = np.linalg.norm(omega)
  n = omega / theta
  skew = np.array([ [0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0] ])
  R = np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * np.dot(skew, skew)
  return R

def rotvec_from_rotation_matrix(R):
  theta = np.arccos((np.trace(R) - 1) / 2)
  n = (1 / (2 * np.sin(theta))) * np.array([R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]])
  return theta * n


def project_points(points, P):
  points2 = np.column_stack((points, np.ones((points.shape[0], 1)))) @ P.T
  points2[:, :2] /= points2[:, 2:3]
  return points2[:, :2]

def from_image_coordinates_to_world(x, y, c, P):
  P_inv = np.linalg.pinv(P)
  point = np.dot(P_inv, np.hstack([x, y, 1])) + np.hstack([c, 1])
  point3d =  point[:3] / point[3]

  ray_dir = point3d - c
  ray_dir /= np.linalg.norm(ray_dir)

  t = -c[1] / ray_dir[1]

  res_point = c + t * ray_dir
  return res_point