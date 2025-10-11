import os
import os.path
import random
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf


class Ransac:
    def __init__(self, thresh, max_trials, seed):
        self.thresh = thresh
        self.max_trials = max_trials
        self.rnd = random.Random()
        self.rnd.seed(seed)

        self.A = None
        self.B = None
        self.C = None
        self.inliners_mask = None
        self.inliners_count = 0

    def ransac_iter(self, X, y):
        n = len(X)
        p1, p2 = np.random.choice(n, 2, replace=False)
        A = y[p1] - y[p2]
        B = X[p2] - X[p1]
        C = X[p1] * y[p2] - X[p2]* y[p1]

        inliners_mask = np.zeros_like(y, dtype=bool)
        inliners_count = 0
        for i in range(n):
            dist = abs(A * X[i] + B * y[i] + C) / (A * A + B * B)**0.5
            if dist <= self.thresh:
                inliners_mask[i] = True
                inliners_count += 1
                
        return (A, B, C, inliners_mask, inliners_count)

    def fit(self, X, y):
        for i in range(self.max_trials):
            A, B, C, inliners_mask, inliners_count = self.ransac_iter(X, y)
            if inliners_count > self.inliners_count:
                self.A = A
                self.B = B
                self.C = C
                self.inliners_mask = inliners_mask
                self.inliners_count = inliners_count


def apply_gaussian_filter(img, sigma):
    n = 2 * np.ceil(3 * sigma).astype(int) + 1
    u = cv2.getGaussianKernel(n, sigma=0)
    G = u @ u.T
    return cv2.filter2D(np.float32(img), ddepth=-1, kernel=G)


def apply_laplas(img):
    L = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    return cv2.filter2D(img, ddepth=-1, kernel=L)


def calc_marr_hildreth_edges(img, thresh_percent):
    T = np.max(img) * thresh_percent

    h, w = img.shape
    pad_image = np.pad(img, ((2, 2), (2, 2)), "constant")
    edges = np.zeros_like(img, dtype=np.uint8)
    for i in range(1, h + 1):
        for j in range(1, w + 1):
            window = pad_image[i - 1 : i + 2, j - 1 : j + 2]
            (a, b, c, d, _, e, f, g, h) = window.flatten()

            ok = False

            if d * e < 0 and np.abs(d - e) > T:
                ok = True
            elif b * g < 0 and np.abs(b - g) > T:
                ok = True
            elif a * h < 0 and np.abs(a - h) > T:
                ok = True
            elif f * c < 0 and np.abs(f - c) > T:
                ok = True

            if ok:
                edges[i - 1, j - 1] = 255
    return edges


def find_line_with_ransac(edges, thresh, max_trials, seed):
    ransac = Ransac(
        max_trials=max_trials,
        thresh=thresh,
        seed=seed,
    )
    cols, rows = np.where(edges != 0)
    edges_points = np.column_stack((rows, cols))
    X = edges_points[:, 0]
    y = edges_points[:, 1]
    ransac.fit(X, y)

    ransac_on_inliners = Ransac(
        max_trials=max_trials,
        thresh=thresh,
        seed=seed
    )
    X_inliners = X[ransac.inliners_mask]
    y_inliners = y[ransac.inliners_mask]
    ransac_on_inliners.fit(X_inliners, y_inliners)

    edges_inliners_mask = np.zeros_like(edges)
    x_min_ind = 0
    x_max_ind = 0
    for i, is_inlier in enumerate(ransac_on_inliners.inliners_mask):
        if not is_inlier:
            continue
        edges_inliners_mask[y_inliners[i], X_inliners[i]] = 1
        if X_inliners[i] > X_inliners[x_max_ind]:
            x_max_ind = i
        if X_inliners[i] < X_inliners[x_min_ind]:
            x_min_ind = i

    A, B, C = ransac_on_inliners.A, ransac_on_inliners.B, ransac_on_inliners.C
    if B == 0: # vertical line
        return (X_inliners[x_min_ind], y_inliners[x_min_ind], X_inliners[x_max_ind], y_inliners[x_max_ind], edges_inliners_mask)
    x_min = X_inliners[x_min_ind]
    x_max = X_inliners[x_max_ind]
    y_min = -int((A * x_min + C) / B)
    y_max = -int((A * x_max + C) / B)
    return (x_min, y_min, x_max, y_max, edges_inliners_mask)


def main(cfg):
    input_image_name = cfg["input_image"]
    init_img = cv2.imread(
        os.path.join(os.path.dirname(__file__), "..", input_image_name),
        cv2.IMREAD_GRAYSCALE,
    )
    if init_img is None:
        print("Error: Could not open image file.")
        raise ValueError("Could not open image file")
    else:
        print("Image file opened successfully")

    sigma = cfg["sigma"]
    filtered_img = apply_laplas(apply_gaussian_filter(init_img, sigma))
    thresh_percent = cfg["thresh_percent"]
    edges = calc_marr_hildreth_edges(filtered_img, thresh_percent)

    seed = cfg["seed"]
    max_trials = cfg["max_trials"]
    ransac_thresh = cfg["threshold"]
    x_min, y_min, x_max, y_max, edges_inliners_mask = find_line_with_ransac(
        edges, ransac_thresh, max_trials, seed
    )

    ransac_image = cv2.cvtColor(init_img, cv2.COLOR_GRAY2BGR)
    cv2.line(ransac_image, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)
    output_image_name = cfg["output_image"]
    cv2.imwrite(output_image_name, ransac_image)

    output_edges_name = cfg["output_edges"]
    h, w = edges.shape
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    for i in range(h):
        for j in range(w):
            if edges_inliners_mask[i, j] == 1:
                edges[i, j] = (255, 0, 255)
    cv2.imwrite(output_edges_name, edges)


if __name__ == "__main__":
    task = Path(__file__).stem
    cfg = OmegaConf.load("params.yaml")[task]

    main(cfg)
