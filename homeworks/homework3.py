import os
import os.path
from pathlib import Path
import datetime

import cv2
import numpy as np
from omegaconf import OmegaConf


def bilateral_filter(
        src_image,
        sigma_space,
        sigma_color
):
    dst_image = np.zeros_like(src_image)
    h,w = src_image.shape
    d = int(2 * sigma_space + 1)
    r = d // 2
    x,y = np.mgrid[-r:r+1, -r:r+1]
    constant_W = np.exp(-(x**2 + y**2)/(2*sigma_space**2))
    
    padded_image = np.pad(src_image, ((r,r), (r,r)), 'reflect')
    
    for i in range(h):
        for j in range(w):
            padded_i = i + r
            padded_j = j + r
            neighbors = padded_image[padded_i - r: padded_i + r + 1, padded_j - r: padded_j + r + 1]

            mutual_W = np.exp(-((neighbors - src_image[i][j])**2 / (2 * sigma_color**2)))

            W = constant_W * mutual_W
            normalizaton = np.sum(W)

            weighted_neighbors = W * neighbors
            dst_image[i,j] = np.sum(weighted_neighbors) / normalizaton
    
    dst_image = np.uint8(np.clip(dst_image, 0, 255))
    assert src_image.shape == dst_image.shape
    return dst_image


def main(cfg):
    src_image = cv2.imread(os.path.join(os.path.dirname(__file__), "..", cfg['input_image_path']), cv2.IMREAD_GRAYSCALE)
    if src_image is None:
        print("Error: incorrect input image path")
        return
    sigma_space = cfg['sigma_space']
    sigma_color = cfg['sigma_color']

    before_opencv = datetime.datetime.now()
    opencv_dst_image = cv2.bilateralFilter(src=src_image,sigmaSpace=sigma_space, sigmaColor=sigma_color, d = int(2 * sigma_space + 1))
    cv2.imwrite(cfg['opencv_output_image_path'], opencv_dst_image)
    after_opencv = datetime.datetime.now()
    print(f"opencv bilateral filtration duration {after_opencv - before_opencv}")

    before_numpy = datetime.datetime.now()
    numpy_dst_image = bilateral_filter(src_image=src_image, sigma_space=sigma_space, sigma_color=sigma_color)
    cv2.imwrite(cfg['output_image_path'], numpy_dst_image)
    after_numpy = datetime.datetime.now()
    print(f"numpy bilateral filtration duration {after_numpy - before_numpy}")


if __name__ == "__main__":
    task = Path(__file__).stem
    cfg = OmegaConf.load("params.yaml")[task]

    main(cfg)
