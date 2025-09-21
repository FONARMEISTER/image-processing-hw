from pathlib import Path

import cv2
import copy
import numpy as np
import os, os.path
import matplotlib.pyplot as plt
from omegaconf import OmegaConf


class RingsFilter:
    
    def __init__(self, path):
        
        self.init_img = cv2.imread(os.path.join(os.path.dirname(__file__), path), cv2.IMREAD_GRAYSCALE)        
        
        if self.init_img is None:
            print("Error: Could not open image file.")
            raise ValueError("Could not open image file")
        else:
            print("Image file opened successfully")

        self.init_specter = self.create_specter(self.init_img) 
        self.h, self.w = self.init_specter.shape

        self.img_window = cv2.namedWindow('Rings')
        self.specter_window = cv2.namedWindow('Specter')
        cv2.createTrackbar('h', 'Specter', 0, self.h // 2, self.on_h_change)
        cv2.createTrackbar('w', 'Specter', 0, self.w // 2, self.on_w_change)
        self.removed_h = 0
        self.removed_w = 0

        self.update_state()

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
    def create_specter(self, img):
        return np.abs(np.fft.fftshift(np.fft.fft2(self.init_img)))


    def on_h_change(self, value):
        self.removed_h = value
        self.update_state()


    def on_w_change(self, value):
        self.removed_w = value
        self.update_state()

    def update_specter(self):
        self.specter = copy.copy(self.init_specter)
        if self.removed_h > 0 and self.removed_w > 0:
          self.specter[0:self.removed_h, self.w // 2 - self.removed_w : self.w // 2 + self.removed_w] = 0
          self.specter[self.h - self.removed_h: self.h, self.w // 2 - self.removed_w : self.w // 2 + self.removed_w] = 0
        
    def draw_specter(self):
        normalized_specter = cv2.normalize(np.log(1 + self.specter), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imshow('Specter', normalized_specter)

    def draw_image(self):
        img = np.real(np.fft.ifft2(np.fft.ifftshift(self.specter)))
        img = cv2.normalize(np.log(1 + img), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imshow('Rings', img)

    def update_state(self):
        self.update_specter()
        self.draw_specter()
        self.draw_image()


def main(cfg):
    img_path = 'saturn_rings.png'
    RingsFilter(img_path)


if __name__ == "__main__":
    task = Path(__file__).stem
    cfg = OmegaConf.load("params.yaml")[task]

    main(cfg)
