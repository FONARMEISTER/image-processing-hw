import copy
import os
import os.path
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf


class RingsFilter:
    def __init__(self, cfg):
        self.image_name = cfg["image_name"]
        self.img_window = cv2.namedWindow("Rings")
        self.spectrum_window = cv2.namedWindow("Spectrum")
        self.w = 0
        self.h = 0
        self.removed_h = 0
        self.removed_w = 0
        self.spectrum = None
        self.abs_spectrum = None
        self.init_img = None
        self.init_abs_spectrum = None
        self.init_arg_spectrum = None

    def create_spectrum(self, img):
        return np.fft.fftshift(np.fft.fft2(img))

    def on_h_change(self, value):
        self.removed_h = value
        self.update_state()

    def on_w_change(self, value):
        self.removed_w = value
        self.update_state()

    def update_spectrum(self):
        self.abs_spectrum = copy.copy(self.init_abs_spectrum)
        if self.removed_h > 0 and self.removed_w > 0:
            self.abs_spectrum[
                0 : self.removed_h,
                self.w // 2 - self.removed_w : self.w // 2 + self.removed_w,
            ] = 0
            self.abs_spectrum[
                self.h - self.removed_h : self.h,
                self.w // 2 - self.removed_w : self.w // 2 + self.removed_w,
            ] = 0
        self.spectrum = self.abs_spectrum * np.exp(1j * self.init_arg_spectrum)

    def draw_spectrum(self):
        normalized_spectrum = cv2.normalize(
            np.log(1 + np.abs(self.spectrum)), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )
        cv2.imshow("Spectrum", normalized_spectrum)

    def draw_image(self):
        img = np.abs(np.fft.ifft2(np.fft.ifftshift(self.spectrum)))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imshow("Rings", img)

    def update_state(self):
        self.update_spectrum()
        self.draw_spectrum()
        self.draw_image()

    def process(self):
        self.init_img = cv2.imread(
            os.path.join(os.path.dirname(__file__), "..", "data", self.image_name),
            cv2.IMREAD_GRAYSCALE,
        )
        if self.init_img is None:
            print("Error: Could not open image file.")
            raise ValueError("Could not open image file")
        else:
            print("Image file opened successfully")

        init_spectrum = self.create_spectrum(self.init_img)
        self.init_abs_spectrum = np.abs(init_spectrum)
        self.init_arg_spectrum = np.angle(init_spectrum)
        self.h, self.w = self.init_abs_spectrum.shape
        cv2.createTrackbar("h", "Spectrum", 0, self.h // 2, self.on_h_change)
        cv2.createTrackbar("w", "Spectrum", 0, self.w // 2, self.on_w_change)
        self.update_state()

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main(cfg):
    rings_filter = RingsFilter(cfg)
    rings_filter.process()


if __name__ == "__main__":
    task = Path(__file__).stem
    cfg = OmegaConf.load("params.yaml")[task]

    main(cfg)
