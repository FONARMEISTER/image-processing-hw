import os
import os.path
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf


class VideoProcessor:
    def __init__(self, path, cfg):
        self.cfg = cfg
        self.vid = cv2.VideoCapture(os.path.join(os.path.dirname(__file__), "cat.mp4"))
        if not self.vid.isOpened():
            print("Error: Could not open video file.")
            raise ValueError("Could not open video file")
        else:
            print("Video file opened successfully")

        self.fps = self.vid.get(cv2.CAP_PROP_FPS)
        self.points = []
        self.max_points = 4
        self.frame = None

        self.brightness = 0
        self.contrast = 0
        self.saturation = 0
        self.hue = 0

        self.original_window = cv2.namedWindow("Original")
        self.transformed_window = cv2.namedWindow("Transformed")
        cv2.createTrackbar("Brightness", "Original", 0, 100, self.on_brightness_change)
        cv2.createTrackbar("Contrast", "Original", 0, 100, self.on_contrast_change)
        cv2.createTrackbar("Saturation", "Original", 0, 100, self.on_saturation_change)
        cv2.createTrackbar("Hue", "Original", 0, 180, self.on_hue_change)

        cv2.setMouseCallback("Original", self.mouse_callback)

        self.transformed_w = 0
        self.transformed_h = 0

    def draw_frame_with_points(self):
        frame_with_points = self.frame.copy()

        for i, point in enumerate(self.points):
            cv2.circle(frame_with_points, (point[0], point[1]), 5, (0, 0, 255), -1)
            cv2.putText(
                frame_with_points,
                str(i + 1),
                (point[0] + 10, point[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Original", frame_with_points)

    def reset_points(self):
        self.points = []

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(self.points) < self.max_points:
            self.points.append([x, y])
            self.draw_frame_with_points()
        if len(self.points) == self.max_points:
            self.transformed_w = self.points[2][0]
            self.transformed_h = self.points[2][1]

    def on_brightness_change(self, value):
        self.brightness = value / 100.0

    def on_contrast_change(self, value):
        self.contrast = value / 100.0

    def on_saturation_change(self, value):
        self.saturation = value / 100.0

    def on_hue_change(self, value):
        self.hue = value

    def apply_colorjitter(self, frame):
        # hue
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
        frame_hsv[:, :, 1] = (frame_hsv[:, :, 1] + self.hue) % 180
        frame = cv2.cvtColor(frame_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # saturation
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame[:, :, 0] = np.clip(
            frame[:, :, 0] * self.saturation + frame_grayscale * (1 - self.saturation),
            0,
            255,
        )
        frame[:, :, 1] = np.clip(
            frame[:, :, 1] * self.saturation + frame_grayscale * (1 - self.saturation),
            0,
            255,
        )
        frame[:, :, 2] = np.clip(
            frame[:, :, 2] * self.saturation + frame_grayscale * (1 - self.saturation),
            0,
            255,
        )

        # contrast
        mean = cv2.mean(frame)[0:3]
        mean = np.mean(mean)
        frame = cv2.convertScaleAbs(
            frame, alpha=self.contrast, beta=(1 - self.contrast) * mean
        )

        # brightness
        frame = cv2.convertScaleAbs(frame, alpha=self.brightness)

        return frame

    def transform_frame(self, frame):
        src_pts = np.array(self.points, dtype=np.float32)

        dst_w = self.transformed_w
        dst_h = self.transformed_h
        tranformed_frame = np.zeros((dst_h, dst_w, 3), dtype=np.float32)
        dst_pts = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], np.float32)

        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        tranformed_frame = cv2.warpPerspective(
            frame, H, (dst_w, dst_h), cv2.INTER_LINEAR
        )
        return self.apply_colorjitter(tranformed_frame)

    def process(self):
        while True:
            ret, self.frame = self.vid.read()
            if not ret:
                print("Read all video")
                break
            self.draw_frame_with_points()
            if len(self.points) == self.max_points:
                transformed_frame = self.transform_frame(self.frame)
                cv2.imshow("Transformed", transformed_frame)
            key = cv2.waitKey(int(10000 // self.fps))

            if key == ord("q") or key == 27:
                print("Video interrupted by user")
                break
            elif key == ord("r") or key == ord("R"):
                self.reset_points()

        self.vid.release()
        cv2.destroyAllWindows()


def main(cfg):
    processor = VideoProcessor("cat.mp4", cfg)
    processor.process()


if __name__ == "__main__":
    task = Path(__file__).stem
    cfg = OmegaConf.load("params.yaml")[task]

    main(cfg)
