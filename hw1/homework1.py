from pathlib import Path

import cv2, os, copy
import os.path
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf


def show_frame_and_transformed_frame(frame, transformed_frame, cfg):
    src_pts = get_src_pts(cfg)
    dst_pts = get_dst_pts(cfg)
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
    ax1.imshow(frame)
    ax1.plot(src_pts[:, 0], src_pts[:, 1], 'o', color='red')
    ax2.imshow(transformed_frame)
    ax2.plot(dst_pts[:, 0], dst_pts[:, 1], 'o', color='red')


def check_cfg(cfg):
    colorjitter = cfg['colorjitter']
    hue = colorjitter['hue']
    saturation = colorjitter['saturation']
    contrast = colorjitter['contrast']
    brightness = colorjitter['brightness']

    if hue < 0 or hue > 360:
        return False
    if saturation < 0 or saturation > 1:
        return False
    if contrast < 0 or contrast > 1:
        return False
    if brightness < 0 or brightness > 1:
        return False
    return True


def apply_colorjitter(img_, hue, saturation, contrast, brightness):
    img = copy.copy(img_)

    # hue
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    img_hsv[:,:,1] = (img_hsv[:,:,1] + hue) % 180
    img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # saturation
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img[:,:,0] = np.clip(img[:,:,0] * saturation + img_grayscale * (1 - saturation), 0, 255)
    img[:,:,1] = np.clip(img[:,:,1] * saturation + img_grayscale * (1 - saturation), 0, 255)
    img[:,:,2] = np.clip(img[:,:,2] * saturation + img_grayscale * (1 - saturation), 0, 255)

    # contrast
    mean = cv2.mean(img)[0:3]
    mean = np.mean(mean)
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=(1 - contrast) * mean)

    # brightness
    img = cv2.convertScaleAbs(img, alpha=brightness)
    
    return img


def get_dst_pts(cfg):
    dst_h = cfg['dst_size']['h']
    dst_w = cfg['dst_size']['w']
    return np.float32([
        [0, 0],
        [dst_w, 0],
        [dst_w, dst_h],
        [0, dst_h]
    ])


def get_src_pts(cfg):
    cfg_src_points = cfg['src_points']
    return np.float32([[cfg_src_points['point1']['x'], cfg_src_points['point1']['y']],
                      [cfg_src_points['point2']['x'], cfg_src_points['point2']['y']],
                      [cfg_src_points['point3']['x'], cfg_src_points['point3']['y']],
                      [cfg_src_points['point4']['x'], cfg_src_points['point4']['y']]]) 


def process_frame(src_img, cfg):
    src_pts = get_src_pts(cfg)

    dst_h = cfg['dst_size']['h']
    dst_w = cfg['dst_size']['w']
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    dst_pts = get_dst_pts(cfg)

    H = cv2.getPerspectiveTransform(src_pts, dst_pts) 
    dst_img = cv2.warpPerspective(src_img, H, (dst_w, dst_h), cv2.INTER_LINEAR)

    colorjitter = cfg['colorjitter']
    hue = colorjitter['hue']
    saturation = colorjitter['saturation']
    contrast = colorjitter['contrast']
    brightness = colorjitter['brightness']
    dst_img_colorjitter = apply_colorjitter(dst_img, hue, saturation, contrast, brightness)
    return dst_img_colorjitter


def main(cfg):
    if not check_cfg(cfg):
        print('Error: Invalid colorjitter configuration')
        return
    
    vid = cv2.VideoCapture(os.path.join(os.path.dirname(__file__),'cat.mp4'))
    if not vid.isOpened():
        print('Error: Could not open video file.')
    else:
        print('Video file opened successfully')
    fps = vid.get(cv2.CAP_PROP_FPS)
    window_name = 'Video Player'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = vid.read()
        if not ret:
            print('Read all video')
            break
        transformed_frame = process_frame(frame, cfg)
        cv2.imshow(window_name, transformed_frame)
        key = cv2.waitKey(int(1000 // fps))
        if key == ord('q') or key == 27:
            print('Video interrupted by user')
            break


if __name__ == '__main__':
    task = Path(__file__).stem
    cfg = OmegaConf.load('params.yaml')[task]
    

    main(cfg)


    
