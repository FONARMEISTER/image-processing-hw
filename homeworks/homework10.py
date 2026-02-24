from dataclasses import dataclass

import cv2
import numpy as np
import rootutils

root = rootutils.setup_root(".", indicator="homeworks", pythonpath=True)

DATA_DIR = root / "data"


@dataclass
class Config:
    video_file = DATA_DIR / "book.mp4"
    image_file = DATA_DIR / "book.jpg"

    detector = cv2.ORB_create(nfeatures=1000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # коэффициенты дисторсии (k1, k2, p1, p2, k3)
    dist_coeff = np.zeros(5)

    # внутренние параметры камеры
    K = np.array([
        [1000, 0, 320],
        [0, 1000,  240],
        [0, 0, 1.0]
    ])

    # минимальное число соответствующих точек в PnP алгоритме
    min_pnp_num = 100
    
    box_lower = np.array([
        [30, 145, 0], 
        [30, 200, 0], 
        [200, 200, 0], 
        [200, 145, 0]
    ], dtype=np.float32)

    box_upper = np.array([
        [30, 145, -50], 
        [30, 200, -50], 
        [200, 200, -50], 
        [200, 145, -50]
    ], dtype=np.float32)


def main(cfg):
    # Загружаем изображение книги (reference image)
    ref_image = cv2.imread(str(cfg.image_file))
    if ref_image is None:
        print(f"Error: Could not load image {cfg.image_file}")
        return
    
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    
    # Детектируем ключевые точки и дескрипторы на reference изображении
    ref_kp, ref_desc = cfg.detector.detectAndCompute(ref_gray, None)
    
    if ref_desc is None or len(ref_kp) == 0:
        print("Error: No keypoints detected in reference image")
        return
    
    print(f"Reference image: {len(ref_kp)} keypoints detected")
    
    # Получаем размеры reference изображения для создания 3D точек
    ref_height, ref_width = ref_gray.shape
    
    # Создаем 3D координаты для reference изображения (плоскость Z=0)
    # Координаты соответствуют пикселям изображения
    object_points_full = np.array([
        [kp.pt[0], kp.pt[1], 0] 
        for kp in ref_kp
    ], dtype=np.float32)
    
    # Открываем видео
    cap = cv2.VideoCapture(str(cfg.video_file))
    
    if not cap.isOpened():
        print(f"Error: Could not open video {cfg.video_file}")
        return
    
    frame_count = 0
    
    while True:
        ok, frame = cap.read()
        
        if not ok:
            break
        
        frame_count += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Детектируем ключевые точки и дескрипторы на текущем кадре
        frame_kp, frame_desc = cfg.detector.detectAndCompute(frame_gray, None)
        
        if frame_desc is None or len(frame_kp) == 0:
            cv2.imshow("Book PnP", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            continue
        
        # Матчинг с использованием Cross Check Matching
        matches = cfg.matcher.match(ref_desc, frame_desc)
        
        if len(matches) < cfg.min_pnp_num:
            cv2.imshow("Book PnP", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            continue
        
        # Извлекаем координаты совпавших точек
        ref_pts = np.float32([ref_kp[m.queryIdx].pt for m in matches])
        frame_pts = np.float32([frame_kp[m.trainIdx].pt for m in matches])
        
        # Фильтруем outliers с помощью RANSAC для гомографии
        H, mask_homography = cv2.findHomography(
            ref_pts, 
            frame_pts, 
            cv2.RANSAC, 
            ransacReprojThreshold=5.0
        )
        
        if H is None:
            cv2.imshow("Book PnP", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            continue
        
        # Применяем маску от гомографии
        mask_homography = mask_homography.ravel().astype(bool)
        filtered_matches = [m for m, mask in zip(matches, mask_homography) if mask]
        
        if len(filtered_matches) < cfg.min_pnp_num:
            cv2.imshow("Book PnP", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            continue
        
        # Подготавливаем данные для PnP
        object_pts = np.float32([object_points_full[m.queryIdx] for m in filtered_matches])
        image_pts = np.float32([frame_kp[m.trainIdx].pt for m in filtered_matches])
        
        # Решаем PnP задачу с RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_pts,
            image_pts,
            cfg.K,
            cfg.dist_coeff,
            reprojectionError=8.0,
            confidence=0.99
        )
        
        if not success or inliers is None or len(inliers) < 10:
            cv2.imshow("Book PnP", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            continue
        
        # Проецируем 3D box на изображение
        image_box_lower, _ = cv2.projectPoints(
            cfg.box_lower,
            rvec, tvec,
            cfg.K, cfg.dist_coeff
        )
        
        image_box_upper, _ = cv2.projectPoints(
            cfg.box_upper,
            rvec, tvec,
            cfg.K, cfg.dist_coeff
        )
        
        # Рисуем результат
        show_image = frame.copy()
        
        # Рисуем нижнюю грань (синий)
        cv2.polylines(
            show_image,
            [np.int32(image_box_lower)],
            True,
            (255, 0, 0),
            3
        )
        
        # Рисуем верхнюю грань (красный)
        cv2.polylines(
            show_image,
            [np.int32(image_box_upper)],
            True,
            (0, 0, 255),
            3
        )
        
        # Рисуем вертикальные ребра (зеленый)
        pts_lower = image_box_lower[:, 0].astype(int)
        pts_upper = image_box_upper[:, 0].astype(int)
        
        for (x1, y1), (x2, y2) in zip(pts_lower, pts_upper):
            cv2.line(
                show_image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                3
            )
        
        # Добавляем информацию о количестве inliers
        cv2.putText(
            show_image,
            f"Inliers: {len(inliers)}",
            (15, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        cv2.imshow("Book PnP", show_image)
        
        key = cv2.waitKey(10)
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    cfg = Config()
    main(cfg)