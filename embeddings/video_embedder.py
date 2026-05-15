import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from gait_model import GaitLSTM
gait_model = GaitLSTM(input_dim=41, hidden_dim=128, emb_dim=128)
gait_model.eval()
def frame_embedding_enhanced(kps: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Эмбеддинг кадра:
      - 26 относительных координат,
      - 12 углов в суставах,
      - 3 нормированные длины.
    """
    hip_center = (kps[11][:2] + kps[12][:2]) / 2.0
    foot_center = (kps[15][:2] + kps[16][:2]) / 2.0
    nose = kps[0][:2]

    # 26 относительных координат (нос + точки тела 5–16)
    ids = [0] + list(range(5, 17))
    coords_26 = []
    for idx in ids:
        x_i, y_i, _ = kps[idx]
        dx = x_i - hip_center[0]
        dy = y_i - hip_center[1]
        coords_26.extend([dx, dy])

    height_approx = np.linalg.norm(nose - foot_center) + 1e-6
    coords_26 = np.array(coords_26, dtype=np.float32) / height_approx

    # 12 углов
    def angle_3pt(a, b, c):
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.arccos(np.clip(cosine, -1.0, 1.0))

    angle_keys = [
        "L_elbow", "R_elbow",
        "L_knee", "R_knee",
        "L_hip", "R_hip",
        "L_shoulder", "R_shoulder",
        "L_wrist", "R_wrist",
        "L_shin", "R_shin",
    ]

    angle_vals = np.array([
        # локти
        angle_3pt(kps[5][:2], kps[7][:2], kps[9][:2]),
        angle_3pt(kps[6][:2], kps[8][:2], kps[10][:2]),
        # колени
        angle_3pt(kps[11][:2], kps[13][:2], kps[15][:2]),
        angle_3pt(kps[12][:2], kps[14][:2], kps[16][:2]),
        # бёдра
        angle_3pt(hip_center, kps[11][:2], kps[13][:2]),
        angle_3pt(hip_center, kps[12][:2], kps[14][:2]),
        # плечи
        angle_3pt(kps[5][:2], kps[11][:2], hip_center),
        angle_3pt(kps[6][:2], kps[12][:2], hip_center),
        # запястья
        angle_3pt(kps[7][:2], kps[9][:2], kps[5][:2]),
        angle_3pt(kps[8][:2], kps[10][:2], kps[6][:2]),
        # голени
        angle_3pt(kps[13][:2], kps[15][:2], foot_center),
        angle_3pt(kps[14][:2], kps[16][:2], foot_center),
    ], dtype=np.float32)

    # 3 нормированные длины
    leg_len = np.linalg.norm(foot_center - hip_center) / height_approx
    torso_len = np.linalg.norm(nose - hip_center) / height_approx
    height_norm = 1.0

    lengths = np.array([leg_len, torso_len, height_norm], dtype=np.float32)

    emb = np.concatenate([coords_26, angle_vals, lengths])  # 26 + 12 + 3 = 41

    debug_info = {
        "angles": dict(zip(angle_keys, angle_vals)),
        "leg_len": float(leg_len),
        "torso_len": float(torso_len),
        "height_norm": float(height_norm),
    }
    return emb, debug_info

def build_video_embedding(model, video_path: str, visualize: bool = True) -> np.ndarray:
    """
    Визуализация всего процесса:
      - на каждом валидном кадре рисуем скелет,
      - вычисляем признаки только когда лицо в зоне 25–75% по ширине окна,
      - пробел: пауза/продолжить, ESC: выход,
      - эмбеддинг = среднее по всем кадрам из рабочей зоны.
    """
    WIN_W, WIN_H = 640, 480
    CONF_THR = 0.5

    EMBEDS = []  # сюда эмбеддинги только из центральной зоны

    cap = cv2.VideoCapture(video_path)
    paused = False

    SKELETON = [
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 6),
        (5, 11), (6, 12),
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
    ]

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            if len(results) == 0:
                key = cv2.waitKey(1) & 0xFF if visualize else -1
                if key == 27:
                    break
                if key == 32:
                    paused = not paused
                continue

            r = results[0]
            if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
                key = cv2.waitKey(1) & 0xFF if visualize else -1
                if key == 27:
                    break
                if key == 32:
                    paused = not paused
                continue

            box = r.boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)

            kps_all = r.keypoints.data.cpu().numpy()
            confs_all = r.keypoints.conf.cpu().numpy()

            kps = kps_all[0]
            conf = confs_all[0]

            needed_ids = [0] + list(range(5, 17))
            all_visible = True
            for idx in needed_ids:
                x, y, _ = kps[idx]
                c = conf[idx]
                if not (c > CONF_THR and x > 0 and y > 0):
                    all_visible = False
                    break

            if not all_visible:
                key = cv2.waitKey(1) & 0xFF if visualize else -1
                if key == 27:
                    break
                if key == 32:
                    paused = not paused
                continue

            person_roi = frame[y1:y2, x1:x2].copy()
            if person_roi.size == 0:
                key = cv2.waitKey(1) & 0xFF if visualize else -1
                if key == 27:
                    break
                if key == 32:
                    paused = not paused
                continue

            h_bbox, w_bbox = person_roi.shape[:2]
            scale = min(WIN_W / w_bbox, WIN_H / h_bbox)
            new_w = int(w_bbox * scale)
            new_h = int(h_bbox * scale)

            person_scaled = cv2.resize(person_roi, (new_w, new_h))

            offset_x = (WIN_W - new_w) // 2
            offset_y = (WIN_H - new_h) // 2

            kps_roi = kps.copy()
            kps_roi[:, 0] = (kps_roi[:, 0] - x1) * scale + offset_x
            kps_roi[:, 1] = (kps_roi[:, 1] - y1) * scale + offset_y

            # координата носа в окне [0, WIN_W]
            x_nose, y_nose, _ = kps_roi[0]
            x_norm = x_nose / WIN_W  # [0, 1]

            in_central_zone = (0.25 <= x_norm <= 0.75)

            if visualize:
                # окно 1: человек + скелет
                win1 = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
                win1[offset_y:offset_y + person_scaled.shape[0],
                     offset_x:offset_x + person_scaled.shape[1]] = person_scaled.copy()

                # окно 2: скелет + служебная инфа
                win2 = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)

                def draw_skeleton(img, color=(0, 255, 0)):
                    for i in range(5, 17):
                        x, y = kps_roi[i][0], kps_roi[i][1]
                        cv2.circle(img, (int(x), int(y)), 4, color, -1)
                    x_n, y_n, _ = kps_roi[0]
                    cv2.circle(img, (int(x_n), int(y_n)), 6, (0, 0, 255), -1)
                    for a, b in SKELETON:
                        xa, ya = kps_roi[a][0], kps_roi[a][1]
                        xb, yb = kps_roi[b][0], kps_roi[b][1]
                        cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (255, 0, 0), 2)

                # рисуем скелет на оба окна
                draw_skeleton(win1)
                draw_skeleton(win2)

                # показываем зону 25–75% вертикальными линиями
                cv2.line(win1, (int(0.25 * WIN_W), 0), (int(0.25 * WIN_W), WIN_H), (128, 128, 128), 1)
                cv2.line(win1, (int(0.75 * WIN_W), 0), (int(0.75 * WIN_W), WIN_H), (128, 128, 128), 1)
                cv2.line(win2, (int(0.25 * WIN_W), 0), (int(0.25 * WIN_W), WIN_H), (128, 128, 128), 1)
                cv2.line(win2, (int(0.75 * WIN_W), 0), (int(0.75 * WIN_W), WIN_H), (128, 128, 128), 1)

                # если кадр в рабочей зоне — считаем эмбеддинг и печатаем несколько чисел
                if in_central_zone:
                    emb, dbg = frame_embedding_enhanced(kps_roi)

                    # краткий набор углов и длин для вывода
                    txts = [
                        f"L_knee : {np.degrees(dbg['angles']['L_knee']):.1f}",  # (левое колено)
                        f"R_knee : {np.degrees(dbg['angles']['R_knee']):.1f}", # (правое колено)
                        f"L_hip  :   {np.degrees(dbg['angles']['L_hip']):.1f}", # (левое бедро)
                        f"R_hip  :  {np.degrees(dbg['angles']['R_hip']):.1f}", # (правое бедро)
                        f"L_elbow : {np.degrees(dbg['angles']['L_elbow']):.1f}", # (левый локоть)
                        f"R_elbow : {np.degrees(dbg['angles']['R_elbow']):.1f}", # (правый локоть)
                        f"Leg length:   {dbg['leg_len']:.2f}",
                        f"Torso length: {dbg['torso_len']:.2f}",
                    ]

                    y0 = 20
                    dy = 18
                    for i, t in enumerate(txts):
                        cv2.putText(
                            win2,
                            t,
                            (10, y0 + i * dy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                else:
                    cv2.putText(
                        win2,
                        "Outside 25-75% zone",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.putText(win2, "[SPACE]=pause/resume, [ESC]=exit",
                            (10, WIN_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1, cv2.LINE_AA)

                cv2.imshow("Person + skeleton", win1)
                cv2.imshow("Skeleton + debug", win2)

            # добавляем эмбеддинг только если кадр в рабочей зоне
            if in_central_zone:
                emb, _ = frame_embedding_enhanced(kps_roi)
                EMBEDS.append(emb)

        # обработка клавиш
        key = cv2.waitKey(1) & 0xFF if visualize else -1
        if key == 27:  # Esc
            break
        if key == 32:  # Space
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

    if not EMBEDS:
        print(f"Не удалось набрать кадры в рабочей зоне, video={video_path}")
        return None

    EMBEDS = np.stack(EMBEDS, axis=0).astype(np.float32)  # (T, 41)

    # Преобразуем в тензор PyTorch: batch=1
    x = torch.from_numpy(EMBEDS).unsqueeze(0)  # (1, T, 41)

    with torch.no_grad():
        video_emb_torch = gait_model(x)  # (1, emb_dim)
    video_embedding = video_emb_torch.cpu().numpy()[0]  # (emb_dim,)

    return video_embedding

def build_sequence_embedding(model, video_path: str, visualize: bool = True) -> np.ndarray:
    """
    Возвращает матрицу EMBEDS: shape (T, 41) для данного видео.
    Логика почти как в build_video_embedding, только мы НЕ усредняем по кадрам.
    """
    WIN_W, WIN_H = 640, 480
    CONF_THR = 0.5

    EMBEDS = []  # сюда эмбеддинги только из центральной зоны

    cap = cv2.VideoCapture(video_path)
    paused = False

    SKELETON = [
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 6),
        (5, 11), (6, 12),
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
    ]

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            if len(results) == 0:
                key = cv2.waitKey(1) & 0xFF if visualize else -1
                if key == 27:
                    break
                if key == 32:
                    paused = not paused
                continue

            r = results[0]
            if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
                key = cv2.waitKey(1) & 0xFF if visualize else -1
                if key == 27:
                    break
                if key == 32:
                    paused = not paused
                continue

            box = r.boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)

            kps_all = r.keypoints.data.cpu().numpy()
            confs_all = r.keypoints.conf.cpu().numpy()

            kps = kps_all[0]
            conf = confs_all[0]

            needed_ids = [0] + list(range(5, 17))
            all_visible = True
            for idx in needed_ids:
                x, y, _ = kps[idx]
                c = conf[idx]
                if not (c > CONF_THR and x > 0 and y > 0):
                    all_visible = False
                    break

            if not all_visible:
                key = cv2.waitKey(1) & 0xFF if visualize else -1
                if key == 27:
                    break
                if key == 32:
                    paused = not paused
                continue

            person_roi = frame[y1:y2, x1:x2].copy()
            if person_roi.size == 0:
                key = cv2.waitKey(1) & 0xFF if visualize else -1
                if key == 27:
                    break
                if key == 32:
                    paused = not paused
                continue

            h_bbox, w_bbox = person_roi.shape[:2]
            scale = min(WIN_W / w_bbox, WIN_H / h_bbox)
            new_w = int(w_bbox * scale)
            new_h = int(h_bbox * scale)

            person_scaled = cv2.resize(person_roi, (new_w, new_h))

            offset_x = (WIN_W - new_w) // 2
            offset_y = (WIN_H - new_h) // 2

            kps_roi = kps.copy()
            kps_roi[:, 0] = (kps_roi[:, 0] - x1) * scale + offset_x
            kps_roi[:, 1] = (kps_roi[:, 1] - y1) * scale + offset_y

            # координата носа в окне [0, WIN_W]
            x_nose, y_nose, _ = kps_roi[0]
            x_norm = x_nose / WIN_W  # [0, 1]

            in_central_zone = (0.25 <= x_norm <= 0.75)

            if visualize:
                # окно 1: человек + скелет
                win1 = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
                win1[offset_y:offset_y + person_scaled.shape[0],
                     offset_x:offset_x + person_scaled.shape[1]] = person_scaled.copy()

                # окно 2: скелет + служебная инфа
                win2 = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)

                def draw_skeleton(img, color=(0, 255, 0)):
                    for i in range(5, 17):
                        x, y = kps_roi[i][0], kps_roi[i][1]
                        cv2.circle(img, (int(x), int(y)), 4, color, -1)
                    x_n, y_n, _ = kps_roi[0]
                    cv2.circle(img, (int(x_n), int(y_n)), 6, (0, 0, 255), -1)
                    for a, b in SKELETON:
                        xa, ya = kps_roi[a][0], kps_roi[a][1]
                        xb, yb = kps_roi[b][0], kps_roi[b][1]
                        cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (255, 0, 0), 2)

                # рисуем скелет на оба окна
                draw_skeleton(win1)
                draw_skeleton(win2)

                # показываем зону 25–75% вертикальными линиями
                cv2.line(win1, (int(0.25 * WIN_W), 0), (int(0.25 * WIN_W), WIN_H), (128, 128, 128), 1)
                cv2.line(win1, (int(0.75 * WIN_W), 0), (int(0.75 * WIN_W), WIN_H), (128, 128, 128), 1)
                cv2.line(win2, (int(0.25 * WIN_W), 0), (int(0.25 * WIN_W), WIN_H), (128, 128, 128), 1)
                cv2.line(win2, (int(0.75 * WIN_W), 0), (int(0.75 * WIN_W), WIN_H), (128, 128, 128), 1)

                # если кадр в рабочей зоне — считаем эмбеддинг и печатаем несколько чисел
                if in_central_zone:
                    emb, dbg = frame_embedding_enhanced(kps_roi)

                    # краткий набор углов и длин для вывода
                    txts = [
                        f"L_knee : {np.degrees(dbg['angles']['L_knee']):.1f}",  # (левое колено)
                        f"R_knee : {np.degrees(dbg['angles']['R_knee']):.1f}", # (правое колено)
                        f"L_hip  :   {np.degrees(dbg['angles']['L_hip']):.1f}", # (левое бедро)
                        f"R_hip  :  {np.degrees(dbg['angles']['R_hip']):.1f}", # (правое бедро)
                        f"L_elbow : {np.degrees(dbg['angles']['L_elbow']):.1f}", # (левый локоть)
                        f"R_elbow : {np.degrees(dbg['angles']['R_elbow']):.1f}", # (правый локоть)
                        f"Leg length:   {dbg['leg_len']:.2f}",
                        f"Torso length: {dbg['torso_len']:.2f}",
                    ]

                    y0 = 20
                    dy = 18
                    for i, t in enumerate(txts):
                        cv2.putText(
                            win2,
                            t,
                            (10, y0 + i * dy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                else:
                    cv2.putText(
                        win2,
                        "Outside 25-75% zone",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.putText(win2, "[SPACE]=pause/resume, [ESC]=exit",
                            (10, WIN_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1, cv2.LINE_AA)

                cv2.imshow("Person + skeleton", win1)
                cv2.imshow("Skeleton + debug", win2)

            # добавляем эмбеддинг только если кадр в рабочей зоне
            if in_central_zone:
                emb, _ = frame_embedding_enhanced(kps_roi)
                EMBEDS.append(emb)

        # обработка клавиш
        key = cv2.waitKey(1) & 0xFF if visualize else -1
        if key == 27:  # Esc
            break
        if key == 32:  # Space
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

    if not EMBEDS:
        print(f"Не удалось набрать кадры в рабочей зоне, video={video_path}")
        return None

    EMBEDS = np.stack(EMBEDS, axis=0).astype(np.float32)  # (T, 41)
    return EMBEDS

gait_model = GaitLSTM(input_dim=41, hidden_dim=128, emb_dim=128, num_layers=1, num_classes=None)
state_dict = torch.load("gait_lstm.pth", map_location="cpu")
gait_model.load_state_dict({k: v for k, v in state_dict.items() if "fc_cls" not in k}, strict=False)
gait_model.eval()

def build_video_embedding_lstm(yolo_model, video_path: str) -> np.ndarray | None:
    seq = build_sequence_embedding(yolo_model, video_path, visualize=False)  # (T, 41)
    if seq is None:
        return None
    x = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0)  # (1, T, 41)
    with torch.no_grad():
        emb, _ = gait_model(x)
    return emb.cpu().numpy()[0]
