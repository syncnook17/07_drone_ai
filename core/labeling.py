"""
ขั้นที่ 3: วาด bounding box ในแอป แล้วแปลงเป็นไฟล์ label รูปแบบ YOLO (.txt)

รูปแบบ YOLO ต่อ 1 บรรทัด: <class_idx> <cx> <cy> <w> <h>  (ทุกค่าหารด้วยขนาดภาพให้อยู่ 0-1)
ค่าที่ image_annotator คืนมา: {"image": ..., "boxes": [{"xmin","ymin","xmax","ymax","label"}, ...]}
โดย xmin/ymin/xmax/ymax เป็นพิกัด pixel ของภาพต้นฉบับ
"""

import os

import cv2


def label_txt_path(labels_dir: str, frame_path: str) -> str:
    # ชื่อไฟล์ label ต้องตรงกับชื่อไฟล์ภาพ (frame_00001.jpg -> frame_00001.txt)
    stem = os.path.splitext(os.path.basename(frame_path))[0]
    return os.path.join(labels_dir, f"{stem}.txt")


def image_size(frame_path: str) -> tuple[int, int]:
    img = cv2.imread(frame_path)
    if img is None:
        raise ValueError(f"อ่านภาพไม่ได้: {frame_path}")
    h, w = img.shape[:2]
    return w, h


def boxes_to_yolo(boxes: list[dict], img_w: int, img_h: int, class_names: list[str]) -> str:
    """แปลง list กล่องจาก annotator เป็นข้อความ label แบบ YOLO (อาจเป็นสตริงว่าง = ไม่มีวัตถุ)"""
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    lines = []
    for b in boxes or []:
        label = b.get("label", "")
        if label not in name_to_idx:
            # กล่องที่ไม่มี label หรือ label ไม่อยู่ในรายการคลาส -> ข้าม
            continue
        x1, y1, x2, y2 = b["xmin"], b["ymin"], b["xmax"], b["ymax"]
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        # clamp ให้อยู่ในกรอบภาพ กันค่าเกินตอนลากกล่องออกนอกขอบ
        x1, x2 = max(0, x1), min(img_w, x2)
        y1, y2 = max(0, y1), min(img_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"{name_to_idx[label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


def yolo_to_boxes(txt_path: str, img_w: int, img_h: int, class_names: list[str]) -> list[dict]:
    """อ่านไฟล์ label กลับมาเป็น list กล่องสำหรับโชว์ใน annotator ตอนเปิดภาพซ้ำ"""
    if not os.path.exists(txt_path):
        return []
    boxes = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            idx, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
            label = class_names[idx] if 0 <= idx < len(class_names) else str(idx)
            bw, bh = w * img_w, h * img_h
            x1 = cx * img_w - bw / 2
            y1 = cy * img_h - bh / 2
            boxes.append(
                {
                    "xmin": int(round(x1)),
                    "ymin": int(round(y1)),
                    "xmax": int(round(x1 + bw)),
                    "ymax": int(round(y1 + bh)),
                    "label": label,
                }
            )
    return boxes


def save_label(labels_dir: str, frame_path: str, boxes: list[dict], class_names: list[str]) -> int:
    """เขียนไฟล์ label ของเฟรมนี้ (boxes ว่าง = เขียนไฟล์เปล่า = ภาพนี้ไม่มีวัตถุ)
    คืนจำนวนกล่องที่บันทึกได้จริง"""
    os.makedirs(labels_dir, exist_ok=True)
    img_w, img_h = image_size(frame_path)
    content = boxes_to_yolo(boxes, img_w, img_h, class_names)
    with open(label_txt_path(labels_dir, frame_path), "w", encoding="utf-8") as f:
        f.write(content + ("\n" if content else ""))
    return 0 if not content else content.count("\n") + 1


def labeled_count(labels_dir: str, frames: list[str]) -> int:
    """นับจำนวนเฟรมที่มีไฟล์ label แล้ว (รวมไฟล์เปล่าที่กด 'ไม่มีวัตถุ')"""
    if not os.path.isdir(labels_dir):
        return 0
    return sum(1 for fr in frames if os.path.exists(label_txt_path(labels_dir, fr)))
