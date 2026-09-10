"""
ตรวจจับวัตถุในชุดภาพนิ่งจากโดรน + คำนวณพิกัดจาก EXIF/XMP ของแต่ละภาพ
(คู่กับ core/detect.py ที่ทำฝั่งวิดีโอ+SRT)
"""

import os

import cv2
import gradio as gr
from ultralytics import YOLO

from core import geo, geomap, paths


def _resolve_class_indices(model, class_filter):
    if not class_filter or not class_filter.strip():
        return None
    wanted = [c.strip().lower() for c in class_filter.split(",") if c.strip()]
    name_to_idx = {name.lower(): idx for idx, name in model.names.items()}
    out = []
    for name in wanted:
        if name not in name_to_idx:
            available = ", ".join(sorted(model.names.values()))
            raise gr.Error(f"ไม่พบ class '{name}' ในโมเดลนี้ กรุณาเลือกจาก: {available}")
        out.append(name_to_idx[name])
    return out


def process_photos(
    photo_paths,
    model_path,
    session_dir,
    conf,
    iou,
    class_filter,
    camera_preset=None,
    manual_hfov=0.0,
    merge_dist_m=2.0,
    progress=gr.Progress(),
):
    if not photo_paths:
        raise gr.Error("ยังไม่มีภาพถ่ายสำหรับตรวจจับ")

    used_pretrained = not (model_path and os.path.exists(model_path))
    resolved_model = paths.YOLOV8N if used_pretrained else model_path
    try:
        model = YOLO(resolved_model)
    except Exception as e:
        raise gr.Error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    class_indices = _resolve_class_indices(model, class_filter)

    annotated_dir = os.path.join(session_dir, "photos_annotated")
    os.makedirs(annotated_dir, exist_ok=True)

    geo_raw: list[dict] = []
    annotated_paths: list[str] = []
    n_total = len(photo_paths)
    n_no_gps = 0
    n_no_alt = 0
    n_detections = 0

    for i, path in enumerate(photo_paths):
        progress(i / max(n_total, 1), desc=f"ตรวจจับภาพ {i + 1}/{n_total}")
        try:
            results = model.predict(
                path, conf=conf, iou=iou, classes=class_indices, verbose=False
            )
        except Exception as e:
            raise gr.Error(f"ตรวจจับภาพ {os.path.basename(path)} ไม่สำเร็จ: {e}")

        res = results[0]
        img_h, img_w = res.orig_shape
        out_img = os.path.join(annotated_dir, os.path.basename(path))
        cv2.imwrite(out_img, res.plot())
        annotated_paths.append(out_img)

        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            continue
        n_detections += len(boxes)

        pose = geo.photo_pose(path)
        if pose is None:
            n_no_gps += 1
            continue
        if not pose.rel_alt:
            n_no_alt += 1
            continue

        hfov = geo.resolve_hfov(pose.focal35, camera_preset, manual_hfov, img_w, img_h)
        for (cx, cy, _bw, _bh), cls_idx, cf in zip(
            boxes.xywh.tolist(), boxes.cls.tolist(), boxes.conf.tolist()
        ):
            world = geo.pixel_to_world(pose, cx, cy, img_w, img_h, hfov)
            if world:
                lat, lon, oblique = world
                geo_raw.append({
                    "cls": model.names[int(cls_idx)],
                    "lat": lat, "lon": lon, "conf": float(cf),
                    "oblique": oblique, "source": os.path.basename(path),
                })

    progress(1.0, desc="รวมผล + สร้างไฟล์แผนที่")

    lines = []
    if used_pretrained:
        lines.append("⚠️ ยังไม่ได้เทรนโมเดล กำลังใช้โมเดลสำเร็จรูป yolov8n (80 คลาสทั่วไป)")
    lines.append(f"ตรวจแล้ว {n_total} ภาพ · พบวัตถุรวม {n_detections} กล่อง")

    geo_summary = None
    if geo_raw:
        points = geomap.dedup_photos(geo_raw, float(merge_dist_m or 2.0))
        geo_summary = geomap.export_all(points, session_dir)
        lines.append(f"📍 พิกัดวัตถุ (รวมจุดซ้ำในรัศมี {merge_dist_m:g} ม.): {geo_summary['n_points']} จุด — ไปแท็บ \"6. แผนที่\"")
        if geo_summary["has_oblique"]:
            lines.append("   (บางภาพกล้องเอียงมาก พิกัดอาจคลาดเคลื่อนสูง)")
    if n_no_gps:
        lines.append(f"⚠️ {n_no_gps} ภาพไม่มี GPS ใน EXIF — ไม่ได้ลงแผนที่ (ใช้ไฟล์ .JPG ตรงจากโดรน)")
    if n_no_alt:
        lines.append(f"⚠️ {n_no_alt} ภาพไม่มีค่าความสูง (RelativeAltitude) — คำนวณพิกัดวัตถุไม่ได้")

    return annotated_paths, "\n".join(lines), geo_summary
