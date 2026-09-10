"""
ขั้นที่ 5: รันโมเดลตรวจจับวัตถุใส่ทุกเฟรมของวิดีโอ + นับจำนวนไม่ซ้ำตัวด้วย tracking
refactor จาก process_video เดิม (ตรรกะเกือบทั้งหมดยกมา)
"""

import os

import cv2
import gradio as gr
import imageio_ffmpeg
import numpy as np
from ultralytics import YOLO

from core import geo, geomap, paths


def process_video(
    video_path,
    model_path,
    session_dir,
    conf,
    iou,
    class_filter,
    srt_path=None,
    camera_preset=None,
    manual_hfov=0.0,
    progress=gr.Progress(),
):
    if not video_path:
        raise gr.Error("กรุณาเลือกวิดีโอสำหรับตรวจจับ")

    # ถ้ายังไม่เคยเทรนโมเดล ให้ fallback ไปใช้ yolov8n.pt สำเร็จรูป (80 class ตาม COCO)
    used_pretrained = not (model_path and os.path.exists(model_path))
    resolved_model = paths.YOLOV8N if used_pretrained else model_path

    try:
        model = YOLO(resolved_model)
    except Exception as e:
        raise gr.Error(f"โหลดโมเดลไม่สำเร็จ: {e}")

    # แปลงชื่อ class ที่ผู้ใช้กรอกเป็น index เพื่อกรองเฉพาะ class ที่ต้องการ
    class_indices = None
    if class_filter and class_filter.strip():
        wanted_names = [c.strip().lower() for c in class_filter.split(",") if c.strip()]
        name_to_idx = {name.lower(): idx for idx, name in model.names.items()}
        class_indices = []
        for name in wanted_names:
            if name not in name_to_idx:
                available = ", ".join(sorted(model.names.values()))
                raise gr.Error(f"ไม่พบ class '{name}' ในโมเดลนี้ กรุณาเลือกจาก: {available}")
            class_indices.append(name_to_idx[name])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("เปิดไฟล์วิดีโอไม่สำเร็จ กรุณาตรวจสอบว่าเป็นไฟล์ .mp4 ที่ถูกต้อง")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    os.makedirs(session_dir, exist_ok=True)
    output_path = os.path.join(session_dir, "output_detected.mp4")

    # ใช้ imageio-ffmpeg เขียนวิดีโอ codec libx264 (H.264) แทน cv2.VideoWriter
    # เพราะ cv2.VideoWriter บนเครื่องนี้เขียนได้แค่ mp4v ซึ่งเบราว์เซอร์เล่นไม่ได้
    writer = imageio_ffmpeg.write_frames(
        output_path,
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="bgr24",
        macro_block_size=1,
    )
    writer.send(None)  # seed generator

    # ---- พิกัดจากไฟล์ .SRT (ถ้ามี) ----
    srt_samples = geo.parse_srt(srt_path) if srt_path and os.path.exists(srt_path) else []
    geo_raw: list[dict] = []

    track_ids_by_class = {}

    # ByteTrack (tracker เริ่มต้น) มีเกณฑ์ในตัว 0.25 แยกจาก conf ที่ผู้ใช้ตั้ง
    # ถ้าโมเดลอ่อนและ conf สูงสุดต่ำกว่า 0.25 จะแทบไม่มี track ใหม่เกิดขึ้นเลย
    # จึงผูกเกณฑ์ของ tracker เข้ากับค่า conf ที่ผู้ใช้ตั้งไว้แทน
    tracker_config_path = os.path.join(session_dir, "tracker.yaml")
    with open(tracker_config_path, "w", encoding="utf-8") as f:
        f.write(
            "tracker_type: bytetrack\n"
            f"track_high_thresh: {max(conf, 0.01)}\n"
            f"track_low_thresh: {max(conf * 0.5, 0.005)}\n"
            f"new_track_thresh: {max(conf, 0.01)}\n"
            "track_buffer: 30\n"
            "match_thresh: 0.8\n"
            "fuse_score: true\n"
        )

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.track(
                frame,
                conf=conf,
                iou=iou,
                classes=class_indices,
                persist=True,
                tracker=tracker_config_path,
                verbose=False,
            )
            annotated = results[0].plot()
            writer.send(np.ascontiguousarray(annotated))

            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                for cls_idx, track_id in zip(boxes.cls.tolist(), boxes.id.tolist()):
                    class_name = model.names[int(cls_idx)]
                    track_ids_by_class.setdefault(class_name, set()).add(int(track_id))

                if srt_samples:
                    pose = geo.pose_at(srt_samples, frame_idx / fps)
                    if pose and pose.rel_alt:
                        hfov = geo.resolve_hfov(pose.focal35, camera_preset, manual_hfov, width, height)
                        for (cx, cy, _bw, _bh), cls_idx, track_id, cf in zip(
                            boxes.xywh.tolist(), boxes.cls.tolist(),
                            boxes.id.tolist(), boxes.conf.tolist(),
                        ):
                            world = geo.pixel_to_world(pose, cx, cy, width, height, hfov)
                            if world:
                                lat, lon, oblique = world
                                geo_raw.append({
                                    "track_id": int(track_id),
                                    "cls": model.names[int(cls_idx)],
                                    "lat": lat, "lon": lon, "conf": float(cf),
                                    "oblique": oblique, "frame": frame_idx,
                                })

            frame_idx += 1
            progress(min(frame_idx / total_frames, 1.0), desc=f"กำลังประมวลผลเฟรม {frame_idx}/{total_frames}")
    except Exception as e:
        raise gr.Error(f"ประมวลผลวิดีโอไม่สำเร็จ: {e}")
    finally:
        cap.release()
        writer.close()

    if frame_idx == 0:
        raise gr.Error("ไม่พบเฟรมในวิดีโอที่อัปโหลด กรุณาตรวจสอบไฟล์อีกครั้ง")

    lines = []
    if used_pretrained:
        lines.append("⚠️ ยังไม่ได้เทรนโมเดล กำลังใช้โมเดลสำเร็จรูป yolov8n (80 คลาสทั่วไป)")
    if track_ids_by_class:
        lines.append("จำนวนที่ตรวจพบ (นับไม่ซ้ำตัวด้วย tracking):")
        lines += [f"  {cls}: {len(ids)} ชิ้น" for cls, ids in sorted(track_ids_by_class.items())]
    else:
        lines.append("ไม่พบวัตถุในวิดีโอนี้")

    geo_summary = None
    if geo_raw:
        points = geomap.dedup_video(geo_raw)
        geo_summary = geomap.export_all(points, session_dir)
        lines.append(f"📍 บันทึกพิกัดวัตถุ {geo_summary['n_points']} จุด — ไปแท็บ \"6. แผนที่\"")
        if geo_summary["has_oblique"]:
            lines.append("   (บางเฟรมกล้องเอียงมาก พิกัดอาจคลาดเคลื่อนสูง)")
    elif srt_path and os.path.exists(srt_path):
        if not srt_samples:
            lines.append("⚠️ อ่านไฟล์ .SRT ไม่พบพิกัด — ตรวจว่าเป็นไฟล์ subtitle จากโดรนจริง")
        else:
            lines.append("⚠️ ไฟล์ .SRT ไม่มีค่าความสูง (rel_alt) จึงคำนวณพิกัดวัตถุไม่ได้")

    return output_path, "\n".join(lines), geo_summary
