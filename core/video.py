"""
ขั้นที่ 2: แตกวิดีโอเป็นเฟรมภาพ .jpg เพื่อเอาไป label ต่อ
"""

import os

import cv2
import gradio as gr

from core.state import robust_rmtree


def extract_frames(video_path, mode, value, max_frames, out_dir, progress=gr.Progress()):
    """
    mode = "every_n"    -> เก็บทุก ๆ value เฟรม
    mode = "interval_s" -> เก็บทุก ๆ value วินาที (คำนวณจาก FPS ของวิดีโอ)
    หยุดเมื่อได้ครบ max_frames ภาพ (กันเครื่องช้า/ดิสก์เต็ม)
    คืน (จำนวนเฟรมที่ได้, list path เฟรมเรียงตามลำดับ)
    """
    if not video_path:
        raise gr.Error("ยังไม่ได้อัปโหลดวิดีโอ กรุณากลับไปขั้นที่ 1")
    if not os.path.exists(video_path):
        raise gr.Error("หาไฟล์วิดีโอไม่พบ กรุณาอัปโหลดใหม่ในขั้นที่ 1")

    value = float(value)
    max_frames = int(max_frames)
    if value <= 0:
        raise gr.Error("ค่าช่วงการเก็บเฟรมต้องมากกว่า 0")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("เปิดไฟล์วิดีโอไม่สำเร็จ กรุณาตรวจสอบว่าเป็นไฟล์ .mp4 ที่ถูกต้อง")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # แปลงโหมด "ทุก X วินาที" ให้เป็น "ทุก N เฟรม" โดยใช้ FPS จริงของวิดีโอ
    if mode == "interval_s":
        step = max(int(round(fps * value)), 1)
    else:
        step = max(int(round(value)), 1)

    # ล้างเฟรมเก่าทุกครั้งก่อนแตกใหม่ กันเฟรมชุดเก่าปนกับชุดใหม่
    robust_rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    progress(0, desc="กำลังแตกเฟรม...")

    saved = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            out_path = os.path.join(out_dir, f"frame_{len(saved) + 1:05d}.jpg")
            cv2.imwrite(out_path, frame)
            saved.append(out_path)
            if total_frames:
                progress(min(frame_idx / total_frames, 1.0), desc=f"เก็บแล้ว {len(saved)} เฟรม")
            if len(saved) >= max_frames:
                break
        frame_idx += 1
    cap.release()

    if not saved:
        raise gr.Error("แตกเฟรมไม่ได้เลย วิดีโออาจสั้นเกินไปหรือช่วงการเก็บเฟรมกว้างเกินไป")

    progress(1.0, desc="แตกเฟรมเสร็จ")
    return len(saved), saved
