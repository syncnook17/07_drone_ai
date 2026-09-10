"""
ขั้นที่ 4: เทรนโมเดล YOLOv8 จากเฟรม + label ที่ทำไว้ในขั้น 2-3
refactor จาก train_model เดิม: เลิกรับไฟล์อัปโหลด อ่านจากโฟลเดอร์ของ session โดยตรง
"""

import csv
import os
import shutil

import gradio as gr
from ultralytics import YOLO

from core import paths
from core.state import robust_rmtree

BASE_MODELS = {
    "yolov8n (เล็ก/เร็ว)": paths.YOLOV8N,
    "yolov8s (ใหญ่/แม่นกว่า)": paths.YOLOV8S,
}


def train_model(
    frames_dir,
    labels_dir,
    dataset_dir,
    runs_dir,
    class_names,
    epochs,
    imgsz,
    base_model_label,
    progress=gr.Progress(),
):
    if not class_names:
        raise gr.Error("ยังไม่ได้กรอกชื่อคลาสในขั้นที่ 3")

    frames = sorted(
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg"))
    ) if os.path.isdir(frames_dir) else []
    if not frames:
        raise gr.Error("ไม่พบเฟรมภาพ กรุณากลับไปแตกเฟรมในขั้นที่ 2")

    labeled = [
        f for f in frames
        if os.path.exists(os.path.join(labels_dir, os.path.splitext(os.path.basename(f))[0] + ".txt"))
    ]
    if not labeled:
        raise gr.Error("ยังไม่ได้ label ภาพเลย กรุณากลับไปวาดกล่องในขั้นที่ 3 อย่างน้อย 1 ภาพ")

    progress(0, desc="กำลังเตรียมโฟลเดอร์ dataset...")

    # ล้าง dataset เก่าทุกครั้งก่อนเทรนใหม่ กันข้อมูลชุดเก่าปนกับชุดใหม่
    robust_rmtree(dataset_dir)
    images_train_dir = os.path.join(dataset_dir, "images", "train")
    labels_train_dir = os.path.join(dataset_dir, "labels", "train")
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)

    for f in frames:
        stem = os.path.splitext(os.path.basename(f))[0]
        shutil.copy(f, os.path.join(images_train_dir, os.path.basename(f)))
        src_label = os.path.join(labels_dir, f"{stem}.txt")
        dst_label = os.path.join(labels_train_dir, f"{stem}.txt")
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
        else:
            # เฟรมที่ผู้เรียนยังไม่ได้ label ถือว่า "ไม่มีวัตถุ" -> สร้างไฟล์ .txt เปล่า
            # ไม่งั้น YOLO training จะ error เพราะหาไฟล์ label คู่กับภาพไม่เจอ
            open(dst_label, "w").close()

    progress(0.1, desc="กำลังสร้างไฟล์ dataset.yaml...")

    # ข้อมูลมีน้อย (เดโม) จึงใช้ images/train ชุดเดียวกันเป็นทั้ง train และ val
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    yaml_lines = [f"path: {dataset_dir}", "train: images/train", "val: images/train", "names:"]
    for i, c in enumerate(class_names):
        yaml_lines.append(f"  {i}: {c}")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")

    weights = BASE_MODELS.get(base_model_label, paths.YOLOV8N)
    progress(0.2, desc=f"กำลังโหลดโมเดลตั้งต้น {os.path.basename(weights)}...")
    try:
        model = YOLO(weights)
    except Exception as e:
        raise gr.Error(f"โหลดโมเดลตั้งต้นไม่สำเร็จ: {e}")

    total_epochs = int(epochs)

    def on_epoch_end(trainer):
        current = trainer.epoch + 1
        progress(0.2 + 0.75 * (current / total_epochs), desc=f"กำลังเทรน epoch {current}/{total_epochs}")

    model.add_callback("on_train_epoch_end", on_epoch_end)

    try:
        robust_rmtree(runs_dir)
        model.train(
            data=yaml_path,
            epochs=total_epochs,
            imgsz=int(imgsz),
            project=runs_dir,
            name="train",
            exist_ok=True,
            verbose=False,
            plots=True,
            amp=False,  # ผลนิ่ง + ไม่ต้องโหลดโมเดลเช็ก AMP จากเน็ต (รันออฟไลน์)
        )
    except Exception as e:
        raise gr.Error(f"เทรนโมเดลไม่สำเร็จ: {e}")

    progress(0.98, desc="กำลังบันทึกโมเดล...")

    best_path = os.path.join(runs_dir, "train", "weights", "best.pt")
    if not os.path.exists(best_path):
        raise gr.Error("เทรนเสร็จแล้วแต่หาไฟล์โมเดล best.pt ไม่พบ กรุณาลองเทรนใหม่")

    metrics = _read_last_metrics(os.path.join(runs_dir, "train", "results.csv"))
    plots = [
        p
        for p in (
            os.path.join(runs_dir, "train", "results.png"),
            os.path.join(runs_dir, "train", "confusion_matrix.png"),
        )
        if os.path.exists(p)
    ]

    summary = [
        "เทรนโมเดลสำเร็จ!",
        f"คลาส: {', '.join(class_names)}",
        f"ภาพที่ label แล้ว: {len(labeled)}/{len(frames)}",
    ]
    if metrics:
        summary.append(
            "ผลตอน validate — "
            f"mAP50: {metrics.get('mAP50', 0):.3f} | "
            f"Precision: {metrics.get('precision', 0):.3f} | "
            f"Recall: {metrics.get('recall', 0):.3f}"
        )
    progress(1.0, desc="เทรนเสร็จสมบูรณ์")
    return best_path, metrics, "\n".join(summary), plots


def _read_last_metrics(csv_path: str) -> dict:
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    last = {k.strip(): v for k, v in rows[-1].items()}

    def g(*keys):
        for k in keys:
            if k in last and last[k] not in ("", None):
                try:
                    return float(last[k])
                except ValueError:
                    pass
        return 0.0

    return {
        "mAP50": g("metrics/mAP50(B)", "metrics/mAP_0.5"),
        "mAP50-95": g("metrics/mAP50-95(B)", "metrics/mAP_0.5:0.95"),
        "precision": g("metrics/precision(B)", "metrics/precision"),
        "recall": g("metrics/recall(B)", "metrics/recall"),
    }
