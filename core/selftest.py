"""
Self-test แบบออฟไลน์ — เดิน pipeline จริงทั้งเส้น (extract -> label -> train -> detect)
โดยไม่ต้องเปิด GUI  ใช้ยืนยันว่า .exe ที่แพ็กแล้วทำงานได้บนเครื่องที่ตัดเน็ต

เรียกใช้:  DroneCVEdu.exe --selftest "C:\\path\\to\\video.mp4"
คืน exit code 0 = ผ่าน, 1 = ไม่ผ่าน
"""

import os
import sys
import traceback


def _noop(*_args, **_kwargs):
    return None


def _find_video(argv) -> str | None:
    for a in argv:
        if a.lower().endswith((".mp4", ".mov", ".avi", ".mkv")) and os.path.exists(a):
            return a
    return None


def _synth_srt(video: str, out_path: str) -> None:
    """สร้างไฟล์ .SRT สังเคราะห์ให้ครอบคลุมความยาววิดีโอ (เดินเส้นตรงรอบพิกัดกลางกรุงเทพฯ)"""
    import cv2

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    dur = int((cap.get(cv2.CAP_PROP_FRAME_COUNT) or fps) / fps) + 1
    cap.release()

    lat0, lon0 = 13.736700, 100.523400
    lines = []
    for i in range(max(dur, 2)):
        t0 = f"00:00:{i:02d},000"
        t1 = f"00:00:{i + 1:02d},000"
        lat = lat0 + i * 0.00003
        lon = lon0 + i * 0.00003
        lines.append(
            f"{i + 1}\n{t0} --> {t1}\n"
            f"<font size=\"28\">[focal_len : 24.00] [latitude: {lat:.6f}] "
            f"[longitude: {lon:.6f}] [rel_alt: 60.000 abs_alt: 80.000] "
            f"[gimbal_yaw: 0.0 gimbal_pitch: -90.0]</font>\n"
        )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run(argv=None) -> int:
    argv = argv if argv is not None else sys.argv

    # บังคับออฟไลน์ + block ทุก HTTP (พิสูจน์ว่าไม่มีการดาวน์โหลด)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("YOLO_OFFLINE", "1")
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:9"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:9"

    video = _find_video(argv)
    if not video:
        print("SELFTEST FAIL: ไม่พบไฟล์วิดีโอใน argument (ใส่ path .mp4 ต่อท้าย --selftest)")
        return 1

    try:
        from core.labeling import image_size, save_label
        from core.state import (
            dataset_dir,
            ensure_dirs,
            frames_dir,
            labels_dir,
            output_dir,
            runs_dir,
        )
        from core.training import train_model
        from core.video import extract_frames
        from core.detect import process_video

        ensure_dirs()
        classes = ["target"]

        print("[1/4] extract frames ...", flush=True)
        n, frames = extract_frames(video, "interval_s", 1.0, 8, frames_dir(), progress=_noop)
        assert n > 0 and frames, "แตกเฟรมไม่ได้"

        print(f"[2/4] label {len(frames)} frames (กล่องกลางภาพ) ...", flush=True)
        for fr in frames:
            w, h = image_size(fr)
            box = [{
                "xmin": int(w * 0.35), "ymin": int(h * 0.35),
                "xmax": int(w * 0.65), "ymax": int(h * 0.65),
                "label": "target",
            }]
            save_label(labels_dir(), fr, box, classes)

        print("[3/4] train 2 epochs (yolov8n, offline) ...", flush=True)
        best, metrics, _summary, _plots = train_model(
            frames_dir(), labels_dir(), dataset_dir(), runs_dir(), classes,
            epochs=2, imgsz=320, base_model_label="yolov8n (เล็ก/เร็ว)", progress=_noop,
        )
        assert best and os.path.exists(best), "ไม่พบไฟล์โมเดล best.pt"

        print("[4/5] detect + geo (SRT สังเคราะห์) ...", flush=True)
        srt_path = os.path.join(output_dir(), "_selftest.srt")
        _synth_srt(video, srt_path)
        # ใช้โมเดลสำเร็จรูป + conf ต่ำ ให้มั่นใจว่าเจอวัตถุ (โมเดลที่เทรน 2 epoch อาจไม่เจออะไรเลย)
        out, report, geo_summary = process_video(
            video, None, output_dir(), 0.10, 0.45, "",
            srt_path=srt_path, progress=_noop,
        )
        size = os.path.getsize(out) if os.path.exists(out) else 0
        assert size > 0, "ไม่มีวิดีโอผลลัพธ์"

        print("[5/5] ตรวจไฟล์แผนที่ ...", flush=True)
        geojson = os.path.join(output_dir(), "detections.geojson")
        map_html = os.path.join(output_dir(), "map.html")
        assert geo_summary and geo_summary["n_points"] >= 1, "ไม่ได้พิกัดวัตถุจาก SRT"
        assert os.path.exists(geojson), "ไม่มี detections.geojson"
        assert os.path.exists(map_html), "ไม่มี map.html"

        print(f"\nSELFTEST PASS  |  metrics={metrics}  |  video={out} ({size} bytes)")
        print(f"  geo: {geo_summary['n_points']} จุด, per_class={geo_summary['per_class']}")
        print(f"  {report}".replace("\n", "\n  "))
        return 0

    except Exception as e:  # noqa: BLE001
        print(f"\nSELFTEST FAIL: {e}")
        traceback.print_exc()
        return 1
