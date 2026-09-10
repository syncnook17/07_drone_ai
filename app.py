"""
คอร์สอบรม CV: เทรน + ทดสอบโมเดลตรวจจับวัตถุด้วย YOLOv8 (Gradio)

โปรแกรมนี้เป็น desktop app ผู้ใช้คนเดียวต่อเครื่อง (สุดท้ายแพ็กเป็น .exe)
state เก็บใน core.state.SESSION ตัวเดียว + persist ลงดิสก์ → รีเฟรช/ปิดเปิดใหม่ งานไม่หาย

flow 5 ขั้นตอนเรียงลำดับ (แต่ละขั้นเป็นแท็บ กดข้ามได้ + มีปุ่ม "ไปขั้นต่อไป"):
  1. อัปโหลดวิดีโอ
  2. แตกเฟรมจากวิดีโอ
  3. Label วัตถุที่ต้องการ (วาดกล่องในแอป)
  4. เทรนโมเดล
  5. ตรวจจับวัตถุในวิดีโอ (วิดีโอเดิม หรือ อัปโหลดใหม่)

parameter ทุกตัวมีปุ่ม ⓘ กดแล้วเด้ง modal อธิบายว่าใช้ทำอะไร ปรับขึ้น/ลงแล้วเกิดอะไร
"""

import os
import shutil

import cv2
import gradio as gr
from gradio_image_annotation import image_annotator

from core.detect import process_video
from core.detect_photos import process_photos
from core.geo import CAMERA_PRESETS, parse_srt
from core.labeling import (
    image_size,
    label_txt_path,
    labeled_count,
    save_label,
    yolo_to_boxes,
)
from core.params_help import PARAM_HELP
from core.state import (
    SESSION,
    WORKDIR,
    dataset_dir,
    frames_dir,
    labels_dir,
    output_dir,
    photos_dir,
    robust_rmtree,
    runs_dir,
    uploads_dir,
)
from core.training import BASE_MODELS, train_model
from core.video import extract_frames

CUSTOM_CSS = """
.info-modal-overlay {
    position: fixed; inset: 0; z-index: 2000;
    background: rgba(0, 0, 0, 0.55);
    display: flex; align-items: center; justify-content: center;
}
.info-modal-box {
    max-width: 560px; width: 90%;
    max-height: 80vh; overflow-y: auto;
    background: var(--body-background-fill);
    border: 1px solid var(--border-color-primary);
    border-radius: 14px; padding: 8px 22px 18px;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
}
.info-btn {
    min-width: 30px !important; max-width: 30px !important;
    height: 30px; padding: 0 !important; flex: 0 0 auto !important;
    border-radius: 50% !important; line-height: 1;
    font-size: 15px;
}
.step-hint { color: var(--body-text-color-subdued); font-size: 0.95em; }

/* หัวข้อพารามิเตอร์ + ปุ่ม ⓘ ให้ชิดกันซ้ายมือ ไม่ลอยไปทางขวา */
.param-head { gap: 8px !important; align-items: center !important; justify-content: flex-start !important; }
.param-head > * { flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; }
.param-head p { margin: 0 !important; font-weight: 600; }
.param-head .md { margin: 0 !important; }

/* กล่อง label รูป: คงสัดส่วนภาพจริง (canvas buffer มี aspect ถูกอยู่แล้ว —
   ปัญหาคือ CSS เดิม height:100% ยืดแนวตั้ง) จำกัดความกว้างกล่อง แล้วให้ความสูง canvas ตามภาพ */
.anno-box { max-width: min(72vh, 100%); margin-left: auto; margin-right: auto; }
.anno-box .canvas-container { max-height: none !important; }
.anno-box canvas.canvas-annotator {
    width: 100% !important;
    height: auto !important;
}
"""


def _classes_from_text(text: str) -> list[str]:
    return [c.strip() for c in (text or "").split(",") if c.strip()]


def _existing_plots() -> list[str] | None:
    plots = [
        p
        for p in (
            os.path.join(runs_dir(), "train", "results.png"),
            os.path.join(runs_dir(), "train", "confusion_matrix.png"),
        )
        if os.path.exists(p)
    ]
    return plots or None


def _model_summary() -> str:
    if SESSION.model_path and os.path.exists(SESSION.model_path):
        m = SESSION.train_metrics or {}
        return (
            "โมเดลที่เทรนไว้พร้อมใช้งาน\n"
            f"คลาส: {', '.join(SESSION.class_names)}\n"
            f"mAP50: {m.get('mAP50', 0):.3f} | Precision: {m.get('precision', 0):.3f} | "
            f"Recall: {m.get('recall', 0):.3f}"
        )
    return ""


# ---------------------------------------------------------------- ขั้นที่ 1
def _is_photos(kind) -> bool:
    return bool(kind) and "ภาพ" in kind


def set_input_kind(kind):
    SESSION.input_kind = "photos" if _is_photos(kind) else "video"
    SESSION.save()
    is_photo = SESSION.input_kind == "photos"
    return gr.update(visible=not is_photo), gr.update(visible=is_photo)


def set_video(path):
    if path:
        # คัดลอกวิดีโอเข้าโฟลเดอร์ของโปรแกรม (ที่ Gradio อัปมาอยู่ /tmp ซึ่งถูกล้างได้)
        # เพื่อให้ปิดเปิดโปรแกรมใหม่แล้ว "ใช้วิดีโอเดิม" ยังใช้ได้
        os.makedirs(uploads_dir(), exist_ok=True)
        dst = os.path.join(uploads_dir(), "source" + os.path.splitext(path)[1])
        if os.path.abspath(path) != os.path.abspath(dst):
            shutil.copy(path, dst)
        SESSION.video_path = dst
        SESSION.save()
        return "อัปโหลดวิดีโอเรียบร้อย ✅ กดปุ่ม \"ไปขั้นต่อไป\" ด้านล่างได้เลย"
    SESSION.video_path = None
    SESSION.save()
    return ""


def set_srt(fileobj):
    path = fileobj if isinstance(fileobj, str) else getattr(fileobj, "name", None)
    if path and os.path.exists(path):
        os.makedirs(uploads_dir(), exist_ok=True)
        dst = os.path.join(uploads_dir(), "flight.srt")
        if os.path.abspath(path) != os.path.abspath(dst):
            shutil.copy(path, dst)
        SESSION.srt_path = dst
        SESSION.save()
        n = len(parse_srt(dst))
        return f"แนบไฟล์ .SRT แล้ว — อ่านพิกัดได้ {n} จุด (ใช้ทำแผนที่ในขั้น 6)"
    SESSION.srt_path = None
    SESSION.save()
    return ""


def set_photos(files):
    pdir, fdir = photos_dir(), frames_dir()
    robust_rmtree(pdir)
    robust_rmtree(fdir)
    os.makedirs(pdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)
    paths_in = [f if isinstance(f, str) else getattr(f, "name", None) for f in (files or [])]
    paths_in = [p for p in paths_in if p and os.path.exists(p)]
    if not paths_in:
        SESSION.rescan_frames()
        SESSION.save()
        return "ยังไม่ได้เลือกภาพถ่าย", None
    bad = 0
    for i, src in enumerate(sorted(paths_in), 1):
        try:
            ext = os.path.splitext(src)[1].lower()
            if ext not in (".jpg", ".jpeg", ".png"):
                ext = ".jpg"
            # เก็บไฟล์ต้นฉบับเต็มความละเอียดไว้ (ใช้ตอนคำนวณพิกัดในขั้น 5)
            shutil.copy(src, os.path.join(pdir, f"photo_{i:05d}{ext}"))
            # สำเนาย่อขนาดไว้ label/เทรน (ให้แอปไม่อืดกับภาพ 12MP)
            img = cv2.imread(src)
            if img is None:
                bad += 1
                continue
            h, w = img.shape[:2]
            if max(h, w) > 1920:
                s = 1920.0 / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)))
            cv2.imwrite(os.path.join(fdir, f"photo_{i:05d}.jpg"), img)
        except Exception:
            bad += 1
    SESSION.rescan_frames()
    SESSION.save()
    n = len(SESSION.frames)
    if n == 0:
        return "❌ อ่านภาพไม่ได้เลย — ตรวจว่าเป็นไฟล์ .jpg/.png ปกติ", None
    msg = f"อัปโหลดภาพถ่าย {n} ภาพ ✅ ใช้เป็นเฟรมให้แล้ว — ข้ามขั้น 2 ไป label ในขั้น 3 ได้เลย"
    if bad:
        msg += f"  (ข้าม {bad} ไฟล์ที่อ่านไม่ได้)"
    return msg, SESSION.frames


# ---------------------------------------------------------------- ขั้นที่ 2
def do_extract(mode_label, value, max_frames):
    if SESSION.input_kind == "photos":
        SESSION.rescan_frames()
        return SESSION.frames or None, "โหมดภาพถ่าย: ใช้ภาพเป็นเฟรมอยู่แล้ว ข้ามไปขั้นที่ 3 ได้เลย"
    if not SESSION.video_path:
        raise gr.Error("ยังไม่ได้อัปโหลดวิดีโอในขั้นที่ 1")
    mode = "interval_s" if "วินาที" in mode_label else "every_n"
    count, frames = extract_frames(SESSION.video_path, mode, value, max_frames, frames_dir())
    SESSION.frames = frames
    SESSION.frame_mode = mode_label
    SESSION.frame_value = float(value)
    SESSION.max_frames = int(max_frames)
    SESSION.save()
    return frames, f"แตกเฟรมได้ {count} ภาพ — ไปขั้นที่ 3 เพื่อ label ได้เลย"


def step2_status():
    SESSION.rescan_frames()
    if SESSION.input_kind == "photos":
        n = len(SESSION.frames)
        if n:
            return f"โหมดภาพถ่าย: ใช้ภาพ {n} ภาพเป็นเฟรมแล้ว — ข้ามไปขั้นที่ 3 ได้เลย", SESSION.frames
        return "⚠️ ยังไม่ได้อัปโหลดภาพถ่ายในขั้นที่ 1", None
    if not SESSION.video_path:
        return "⚠️ ยังไม่มีวิดีโอ กรุณากลับไปขั้นที่ 1", None
    if SESSION.frames:
        return f"มีเฟรมที่แตกไว้แล้ว {len(SESSION.frames)} ภาพ (แตกใหม่จะทับของเดิม)", SESSION.frames
    return "พร้อมแตกเฟรมจากวิดีโอที่อัปโหลด", None


# ---------------------------------------------------------------- ขั้นที่ 3
def set_classes(text):
    classes = _classes_from_text(text)
    if not classes:
        raise gr.Error("กรุณากรอกชื่อคลาสอย่างน้อย 1 ชื่อ เช่น car หรือ car,truck")
    SESSION.class_names = classes
    SESSION.save()
    return f"ตั้งคลาสแล้ว: {', '.join(classes)}  ({len(classes)} คลาส)"


def refresh_label_ctx():
    SESSION.rescan_frames()
    return {
        "frames": list(SESSION.frames),
        "labels_dir": labels_dir(),
        "classes": list(SESSION.class_names),
        "idx": 0,
        "tick": 0,
    }


def step3_enter():
    return (
        ", ".join(SESSION.class_names),
        (f"ตั้งคลาสแล้ว: {', '.join(SESSION.class_names)}" if SESSION.class_names else ""),
        refresh_label_ctx(),
    )


# ---------------------------------------------------------------- ขั้นที่ 4
def step4_status():
    SESSION.rescan_frames()
    frames = SESSION.frames
    done = labeled_count(labels_dir(), frames)
    parts = [
        f"คลาส: {', '.join(SESSION.class_names) if SESSION.class_names else '— ยังไม่ได้ตั้ง'}",
        f"เฟรมทั้งหมด: {len(frames)}",
        f"label แล้ว: {done} ภาพ",
    ]
    if not frames:
        parts.append("⚠️ ยังไม่มีเฟรม กลับไปขั้นที่ 2")
    elif done == 0:
        parts.append("⚠️ ยังไม่ได้ label เลย กลับไปขั้นที่ 3")
    else:
        parts.append("พร้อมเทรน ✅")
    return "  •  ".join(parts)


def do_train(epochs, imgsz, base_model):
    best, metrics, summary, plots = train_model(
        frames_dir(),
        labels_dir(),
        dataset_dir(),
        runs_dir(),
        SESSION.class_names,
        epochs,
        imgsz,
        base_model,
    )
    SESSION.model_path = best
    SESSION.train_metrics = metrics
    SESSION.save()
    return summary, (plots or None)


# ---------------------------------------------------------------- ขั้นที่ 5
def step5_status():
    if SESSION.model_path and os.path.exists(SESSION.model_path):
        m = SESSION.train_metrics or {}
        return (
            f"ใช้โมเดลที่เทรนเอง (mAP50 = {m.get('mAP50', 0):.3f})  •  "
            f"คลาส: {', '.join(SESSION.class_names)}"
        )
    return "⚠️ ยังไม่ได้เทรนโมเดล ถ้าตรวจจับตอนนี้จะใช้โมเดลสำเร็จรูป yolov8n (80 คลาสทั่วไป)"


_SRC_UPLOAD = "อัปโหลดวิดีโอใหม่"
_SRC_PHOTOS = "ใช้ชุดภาพถ่ายจากขั้นที่ 1"


def toggle_source(choice):
    return gr.update(visible=choice == _SRC_UPLOAD)


def _save_geo_opts(preset, hfov, merge_dist):
    SESSION.camera_preset = preset or "อ่านจากไฟล์อัตโนมัติ"
    SESSION.manual_hfov = float(hfov or 0)
    SESSION.merge_dist_m = float(merge_dist or 2.0)
    SESSION.save()


def do_detect(choice, uploaded_video, uploaded_srt, conf, iou, cfilter, preset, hfov, merge_dist):
    _save_geo_opts(preset, hfov, merge_dist)

    if choice == _SRC_PHOTOS:
        photos = SESSION.photo_files()
        if not photos:
            raise gr.Error("ยังไม่มีภาพถ่ายจากขั้นที่ 1 — เลือกโหมด \"ชุดภาพถ่าย\" ในขั้นที่ 1 ก่อน")
        gallery, report, geo_summary = process_photos(
            photos, SESSION.model_path, output_dir(), conf, iou, cfilter,
            SESSION.camera_preset, SESSION.manual_hfov, SESSION.merge_dist_m,
        )
        SESSION.geo_summary = geo_summary
        SESSION.save()
        return None, gallery, report

    if choice == _SRC_UPLOAD:
        video = uploaded_video
        if not video:
            raise gr.Error("กรุณาอัปโหลดวิดีโอสำหรับตรวจจับ")
        srt = uploaded_srt if isinstance(uploaded_srt, str) else getattr(uploaded_srt, "name", None)
    else:
        video = SESSION.video_path
        if not video:
            raise gr.Error("ยังไม่มีวิดีโอเดิมจากขั้นที่ 1 — เลือก \"อัปโหลดวิดีโอใหม่\" แทน")
        srt = SESSION.srt_path

    out_video, report, geo_summary = process_video(
        video, SESSION.model_path, output_dir(), conf, iou, cfilter,
        srt_path=srt, camera_preset=SESSION.camera_preset, manual_hfov=SESSION.manual_hfov,
    )
    SESSION.geo_summary = geo_summary
    SESSION.save()
    return out_video, None, report


# ---------------------------------------------------------------- ขั้นที่ 6: แผนที่
def _file_url(path: str) -> str:
    import urllib.parse

    return "/gradio_api/file=" + urllib.parse.quote(os.path.abspath(path))


def build_map_view():
    gs = SESSION.geo_summary
    map_html_path = os.path.join(output_dir(), "map.html")
    if not gs or not gs.get("n_points") or not os.path.exists(map_html_path):
        return (
            "ยังไม่มีข้อมูลพิกัด — กลับไปขั้นที่ 5 แล้วตรวจจับด้วย **ภาพถ่ายที่มี GPS** "
            "หรือ **วิดีโอ + ไฟล์ .SRT** ก่อน",
            "", None, None, None,
        )
    url = _file_url(map_html_path)
    iframe = (
        f'<a href="{url}" target="_blank" style="display:inline-block;margin-bottom:6px">↗ เปิดแผนที่เต็มจอ (แท็บใหม่)</a>'
        f'<iframe src="{url}" '
        'style="width:100%;height:520px;border:1px solid var(--border-color-primary);border-radius:8px">'
        "</iframe>"
    )
    rows = [[k, v] for k, v in sorted(gs["per_class"].items())]
    files = [
        p for key, p in gs["files"].items()
        if key in ("geojson", "kml", "csv", "map_html") and p and os.path.exists(p)
    ]
    bbox = gs.get("bbox")
    status = f"**{gs['n_points']} จุด**"
    if bbox:
        status += (f"  ·  พื้นที่ ~lon {bbox[0]:.5f}–{bbox[2]:.5f}, "
                   f"lat {bbox[1]:.5f}–{bbox[3]:.5f}")
    if gs.get("has_oblique"):
        status = "⚠️ บางภาพกล้องเอียงมาก พิกัดคลาดเคลื่อนสูง\n\n" + status
    scatter = gs["files"].get("scatter")
    return status, iframe, (scatter if scatter and os.path.exists(scatter) else None), rows, files


# ---------------------------------------------------------------- restore / reset
def restore_values():
    """คืนค่าที่บันทึกไว้กลับเข้า component ทั้งหมด (ใช้ตอนโหลดหน้า + ตอนกดล้างข้อมูล)"""
    SESSION.rescan_frames()
    is_photo = SESSION.input_kind == "photos"
    has_video = bool(SESSION.video_path)
    out_video = os.path.join(output_dir(), "output_detected.mp4")
    if is_photo:
        v1 = f"โหมดภาพถ่าย: {len(SESSION.frames)} ภาพ ✅" if SESSION.frames else ""
    else:
        v1 = "มีวิดีโอที่อัปโหลดไว้แล้ว ✅" if has_video else ""
    return (
        SESSION.video_path,                                              # video_input
        v1,                                                              # v1_status
        SESSION.frame_mode,                                              # frame_mode
        SESSION.frame_value,                                             # frame_value
        SESSION.max_frames,                                              # max_frames
        step2_status()[0],                                               # s2_status
        SESSION.frames or None,                                          # frame_gallery
        ", ".join(SESSION.class_names),                                  # class_input
        (f"ตั้งคลาสแล้ว: {', '.join(SESSION.class_names)}" if SESSION.class_names else ""),  # class_status
        refresh_label_ctx(),                                             # label_ctx
        step4_status(),                                                  # s4_status
        _model_summary(),                                                # train_output
        _existing_plots(),                                               # train_plots
        step5_status(),                                                  # s5_status
        out_video if os.path.exists(out_video) else None,                # video_output
    )


def restore_geo_ui():
    """คืนค่า component ใหม่ (ขั้น 1 toggle, พารามิเตอร์กล้อง, แท็บแผนที่)"""
    import glob as _glob

    is_photo = SESSION.input_kind == "photos"
    ann = sorted(_glob.glob(os.path.join(output_dir(), "photos_annotated", "*")))
    s6, iframe, scat, rows, files = build_map_view()
    return (
        ("ชุดภาพถ่าย (มีพิกัด)" if is_photo else "วิดีโอ"),   # input_kind
        gr.update(visible=not is_photo),                       # video_group
        gr.update(visible=is_photo),                           # photos_group
        SESSION.srt_path,                                      # srt_input
        SESSION.camera_preset,                                 # camera_preset_dd
        (SESSION.manual_hfov or None),                         # hfov_num
        SESSION.merge_dist_m,                                  # merge_dist_num
        (ann or None),                                         # photo_gallery_out
        s6, iframe, scat, rows, files,                         # map tab
    )


def do_reset():
    SESSION.reset()
    return (gr.update(visible=False), gr.Tabs(selected=1), *restore_values(), *restore_geo_ui())


# ================================================================ UI
with gr.Blocks(title="คอร์สอบรม: เทรนโมเดลตรวจจับวัตถุ (YOLOv8)") as demo:
    with gr.Row():
        gr.Markdown("# คอร์สอบรม: เทรนและใช้งานโมเดลตรวจจับวัตถุ (YOLOv8)")
        reset_btn = gr.Button("🗑️ ล้างข้อมูล เริ่มใหม่", scale=0, size="sm", variant="stop", min_width=170)
    gr.Markdown(
        "ทำตามลำดับ 5 ขั้นตอน — แต่ละพารามิเตอร์มีปุ่ม **ⓘ** กดดูคำอธิบายได้ "
        "· งานถูกบันทึกอัตโนมัติ ปิด/รีเฟรชแล้วเปิดใหม่ได้",
        elem_classes=["step-hint"],
    )

    # ---- modal อธิบาย parameter (ตัวเดียวใช้ร่วมทั้งแอป) ----
    with gr.Group(visible=False, elem_classes=["info-modal-overlay"]) as info_modal:
        with gr.Column(elem_classes=["info-modal-box"]):
            info_title = gr.Markdown()
            info_body = gr.Markdown()
            close_info = gr.Button("ปิด", variant="primary")
    close_info.click(lambda: gr.update(visible=False), None, info_modal)

    # ---- modal ยืนยันล้างข้อมูล ----
    with gr.Group(visible=False, elem_classes=["info-modal-overlay"]) as reset_modal:
        with gr.Column(elem_classes=["info-modal-box"]):
            gr.Markdown(
                "### ล้างข้อมูลทั้งหมด?\n"
                "วิดีโอ เฟรม label และโมเดลที่เทรนไว้ **จะถูกลบทั้งหมด** เริ่มใหม่ตั้งแต่ต้น"
            )
            with gr.Row():
                reset_yes = gr.Button("ยืนยันล้างข้อมูล", variant="stop")
                reset_no = gr.Button("ยกเลิก", variant="primary")
    reset_btn.click(lambda: gr.update(visible=True), None, reset_modal)
    reset_no.click(lambda: gr.update(visible=False), None, reset_modal)

    def make_info_btn(key: str):
        btn = gr.Button("ⓘ", elem_classes=["info-btn"], scale=0, min_width=30)
        btn.click(
            lambda k=key: (
                gr.update(visible=True),
                f"### {PARAM_HELP[k]['title']}",
                PARAM_HELP[k]["body_md"],
            ),
            None,
            [info_modal, info_title, info_body],
        )
        return btn

    def param_head(title: str, key: str):
        # หัวข้อพารามิเตอร์ + ปุ่ม ⓘ ในแถวเดียว ชิดซ้าย วางไว้เหนือ control
        with gr.Row(elem_classes=["param-head"]):
            gr.Markdown(f"**{title}**")
            make_info_btn(key)

    with gr.Tabs() as tabs:
        # ============================================ ขั้นที่ 1
        with gr.Tab("1. เลือกข้อมูลนำเข้า", id=1) as tab1:
            gr.Markdown("เลือกว่าจะนำเข้าเป็น **วิดีโอ** หรือ **ชุดภาพถ่ายจากโดรน** (ภาพถ่ายมีพิกัด GPS ใช้ทำแผนที่ได้)")
            param_head("ประเภทข้อมูลนำเข้า", "input_kind")
            input_kind = gr.Radio(
                ["วิดีโอ", "ชุดภาพถ่าย (มีพิกัด)"], value="วิดีโอ",
                show_label=False, container=False,
            )
            with gr.Group() as video_group:
                video_input = gr.Video(label="วิดีโอต้นทาง (.mp4)")
                param_head("ไฟล์พิกัด .SRT (ถ้ามี)", "srt_file")
                srt_input = gr.File(
                    label="ไฟล์ subtitle .SRT ที่มาคู่กับวิดีโอ — ใช้คำนวณพิกัดวัตถุในขั้น 6",
                    file_types=[".srt"], file_count="single", type="filepath",
                )
            with gr.Group(visible=False) as photos_group:
                photos_input = gr.File(
                    label="ภาพถ่ายจากโดรน (.jpg ดิบ หลายไฟล์) — จะใช้เป็นเฟรมสำหรับ label/เทรนด้วย",
                    file_count="multiple", file_types=["image"], type="filepath",
                )
            v1_status = gr.Markdown()
            next_1 = gr.Button("ไปขั้นต่อไป ▶", variant="primary")

        # ============================================ ขั้นที่ 2
        with gr.Tab("2. แตกเฟรม", id=2) as tab2:
            gr.Markdown("แบ่งวิดีโอออกเป็นภาพนิ่งหลาย ๆ ภาพ เพื่อเอาไป label ในขั้นถัดไป "
                        "(โหมดภาพถ่าย: ข้ามขั้นนี้ได้เลย)")
            s2_status = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=1):
                    param_head("วิธีเลือกเฟรม", "frame_mode")
                    frame_mode = gr.Radio(
                        ["ทุก X วินาที", "ทุก N เฟรม"], value="ทุก X วินาที", show_label=False, container=False
                    )
                    param_head("ช่วงการเก็บเฟรม (วินาที หรือ จำนวนเฟรม)", "frame_value")
                    frame_value = gr.Number(value=1, show_label=False, container=False)
                    param_head("จำนวนเฟรมสูงสุด", "max_frames")
                    max_frames = gr.Slider(10, 100, value=40, step=5, show_label=False, container=False)
                    extract_btn = gr.Button("แตกเฟรม", variant="primary")
                with gr.Column(scale=2):
                    frame_gallery = gr.Gallery(label="เฟรมที่ได้", columns=5, height=420)
                    s2_result = gr.Markdown()
            next_2 = gr.Button("ไปขั้นต่อไป: Label วัตถุ ▶", variant="primary")

        # ============================================ ขั้นที่ 3
        with gr.Tab("3. Label วัตถุ", id=3) as tab3:
            gr.Markdown(
                "วาดกรอบสี่เหลี่ยมรอบวัตถุในแต่ละภาพ แล้วเลือกชนิด — "
                "ภาพไหนไม่มีวัตถุให้กด \"ภาพนี้ไม่มีวัตถุ\""
            )
            with gr.Row():
                with gr.Column(scale=3):
                    param_head("ชื่อคลาส (คั่นด้วยจุลภาค) — พิมพ์แล้วกด Enter", "class_names")
                    class_input = gr.Textbox(
                        placeholder="เช่น car หรือ car,truck", show_label=False, container=False
                    )
                class_status = gr.Markdown()

            label_ctx = gr.State()

            @gr.render(inputs=[label_ctx])
            def render_labeler(ctx):
                if not ctx or not ctx.get("frames"):
                    gr.Markdown("⚠️ ยังไม่มีเฟรม — กลับไปแตกเฟรมในขั้นที่ 2 ก่อน")
                    return
                if not ctx.get("classes"):
                    gr.Markdown("⚠️ กรอกชื่อคลาสด้านบนแล้วกด Enter ก่อนเริ่ม label")
                    return

                frames = ctx["frames"]
                idx = ctx["idx"] % len(frames)
                frame = frames[idx]
                w, h = image_size(frame)
                txt = label_txt_path(ctx["labels_dir"], frame)
                existing = yolo_to_boxes(txt, w, h, ctx["classes"])
                done = labeled_count(ctx["labels_dir"], frames)

                mark = "  ✅ ภาพนี้ label แล้ว" if os.path.exists(txt) else ""
                gr.Markdown(f"**ภาพที่ {idx + 1}/{len(frames)}** — label แล้ว {done}/{len(frames)} ภาพ{mark}")

                # ปุ่มนำทางอยู่ "เหนือ" รูป จะได้ไม่ต้องเลื่อนจอลงไปกดทุกครั้ง
                with gr.Row():
                    prev_b = gr.Button("◀ ก่อนหน้า", scale=1)
                    save_b = gr.Button("💾 บันทึก label ภาพนี้", variant="primary", scale=3)
                    none_b = gr.Button("ภาพนี้ไม่มีวัตถุ", scale=1)
                    next_b = gr.Button("ถัดไป ▶", scale=1)

                annotator = image_annotator(
                    value={"image": frame, "boxes": existing},
                    label_list=ctx["classes"],
                    use_default_label=len(ctx["classes"]) == 1,
                    image_type="filepath",
                    sources=None,
                    show_label=False,
                    elem_classes=["anno-box"],
                )

                def _boxes(ann_val):
                    if isinstance(ann_val, dict):
                        return ann_val.get("boxes") or []
                    return []

                def _save(ann_val, c):
                    cur = c["idx"] % len(c["frames"])
                    save_label(c["labels_dir"], c["frames"][cur], _boxes(ann_val), c["classes"])
                    return {**c, "idx": (cur + 1) % len(c["frames"]), "tick": c.get("tick", 0) + 1}

                def _save_none(c):
                    cur = c["idx"] % len(c["frames"])
                    save_label(c["labels_dir"], c["frames"][cur], [], c["classes"])
                    return {**c, "idx": (cur + 1) % len(c["frames"]), "tick": c.get("tick", 0) + 1}

                prev_b.click(lambda c: {**c, "idx": (c["idx"] - 1) % len(c["frames"])}, [label_ctx], [label_ctx])
                next_b.click(lambda c: {**c, "idx": (c["idx"] + 1) % len(c["frames"])}, [label_ctx], [label_ctx])
                save_b.click(_save, [annotator, label_ctx], [label_ctx])
                none_b.click(_save_none, [label_ctx], [label_ctx])

            next_3 = gr.Button("ไปขั้นต่อไป: เทรนโมเดล ▶", variant="primary")

        # ============================================ ขั้นที่ 4
        with gr.Tab("4. เทรนโมเดล", id=4) as tab4:
            gr.Markdown("สอนโมเดลให้รู้จักวัตถุจากภาพที่ label ไว้ ยิ่ง label เยอะและครบมุม โมเดลยิ่งแม่น")
            s4_status = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=1):
                    param_head("จำนวน Epoch", "epochs")
                    epoch_slider = gr.Slider(5, 100, value=100, step=1, show_label=False, container=False)
                    param_head("ขนาดภาพที่ใช้เทรน (imgsz)", "imgsz")
                    imgsz_slider = gr.Slider(320, 960, value=640, step=32, show_label=False, container=False)
                    param_head("โมเดลตั้งต้น", "base_model")
                    base_model_dd = gr.Dropdown(
                        list(BASE_MODELS.keys()), value=list(BASE_MODELS.keys())[0],
                        show_label=False, container=False,
                    )
                    train_btn = gr.Button("เริ่มเทรน", variant="primary")
                    train_output = gr.Textbox(label="ผลการเทรน", lines=6)
                with gr.Column(scale=2):
                    train_plots = gr.Gallery(
                        label="กราฟผลการเทรน / confusion matrix", columns=1, height=520
                    )
            next_4 = gr.Button("ไปขั้นต่อไป: ตรวจจับวัตถุ ▶", variant="primary")

        # ============================================ ขั้นที่ 5
        with gr.Tab("5. ตรวจจับวัตถุ", id=5) as tab5:
            gr.Markdown("รันโมเดลกับวิดีโอ/ภาพถ่าย วาดกรอบวัตถุที่เจอ นับจำนวนไม่ซ้ำ "
                        "และ (ถ้ามีพิกัด) คำนวณตำแหน่งวัตถุบนพื้นเพื่อทำแผนที่ในขั้น 6")
            s5_status = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=1):
                    param_head("แหล่งข้อมูลที่จะตรวจจับ", "detect_source")
                    detect_source = gr.Radio(
                        ["ใช้วิดีโอเดิมจากขั้นที่ 1", "อัปโหลดวิดีโอใหม่",
                         "ใช้ชุดภาพถ่ายจากขั้นที่ 1"],
                        value="ใช้วิดีโอเดิมจากขั้นที่ 1",
                        show_label=False,
                        container=False,
                    )
                    detect_video_input = gr.Video(label="อัปโหลดวิดีโอใหม่ (.mp4)", visible=False)
                    detect_srt_input = gr.File(
                        label="ไฟล์ .SRT ของวิดีโอใหม่ (ถ้ามี)", file_types=[".srt"],
                        file_count="single", type="filepath", visible=False,
                    )
                    param_head("Confidence Threshold", "conf")
                    conf_slider = gr.Slider(0.0, 1.0, value=0.25, step=0.05, show_label=False, container=False)
                    param_head("IoU Threshold", "iou")
                    iou_slider = gr.Slider(0.0, 1.0, value=0.45, step=0.05, show_label=False, container=False)
                    param_head("กรองเฉพาะ class (เว้นว่าง = แสดงทุก class)", "class_filter")
                    class_filter_input = gr.Textbox(
                        placeholder="เช่น car หรือ car,truck", show_label=False, container=False
                    )
                    with gr.Group() as geo_opts_group:  # noqa: F841
                        param_head("รุ่นกล้อง / โดรน (สำหรับคำนวณพิกัด)", "camera_model")
                        camera_preset_dd = gr.Dropdown(
                            list(CAMERA_PRESETS.keys()), value="อ่านจากไฟล์อัตโนมัติ",
                            show_label=False, container=False,
                        )
                        param_head("มุมมองภาพแนวนอน HFOV (°) — เว้นว่าง = อ่านจากไฟล์", "hfov")
                        hfov_num = gr.Number(value=None, show_label=False, container=False)
                        param_head("ระยะรวมจุดซ้ำ (เมตร) — โหมดภาพถ่าย", "merge_dist")
                        merge_dist_num = gr.Number(value=2.0, show_label=False, container=False)
                    detect_btn = gr.Button("ประมวลผล", variant="primary", size="lg")
                with gr.Column(scale=2):
                    video_output = gr.Video(label="วิดีโอผลลัพธ์", height=420)
                    photo_gallery_out = gr.Gallery(label="ภาพผลลัพธ์ (โหมดภาพถ่าย)", columns=3, height=420)
                    count_output = gr.Textbox(label="สรุปผลการตรวจจับ", lines=8)

        # ============================================ ขั้นที่ 6
        with gr.Tab("6. แผนที่", id=6) as tab6:
            gr.Markdown("แผนที่ตำแหน่งวัตถุที่ตรวจพบ (จากพิกัดในภาพถ่าย / ไฟล์ .SRT) — "
                        "แผนที่ฐานต้องต่อเน็ต · ไฟล์ GeoJSON/KML/CSV เปิดใน Google Earth / QGIS ได้")
            map_refresh_btn = gr.Button("🔄 สร้าง / รีเฟรชแผนที่จากผลตรวจจับล่าสุด", variant="primary")
            s6_status = gr.Markdown()
            map_html = gr.HTML()
            with gr.Row():
                map_scatter = gr.Image(label="ผังจุด (ใช้ได้แม้ไม่มีเน็ต)", height=380)
                map_table = gr.Dataframe(
                    headers=["class", "จำนวนจุด"], label="สรุปจำนวนต่อ class",
                    interactive=False, wrap=True,
                )
            save_local_btn = gr.Button(
                "💾 บันทึกไฟล์แผนที่ลงเครื่อง (เลือกโฟลเดอร์)…", variant="primary"
            )
            gr.Markdown(
                "กดปุ่มด้านบนแล้วเลือกโฟลเดอร์ปลายทาง — ไฟล์ GeoJSON / KML / CSV / map.html "
                "จะถูกคัดลอกไปที่นั่น · (เปิดผ่านเบราว์เซอร์: ใช้รายการดาวน์โหลดด้านล่าง)",
                elem_classes=["step-hint"],
            )
            map_files = gr.File(
                label="ดาวน์โหลดทีละไฟล์: GeoJSON / KML / CSV / map.html", file_count="multiple"
            )

    # component ที่ restore_values() คืนค่าให้ (ลำดับต้องตรงกับ tuple ใน restore_values)
    RESTORE_OUTPUTS = [
        video_input, v1_status, frame_mode, frame_value, max_frames,
        s2_status, frame_gallery, class_input, class_status, label_ctx,
        s4_status, train_output, train_plots, s5_status, video_output,
    ]
    # component ใหม่ (ลำดับต้องตรงกับ restore_geo_ui)
    GEO_OUTPUTS = [
        input_kind, video_group, photos_group, srt_input,
        camera_preset_dd, hfov_num, merge_dist_num, photo_gallery_out,
        s6_status, map_html, map_scatter, map_table, map_files,
    ]

    # ---------------- เดินสาย event ----------------
    input_kind.change(set_input_kind, [input_kind], [video_group, photos_group]).then(
        step2_status, None, [s2_status, frame_gallery]
    )
    video_input.change(set_video, [video_input], [v1_status])
    srt_input.change(set_srt, [srt_input], [v1_status])
    photos_input.change(set_photos, [photos_input], [v1_status, frame_gallery])
    next_1.click(lambda: gr.Tabs(selected=2), None, tabs).then(
        step2_status, None, [s2_status, frame_gallery]
    )

    extract_btn.click(do_extract, [frame_mode, frame_value, max_frames], [frame_gallery, s2_result])
    next_2.click(lambda: gr.Tabs(selected=3), None, tabs).then(
        step3_enter, None, [class_input, class_status, label_ctx]
    )

    class_input.submit(set_classes, [class_input], [class_status]).then(
        refresh_label_ctx, None, [label_ctx]
    )
    next_3.click(lambda: gr.Tabs(selected=4), None, tabs).then(step4_status, None, [s4_status])

    train_btn.click(do_train, [epoch_slider, imgsz_slider, base_model_dd], [train_output, train_plots])
    next_4.click(lambda: gr.Tabs(selected=5), None, tabs).then(step5_status, None, [s5_status])

    detect_source.change(
        toggle_source, [detect_source], [detect_video_input]
    ).then(
        lambda c: gr.update(visible=c == _SRC_UPLOAD), [detect_source], [detect_srt_input]
    )
    detect_btn.click(
        do_detect,
        [detect_source, detect_video_input, detect_srt_input, conf_slider, iou_slider,
         class_filter_input, camera_preset_dd, hfov_num, merge_dist_num],
        [video_output, photo_gallery_out, count_output],
    ).then(build_map_view, None, [s6_status, map_html, map_scatter, map_table, map_files])

    map_refresh_btn.click(build_map_view, None, [s6_status, map_html, map_scatter, map_table, map_files])

    save_local_btn.click(
        None, None, None,
        js="""async () => {
            const api = window.pywebview && window.pywebview.api;
            let msg;
            if (!api || !api.save_map_outputs) {
                msg = 'ปุ่มนี้ใช้ได้เฉพาะในหน้าต่างโปรแกรม — ถ้าเปิดผ่านเบราว์เซอร์ ให้ใช้รายการดาวน์โหลดด้านล่าง';
            } else {
                try { const r = await api.save_map_outputs(); msg = (r && r.msg) ? r.msg : 'เสร็จ'; }
                catch (e) { msg = 'บันทึกไม่สำเร็จ: ' + e; }
            }
            window.alert(msg);
        }""",
    )

    # เปิดเข้าแต่ละแท็บ = ดึง state ล่าสุดมาแสดง
    tab2.select(step2_status, None, [s2_status, frame_gallery])
    tab3.select(step3_enter, None, [class_input, class_status, label_ctx])
    tab4.select(step4_status, None, [s4_status])
    tab5.select(step5_status, None, [s5_status])
    tab6.select(build_map_view, None, [s6_status, map_html, map_scatter, map_table, map_files])

    # โหลดหน้า = กู้สถานะกลับมาทั้งหมด (รีเฟรช/เปิดใหม่ งานไม่หาย)
    demo.load(restore_values, None, RESTORE_OUTPUTS).then(restore_geo_ui, None, GEO_OUTPUTS)

    # ยืนยันล้างข้อมูล
    reset_yes.click(do_reset, None, [reset_modal, tabs, *RESTORE_OUTPUTS, *GEO_OUTPUTS])


if __name__ == "__main__":
    from core.state import load_session

    load_session()
    demo.queue().launch(
        server_name="0.0.0.0", server_port=6066, css=CUSTOM_CSS, allowed_paths=[WORKDIR]
    )
