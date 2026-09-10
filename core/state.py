"""
State ของโปรแกรม — โปรแกรมนี้เป็น desktop app ผู้ใช้คนเดียวต่อเครื่อง
จึงใช้ SESSION ตัวเดียวระดับโมดูล (ไม่มี session_hash) และ persist ลงดิสก์
เพื่อให้รีเฟรชหน้า / ปิดเปิดโปรแกรมใหม่ แล้วงานยังอยู่ครบ
"""

import glob
import json
import os
import shutil
import stat
import time
from dataclasses import dataclass, field

from core.paths import WORKSPACE_DIR as WORKDIR

SESSION_FILE = os.path.join(WORKDIR, "session.json")

_SUBDIRS = ("uploads", "frames", "labels", "dataset", "runs", "model", "output")


def robust_rmtree(path: str, retries: int = 5) -> None:
    """ลบโฟลเดอร์แบบทนต่อ Windows: ปลด read-only + retry (ไฟล์ถูกล็อกชั่วคราวจาก
    Explorer / Antivirus / รอบเทรนก่อนหน้า) — ถ้าลบไม่ได้จริง ๆ ย้ายทิ้งไปที่อื่นแทน"""
    if not os.path.exists(path):
        return

    def _onerror(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass

    for i in range(retries):
        try:
            shutil.rmtree(path, onerror=_onerror)
        except OSError:
            pass
        if not os.path.exists(path):
            return
        time.sleep(0.4 * (i + 1))

    # ยังลบไม่ได้ → เปลี่ยนชื่อออกไปให้พ้นทาง แล้วพยายามลบทีหลัง
    try:
        trash = f"{path}.old_{int(time.time())}"
        os.rename(path, trash)
        shutil.rmtree(trash, ignore_errors=True)
    except OSError:
        pass


def _d(name: str) -> str:
    return os.path.join(WORKDIR, name)


def uploads_dir() -> str:
    return _d("uploads")


def photos_dir() -> str:
    return os.path.join(uploads_dir(), "photos")


def photos_annotated_dir() -> str:
    return os.path.join(output_dir(), "photos_annotated")


def frames_dir() -> str:
    return _d("frames")


def labels_dir() -> str:
    return _d("labels")


def dataset_dir() -> str:
    return _d("dataset")


def runs_dir() -> str:
    return _d("runs")


def model_dir() -> str:
    return _d("model")


def output_dir() -> str:
    return _d("output")


def ensure_dirs() -> None:
    for name in _SUBDIRS:
        os.makedirs(_d(name), exist_ok=True)


@dataclass
class SessionState:
    video_path: str | None = None
    class_names: list[str] = field(default_factory=list)
    model_path: str | None = None
    train_metrics: dict | None = None
    # ค่าพารามิเตอร์แตกเฟรมล่าสุด (จำไว้ให้ผู้เรียนไม่ต้องตั้งซ้ำ)
    frame_mode: str = "ทุก X วินาที"
    frame_value: float = 1.0
    max_frames: int = 40
    # ---- โหมดข้อมูลนำเข้า + พิกัด/แผนที่ ----
    input_kind: str = "video"              # "video" | "photos"
    srt_path: str | None = None            # ไฟล์ .SRT ที่มาคู่วิดีโอ (พิกัดต่อเฟรม)
    camera_preset: str = "อ่านจากไฟล์อัตโนมัติ"
    manual_hfov: float = 0.0               # 0 = ไม่กำหนดเอง
    merge_dist_m: float = 2.0              # ระยะรวมจุดซ้ำ (โหมดภาพถ่าย)
    geo_summary: dict | None = None        # ผลจาก geomap.export_all()
    # runtime เท่านั้น ไม่ต้อง persist
    frames: list[str] = field(default_factory=list, metadata={"transient": True})

    # ---- persistence ----
    _PERSIST = (
        "video_path",
        "class_names",
        "model_path",
        "train_metrics",
        "frame_mode",
        "frame_value",
        "max_frames",
        "input_kind",
        "srt_path",
        "camera_preset",
        "manual_hfov",
        "merge_dist_m",
        "geo_summary",
    )

    def save(self) -> None:
        ensure_dirs()
        data = {k: getattr(self, k) for k in self._PERSIST}
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        if os.path.exists(SESSION_FILE):
            try:
                with open(SESSION_FILE, encoding="utf-8") as f:
                    data = json.load(f)
                for k in self._PERSIST:
                    if k in data:
                        setattr(self, k, data[k])
            except (json.JSONDecodeError, OSError):
                pass
        # ตรวจสอบไฟล์บนดิสก์จริง
        if self.video_path and not os.path.exists(self.video_path):
            self.video_path = None
        if self.srt_path and not os.path.exists(self.srt_path):
            self.srt_path = None
        if self.model_path and not os.path.exists(self.model_path):
            self.model_path = None
            self.train_metrics = None
        self.rescan_frames()

    def rescan_frames(self) -> None:
        found = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            found += glob.glob(os.path.join(frames_dir(), ext))
        self.frames = sorted(set(found))

    def photo_files(self) -> list[str]:
        found = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            found += glob.glob(os.path.join(photos_dir(), ext))
        return sorted(set(found))

    def reset(self) -> None:
        for name in _SUBDIRS:
            robust_rmtree(_d(name))
        if os.path.exists(SESSION_FILE):
            os.remove(SESSION_FILE)
        self.video_path = None
        self.class_names = []
        self.model_path = None
        self.train_metrics = None
        self.frame_mode = "ทุก X วินาที"
        self.frame_value = 1.0
        self.max_frames = 40
        self.input_kind = "video"
        self.srt_path = None
        self.camera_preset = "อ่านจากไฟล์อัตโนมัติ"
        self.manual_hfov = 0.0
        self.merge_dist_m = 2.0
        self.geo_summary = None
        self.frames = []
        ensure_dirs()


SESSION = SessionState()


def load_session() -> None:
    ensure_dirs()
    SESSION.load()
