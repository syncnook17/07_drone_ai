"""
ตำแหน่งไฟล์ asset ที่ bundle มากับโปรแกรม — ใช้ได้ทั้งตอนรันจาก source และตอน frozen (.exe)
รวมถึงตำแหน่งโฟลเดอร์ทำงาน (yolo_workspace) ที่ต้องอยู่ "ข้าง ๆ ตัวโปรแกรม" เสมอ
"""

import os
import sys
import tempfile

# ตอน PyInstaller frozen: sys._MEIPASS ชี้ไปโฟลเดอร์ที่แตก asset ไว้ (temp, ถูกลบเมื่อปิดโปรแกรม)
# ตอนรันปกติ: ใช้ราก repo (โฟลเดอร์แม่ของ core/)
_ROOT = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resource_path(rel: str) -> str:
    return os.path.join(_ROOT, rel)


ASSETS_DIR = resource_path("assets")
YOLOV8N = resource_path(os.path.join("assets", "yolov8n.pt"))
YOLOV8S = resource_path(os.path.join("assets", "yolov8s.pt"))
ARIAL_TTF = resource_path(os.path.join("assets", "Arial.ttf"))


def app_base_dir() -> str:
    """โฟลเดอร์ที่ 'ตัวโปรแกรม' วางอยู่จริง — สำหรับเก็บงานของผู้เรียนไว้ข้าง ๆ
    frozen (.exe แบบ onefile): โฟลเดอร์ที่ผู้ใช้วางไฟล์ .exe (ไม่ใช่ temp _MEIPASS)
    รันจาก source: โฟลเดอร์ปัจจุบัน (ที่สั่ง python main.py)"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.getcwd()


def _resolve_workspace() -> str:
    base = os.path.join(app_base_dir(), "yolo_workspace")
    try:
        os.makedirs(base, exist_ok=True)
        probe = os.path.join(base, ".wtest")
        with open(probe, "w"):
            pass
        os.remove(probe)
        return base
    except OSError:
        # เผลอวางไฟล์ไว้ในที่เขียนไม่ได้ (เช่น Program Files) → ถอยไปใช้ LOCALAPPDATA
        alt = os.path.join(
            os.environ.get("LOCALAPPDATA") or tempfile.gettempdir(),
            "DroneCVEdu",
            "yolo_workspace",
        )
        os.makedirs(alt, exist_ok=True)
        return alt


WORKSPACE_DIR = _resolve_workspace()
