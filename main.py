"""
Entry point สำหรับใช้งานจริง / แพ็กเป็น .exe

- ตั้ง env ให้รันแบบออฟไลน์เต็มที่ (ก่อน import gradio / ultralytics)
- เปิด Gradio ใน thread แล้วครอบด้วยหน้าต่าง native (pywebview)
- ถ้าไม่มี pywebview (เช่นบนเครื่อง dev Linux) → fallback เปิดใน browser
"""

import os
import shutil
import socket
import sys
import threading
import time

# ---- ต้องตั้งก่อน import ใด ๆ ที่แตะเน็ต ----
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

from core.paths import ARIAL_TTF, WORKSPACE_DIR as WORKDIR  # noqa: E402

_ULTRA_CFG = os.path.join(WORKDIR, ".ultralytics")
os.makedirs(_ULTRA_CFG, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", _ULTRA_CFG)

# วางฟอนต์ให้ ultralytics ใช้วาดกรอบ (กันไปโหลดจากเน็ต)
_arial_dst = os.path.join(_ULTRA_CFG, "Arial.ttf")
if os.path.exists(ARIAL_TTF) and not os.path.exists(_arial_dst):
    try:
        shutil.copy(ARIAL_TTF, _arial_dst)
    except OSError:
        pass

try:
    from ultralytics import settings as _ultra_settings

    _ultra_settings.update({"sync": False})
except Exception:
    pass

from app import CUSTOM_CSS, demo  # noqa: E402
from core.state import load_session  # noqa: E402

HOST = "127.0.0.1"
PREFERRED_PORT = 6066


def _pick_port(preferred=PREFERRED_PORT):
    # ใช้ 6066 ถ้าว่าง — ถ้าไม่ว่าง (เช่นเครื่องมีโปรแกรมอื่นจองไว้) ให้ OS หาพอร์ตว่างแทน
    with socket.socket() as s:
        if s.connect_ex((HOST, preferred)) != 0:
            return preferred
    with socket.socket() as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def _wait_until_up(port, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            if s.connect_ex((HOST, port)) == 0:
                return True
        time.sleep(0.2)
    return False


def _close_splash():
    # ปิดจอ splash ของ PyInstaller onefile (ไม่มีตอนรันจาก source → เงียบไว้)
    try:
        import pyi_splash

        pyi_splash.close()
    except Exception:
        pass


def main():
    if "--selftest" in sys.argv:
        from core.selftest import run

        raise SystemExit(run(sys.argv))

    load_session()

    port = _pick_port()
    url = f"http://{HOST}:{port}"

    demo.queue(api_open=False).launch(
        server_name=HOST,
        server_port=port,
        css=CUSTOM_CSS,
        allowed_paths=[WORKDIR],
        prevent_thread_lock=True,
        quiet=True,
        inbrowser=False,
        share=False,
    )
    _wait_until_up(port)
    _close_splash()

    try:
        import webview

        webview.create_window("คอร์สอบรม CV — YOLOv8", url, width=1280, height=880)
        webview.start()
    except Exception:
        # ไม่มี pywebview / WebView2 → เปิดใน browser แทน แล้วค้าง process ไว้
        import webbrowser

        webbrowser.open(url)
        print(f"เปิดใช้งานที่ {url}  (กด Ctrl+C เพื่อปิด)")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
