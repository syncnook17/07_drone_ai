# รันก่อนโมดูลใด ๆ ถูก import ตอนเป็น .exe

import os
import sys

# --- แก้ crash ตอน console=False: sys.stdout/stderr เป็น None ---
# uvicorn (ที่ gradio เรียก) ทำ sys.stdout.isatty() -> AttributeError -> "Unable to configure formatter 'default'"
if getattr(sys, "frozen", False) and (sys.stdout is None or sys.stderr is None):
    try:
        _log = open(
            os.path.join(os.path.dirname(sys.executable), "DroneCVEdu.log"),
            "w",
            encoding="utf-8",
            buffering=1,
        )
    except OSError:
        _log = open(os.devnull, "w")
    if sys.stdout is None:
        sys.stdout = _log
    if sys.stderr is None:
        sys.stderr = _log

# --- บังคับโหมดออฟไลน์ ---
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
