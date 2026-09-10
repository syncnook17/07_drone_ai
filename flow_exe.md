# flow: เอาโค้ดไป Windows → รันได้ → แพ็กเป็น .exe แจกออฟไลน์

> เป้าหมาย: โปรแกรม `DroneCVEdu` (โฟลเดอร์เดียว มี `.exe` ข้างใน) ที่ก๊อปไปเครื่องผู้เรียน
> **เปิดใช้ได้ทันทีโดยไม่ต้องต่อเน็ต / ไม่ต้องลง Python**

ทุกขั้นตอนทำ **บนเครื่อง Windows** (PyInstaller ข้าม OS ไม่ได้)

---

## 0. ภาพรวม flow

```
[เครื่อง dev นี้: Linux]                    [เครื่อง Windows ของคุณ]                 [เครื่องผู้เรียน 60+]
  โค้ด + assets/  ──ก๊อปทั้งโฟลเดอร์──►  1. ลง Python 3.10 + venv
                                        2. pip install (torch CPU)
                                        3. python main.py  ← ทดสอบ
                                        4. สร้าง build/ (spec+hook)
                                        5. pyinstaller  ──►  dist\DroneCVEdu\  ──zip──►  แตก zip
                                        6. ทดสอบ .exe ตัดเน็ต                              ดับเบิลคลิก .exe
```

---

## 1. เตรียมเครื่อง Windows (ครั้งเดียว)

| ต้องมี | หมายเหตุ |
|---|---|
| **Python 3.10 (64-bit)** | ห้าม 3.11+ (เผื่อ pyhula ในอนาคต) — ดาวน์โหลดจาก python.org ติ๊ก "Add to PATH" |
| **Edge WebView2 Runtime** | Win11 มักมีแล้ว · Win10 บางเครื่องไม่มี → โหลด "Evergreen Standalone Installer" จาก Microsoft เก็บไว้แจกด้วย |
| เน็ต | ใช้เฉพาะตอน `pip install` — หลังจากนั้นตัดได้ |

---

## 2. ก๊อปโปรเจกต์มา

ก๊อป `07_ai_reader/` **ทั้งโฟลเดอร์** มา ตรวจว่ามีครบ:

```
07_ai_reader/
  app.py  main.py
  core/            (state.py video.py labeling.py training.py detect.py params_help.py paths.py)
  assets/          ← สำคัญ: yolov8n.pt (6.5MB), yolov8s.pt (22MB), Arial.ttf
  requirements-win.txt
```

> ไม่ต้องเอา `venv/` , `yolo_workspace/` , `app.log` มา

---

## 3. สร้าง venv + ติดตั้ง library

เปิด **PowerShell** ในโฟลเดอร์โปรเจกต์:

```powershell
py -3.10 -m venv venv
venv\Scripts\activate

# torch เวอร์ชัน CPU ก่อน (เครื่องผู้เรียนอาจไม่มีการ์ดจอ) — ตัวนี้ทำให้ .exe เล็กลงเยอะ
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements-win.txt
```

---

## 4. ทดสอบว่ารันได้ ก่อนแพ็ก (สำคัญมาก)

```powershell
python main.py
```

ต้องได้:
- หน้าต่างโปรแกรมเปิดขึ้น (ถ้าไม่ขึ้น = ไม่มี WebView2 → มันจะเปิดใน browser ให้แทน ถือว่ายังโอเค)
- เดินครบ 5 ขั้น: อัปวิดีโอ → แตกเฟรม → label 5-10 ภาพ → **เทรน 15 epoch จนได้ผล** → detect ได้วิดีโอมีกรอบ
- ปิดโปรแกรม เปิด `python main.py` ใหม่ → งานเดิมยังอยู่

ถ้าขั้นนี้ไม่ผ่าน **อย่าเพิ่งไปแพ็ก** — แก้ให้ผ่านก่อน

---

## 5. สร้างไฟล์ build

สร้างโฟลเดอร์ `build/` แล้วสร้าง 3 ไฟล์นี้:

### 5.1 `build/runtime_hook.py`

```python
# รันก่อนโมดูลใด ๆ ถูก import ตอนเป็น .exe
import os, sys

# console=False -> sys.stdout/stderr เป็น None -> uvicorn ทำ .isatty() แล้วพัง
# ("Unable to configure formatter 'default'") จึงต้อง shim ก่อนทุกอย่าง + เก็บ log ไว้ดีบั๊ก
if getattr(sys, "frozen", False) and (sys.stdout is None or sys.stderr is None):
    try:
        _log = open(os.path.join(os.path.dirname(sys.executable), "DroneCVEdu.log"),
                    "w", encoding="utf-8", buffering=1)
    except OSError:
        _log = open(os.devnull, "w")
    if sys.stdout is None: sys.stdout = _log
    if sys.stderr is None: sys.stderr = _log

# บังคับโหมดออฟไลน์
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
```

### 5.2 `build/DroneCVEdu.spec`  (onefile — ไฟล์เดียว)

> - path ทุกตัวอ้าง `SPECPATH` เสมอ — รัน `pyinstaller` จากโฟลเดอร์ไหนก็ได้
> - **onefile**: ใส่ `a.binaries` + `a.datas` ลง `EXE()` โดยตรง ไม่มี `COLLECT` → ได้ `dist\DroneCVEdu.exe` ไฟล์เดียว
> - `torchvision 0.29` เก็บ native ops ใน `_C_stable.pyd` (โหลดผ่าน `torch.ops.load_library` ไม่ใช่ `import`)
>   PyInstaller เก็บไม่ครบ → error `operator torchvision::nms does not exist` → ต้อง copy `.pyd/.dll` เอง
> - `Splash()` = จอ "กำลังเปิดโปรแกรม..." ระหว่าง onefile แตกไฟล์ (~20–40 วิครั้งแรก)

```python
# -*- mode: python ; coding: utf-8 -*-
import glob, os
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

_HERE = os.path.abspath(SPECPATH)          # โฟลเดอร์ build/
_ROOT = os.path.dirname(_HERE)             # โฟลเดอร์โปรเจกต์

datas, binaries, hiddenimports = [], [], []
for pkg in ("gradio", "gradio_client", "safehttpx", "groovy",
            "gradio_image_annotation", "ultralytics"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

datas += collect_data_files("ultralytics")
datas += [(os.path.join(_ROOT, "assets"), "assets")]   # << bundle โมเดล + ฟอนต์
hiddenimports += ["cv2", "imageio_ffmpeg", "psutil"]

import torchvision as _tv                              # torchvision native libs
_tvd = os.path.dirname(_tv.__file__)
for _f in glob.glob(_tvd + "/*.pyd") + glob.glob(_tvd + "/*.dll"):
    binaries += [(_f, "torchvision")]
binaries += collect_dynamic_libs("torchvision")
hiddenimports += ["torchvision", "torchvision.ops", "torchvision._meta_registrations"]

a = Analysis(
    [os.path.join(_ROOT, "main.py")],
    pathex=[_ROOT], binaries=binaries, datas=datas, hiddenimports=hiddenimports,
    hookspath=[], runtime_hooks=[os.path.join(_HERE, "runtime_hook.py")],
    excludes=[], noarchive=False,
)
pyz = PYZ(a.pure)

splash = Splash(os.path.join(_HERE, "splash.png"), binaries=a.binaries, datas=a.datas,
                text_pos=(12, 272), text_size=9, text_color="white")

exe = EXE(
    pyz, a.scripts, splash, splash.binaries, a.binaries, a.datas, [],
    name="DroneCVEdu",
    console=False,          # ไม่มีหน้าต่าง cmd ดำ ๆ (เปลี่ยนเป็น True ตอนดีบั๊ก)
    icon=None, upx=False,
)
```

`build/splash.png` = รูป 480×300 ทำเองก็ได้ (พื้นเข้ม ข้อความ "กำลังเปิดโปรแกรม ครั้งแรกใช้เวลาสักครู่…")

### 5.3 `build/build.bat`

```bat
@echo off
cd /d %~dp0\..
call venv\Scripts\activate
pyinstaller build\DroneCVEdu.spec --noconfirm --clean
echo.
echo เสร็จ - ผลอยู่ที่ dist\DroneCVEdu.exe  (ไฟล์เดียว ดับเบิลคลิกเปิดได้เลย)
pause
```

---

## 6. แพ็ก

```powershell
build\build.bat
```

ใช้เวลา ~5-15 นาที ได้ **`dist\DroneCVEdu.exe`** ไฟล์เดียว (~380 MB · torch CPU)

---

## 7. ทดสอบ .exe (ตัดเน็ต)

**7.1 self-test อัตโนมัติ (ไม่ต้องคลิก GUI)** — เดิน extract→label→train→detect ให้ครบ:
```powershell
dist\DroneCVEdu.exe --selftest "C:\path\to\video.mp4"
```
ต้องได้ `SELFTEST PASS` + exit code 0 + มี `yolo_workspace\output\output_detected.mp4` ข้าง ๆ .exe

**7.2 ทดสอบ GUI จริง (ปิด WiFi/LAN):**
1. ดับเบิลคลิก `DroneCVEdu.exe` → เห็นจอ splash → หน้าต่างโปรแกรมเปิด (ไม่มี dialog error)
2. เดินครบ 5 ขั้น — **ต้องเทรน + detect ได้ ไม่มี error ดาวน์โหลด**
3. ปิดแล้วเปิดใหม่ → งานเดิมยังอยู่
4. ข้อมูลงานอยู่ที่ `yolo_workspace\` ข้าง ๆ .exe

ถ้าเปิดไม่ขึ้น → ดู `DroneCVEdu.log` (ข้าง ๆ .exe) หรือ set `console=True` ใน spec แล้ว build ใหม่

---

## 8. แจกจ่าย

- ส่ง **`DroneCVEdu.exe` ไฟล์เดียว** + `วิธีเปิดโปรแกรม DroneCVEdu.txt` ให้ผู้เรียน (ไม่ต้อง zip / ไม่ต้องแตกไฟล์)
- ผู้เรียนวางไฟล์ที่ Desktop/Documents (**อย่าไว้ใน Program Files**) → ดับเบิลคลิก
- ครั้งแรกเจอ SmartScreen "Windows protected your PC" → **More info → Run anyway** (แก้ถาวรต้องซื้อ code-signing cert)
- ครั้งแรกช้า ~20–40 วิ (แตกไฟล์ลง temp) เป็นปกติ · ต้องมีที่ว่างใน `%TEMP%` ~1 GB
- บาง Antivirus อาจกัก quarantine ไฟล์ onefile → ผู้เรียนกด Allow/Restore
- ปุ่ม "🗑️ ล้างข้อมูล เริ่มใหม่" ในแอป = เริ่มงานรอบใหม่

---

## 8.1 โหมดพิกัด / แผนที่ (ขั้น 1, 5, 6)

- **ต้องใช้ไฟล์ดิบจากโดรน** — อย่าตัดต่อ/ย่อ/ส่งผ่าน LINE ก่อน ไม่งั้นพิกัดหาย
  - **ภาพนิ่ง**: ไฟล์ `.JPG` ตรงจากการ์ด (มี EXIF GPS + XMP `drone-dji`)
  - **วิดีโอ**: ต้องมีไฟล์ `.SRT` ชื่อเดียวกัน → เปิดในแอปโดรนก่อนบิน:
    Camera → ตั้งค่า → **Video Caption / Subtitles = เปิด**
    (ไฟล์ `.LRF` = วิดีโอ proxy ไม่ใช่พิกัด · telemetry ที่ฝังในไฟล์ MP4 ของ DJI รุ่นใหม่
     เช่น "Lito X1" เป็น protobuf อ่านไม่ได้ ต้องใช้ `.SRT`)
  - SRT บางรุ่นไม่มีมุม gimbal → โปรแกรมเดา "ทิศหัวภาพ" จากทิศที่โดรนบิน + สมมติกล้องถ่ายดิ่ง
- แท็บ **"6. แผนที่" ต้องต่อเน็ต** (โหลดแผนที่ฐาน OpenStreetMap) — ถ้าไม่มีเน็ต ยังได้ผังจุด + ไฟล์ GeoJSON/KML/CSV
- ความแม่น **ระดับกลาง**: สมมติกล้องถ่ายดิ่ง (nadir) + ใช้ความสูง `rel_alt` คำนวณสเกล
  พื้นที่ลาดชัน / อาคารสูง / กล้องเอียงมาก จะคลาดเคลื่อน — แอปจะเตือนถ้ากล้องเอียงเกิน 20°
- ทดสอบเร็ว (ในเครื่อง dev): `venv\Scripts\python scripts\make_geo_photos.py sample_photos 9`
  แล้วอัปโฟลเดอร์นั้นในโหมด "ชุดภาพถ่าย"
- self-test ครอบพิกัดด้วยแล้ว (สร้าง `.SRT` สังเคราะห์ให้อัตโนมัติ) — `SELFTEST PASS` = geo pipeline ผ่าน

---

## 9. ปัญหาที่เจอบ่อย

| อาการ | แก้ |
|---|---|
| `.exe` เปิดแล้วปิดทันที | build ด้วย `console=True` ดู error จริง / ดู `DroneCVEdu.log` |
| `ValueError: Unable to configure formatter 'default'` (`NoneType ... isatty`) | `console=False` ทำ `sys.stdout=None` → uvicorn พัง — เช็ก stdout shim ใน `runtime_hook.py` (§5.1) |
| `operator torchvision::nms does not exist` (ตอนเทรน) | `_C_stable.pyd` ของ torchvision ไม่ถูก bundle — เช็กบล็อก `torchvision` ใน spec (§5.2) |
| **"Application Control policy has blocked this file" / "Smart App Control"** | Win11 ที่ SAC = on/evaluation บล็อก `.exe` ที่ไม่ได้เซ็น → Settings → Windows Security → App & browser control → **Smart App Control → Off** (ปิดแล้วเปิดกลับไม่ได้) · ทางแก้ถาวร = code-signing cert + `signtool sign` |
| `Cannot find empty port 6066` | `main.py` เวอร์ชันนี้เลือกพอร์ตว่างอัตโนมัติแล้ว — ถ้ายังเจอ แปลว่าใช้ `main.py` เก่า |
| หน้าต่างขาว / ไม่ขึ้น | ลง Edge WebView2 Runtime (ถ้าไม่มี จะ fallback เปิดใน browser ให้เอง) |
| เซฟงานไม่ได้ / permission denied | อย่าวาง `.exe` ใน Program Files — ย้ายไป Desktop/Documents (จะ fallback ไป `%LOCALAPPDATA%\DroneCVEdu` ให้) |
| แท็บ 6 แผนที่ว่าง / ไม่มีหมุด | (1) ไฟล์ไม่มีพิกัด — ใช้ .JPG ดิบ หรือวิดีโอ+`.SRT` (2) ไม่มีเน็ต — ยังได้ผังจุด+ไฟล์ GeoJSON/KML (3) `.SRT` ไม่มี `rel_alt` |
| หมุดในแผนที่เพี้ยน/เลื่อน | ระบุ HFOV / รุ่นกล้องผิด (ขั้น 5) หรือกล้องไม่ได้ถ่ายดิ่ง — ลองเลือกรุ่นโดรน หรือกรอก HFOV จริง |
| `operator torchvision::nms` (หลังเพิ่มฟีเจอร์) | ยังเป็นเรื่อง torchvision เดิม — บล็อก `torchvision` ใน spec (§5.2) ต้องอยู่ครบ |
| หน้า label ว่างเปล่า | `gradio_image_annotation` collect ไม่ครบ — เช็ก `collect_all` ใน spec |
| `safehttpx` / `groovy` error หา `version.txt` | เพิ่ม `collect_data_files("safehttpx")` , `collect_data_files("groovy")` ใน spec |
| เทรนแล้ว error `ConnectionError` | `HF_HUB_OFFLINE` ไม่ติด — เช็ก `runtime_hooks` ใน spec ว่าชี้ `runtime_hook.py` ถูก |
| .exe ใหญ่มาก (>4GB) | ยืนยันว่าลง `torch` CPU (`pip show torch` ต้องไม่มี `+cu...`) |

---

## 10. โครงสร้างสุดท้ายที่แจก

แจกแค่ 2 ไฟล์:
```
DroneCVEdu.exe                      ← ไฟล์เดียว (library + assets + โมเดล ฝังอยู่ในนี้หมด) ดับเบิลคลิก
วิธีเปิดโปรแกรม DroneCVEdu.txt       ← คู่มือผู้เรียน (SmartScreen / วางไฟล์ที่ไหน)
```

ตอนรันครั้งแรก โปรแกรมสร้างข้าง ๆ `.exe` เอง:
```
yolo_workspace\   ← งานของผู้เรียน (วิดีโอ/เฟรม/label/โมเดล + session.json)
DroneCVEdu.log    ← log ไว้ดีบั๊กถ้ามีปัญหา
```
