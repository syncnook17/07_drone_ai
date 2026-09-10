# DroneCVEdu — คอร์สอบรม CV: เทรนโมเดลตรวจจับวัตถุ (YOLOv8)

เว็บแอป (Gradio) สำหรับสอนการเทรนและใช้งานโมเดลตรวจจับวัตถุด้วย YOLOv8
ทั้ง flow: อัปวิดีโอ/ภาพถ่ายโดรน → แตกเฟรม → label → เทรน → ตรวจจับ → **แผนที่ตำแหน่งวัตถุ (จาก EXIF GPS / ไฟล์ .SRT)**

สุดท้ายแพ็กเป็นไฟล์ `.exe` เดียว แจกให้ผู้เรียนใช้แบบออฟไลน์ (ไม่ต้องลง Python)

## รันจาก source

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
python main.py                   # เปิดหน้าต่างแอป (fallback เป็น browser ถ้าไม่มี pywebview)
```

## แพ็กเป็น .exe (Windows + Python 3.10)

ดูขั้นตอนละเอียดใน [`flow_exe.md`](flow_exe.md)

```powershell
py -3.10 -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-win.txt
build\build.bat                  # -> dist\DroneCVEdu.exe (onefile)
```

ทดสอบ .exe แบบไม่ต้องคลิก GUI:

```powershell
dist\DroneCVEdu.exe --selftest "path\to\drone_video.mp4"
```

## โครงสร้าง

| ไฟล์ / โฟลเดอร์ | หน้าที่ |
|---|---|
| `main.py` | entry point (ตั้ง env ออฟไลน์ + ครอบ Gradio ด้วยหน้าต่าง native) |
| `app.py` | UI 6 ขั้นตอน + การเดินสาย event |
| `core/video.py` `labeling.py` `training.py` `detect.py` `detect_photos.py` | ตรรกะแต่ละขั้น |
| `core/geo.py` `geomap.py` | อ่านพิกัด EXIF/XMP/SRT → ฉายพิกเซลเป็น lat/lon → GeoJSON/KML + แผนที่ Leaflet |
| `core/state.py` | session เดียวต่อเครื่อง + persist ลงดิสก์ |
| `core/params_help.py` | เนื้อหาปุ่ม ⓘ อธิบายแต่ละพารามิเตอร์ |
| `assets/` | `yolov8n.pt`, `yolov8s.pt`, `Arial.ttf` (bundle เข้า .exe) |
| `build/` | PyInstaller spec + runtime hook + splash |
| `scripts/make_geo_photos.py` | สร้างภาพทดสอบที่มี EXIF GPS (dev, ต้องมี piexif) |

## หมายเหตุ

- ไฟล์ build (`dist/`, `build/_work/`), `venv/`, `yolo_workspace/` และไฟล์ดิบจากโดรน (`picture_raw/`, `video_raw/`) ไม่เข้า repo — สร้าง/ใส่เองในเครื่อง
- แท็บ "แผนที่" ต้องต่ออินเทอร์เน็ต (โหลดแผนที่ฐาน OpenStreetMap) · ถ้าไม่มีเน็ตยังได้ไฟล์ GeoJSON/KML/CSV
