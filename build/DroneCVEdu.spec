# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

# แก้ path ให้อ้างอิงตำแหน่ง .spec เสมอ ไม่ว่าจะรัน pyinstaller จาก cwd ไหน
_HERE = os.path.abspath(SPECPATH)
_ROOT = os.path.dirname(_HERE)

datas, binaries, hiddenimports = [], [], []
for pkg in ("gradio", "gradio_client", "safehttpx", "groovy",
            "gradio_image_annotation", "ultralytics"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

datas += collect_data_files("ultralytics")
datas += [(os.path.join(_ROOT, "assets"), "assets")]   # << bundle โมเดล + ฟอนต์
hiddenimports += ["cv2", "imageio_ffmpeg", "psutil",
                  "PIL.ExifTags", "PIL.Image",           # อ่าน EXIF GPS ของภาพโดรน
                  "core.geo", "core.geomap", "core.detect_photos", "core.selftest"]

# torchvision 0.29: native ops อยู่ใน _C_stable.pyd / image_stable.pyd และถูกโหลดผ่าน
# torch.ops.load_library (ไม่ใช่ import) → PyInstaller เก็บไม่ครบ → "operator torchvision::nms does not exist"
# ต้อง copy ทั้ง .pyd และ .dll ใต้ torchvision/ เข้ามาเองให้ลงที่โฟลเดอร์ torchvision/
import glob as _glob
import torchvision as _tv

_tv_dir = os.path.dirname(_tv.__file__)
for _f in _glob.glob(os.path.join(_tv_dir, "*.pyd")) + _glob.glob(os.path.join(_tv_dir, "*.dll")):
    binaries += [(_f, "torchvision")]
binaries += collect_dynamic_libs("torchvision")
hiddenimports += ["torchvision", "torchvision.ops", "torchvision._meta_registrations"]

a = Analysis(
    [os.path.join(_ROOT, "main.py")],
    pathex=[_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[os.path.join(_HERE, "runtime_hook.py")],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

splash = Splash(
    os.path.join(_HERE, "splash.png"),
    binaries=a.binaries,
    datas=a.datas,
    # ไม่ใส่ text_pos: ข้อความ progress ของ PyInstaller ใช้ฟอนต์ที่ไม่มีสระไทย (โชว์เป็นสี่เหลี่ยม)
    # ข้อความทั้งหมด baked ลงในรูป splash.png ด้วยฟอนต์ Tahoma แล้ว
)

# onefile: รวม binaries + datas ไว้ใน EXE เดียว (ไม่มี COLLECT)
exe = EXE(
    pyz,
    a.scripts,
    splash,
    splash.binaries,
    a.binaries,
    a.datas,
    [],
    name="DroneCVEdu",
    console=False,          # ไม่มีหน้าต่าง cmd ดำ ๆ (เปลี่ยนเป็น True ตอนดีบั๊ก)
    icon=None,
    upx=False,
)
