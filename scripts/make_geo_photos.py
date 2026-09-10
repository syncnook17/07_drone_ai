"""
สร้างภาพ .JPG สังเคราะห์ที่มี EXIF GPS + XMP drone-dji (RelativeAltitude/Gimbal)
ไว้ทดสอบโหมด "ชุดภาพถ่าย" ในขั้นที่ 1/5/6 โดยไม่ต้องมีโดรนจริง

ต้องมี piexif:  venv\Scripts\python -m pip install piexif
รัน:            venv\Scripts\python scripts\make_geo_photos.py [โฟลเดอร์ปลายทาง] [จำนวนภาพ]
"""

import os
import sys

import numpy as np
import piexif
from PIL import Image

OUT = sys.argv[1] if len(sys.argv) > 1 else "sample_photos"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 9
LAT0, LON0, REL_ALT = 13.736700, 100.523400, 60.0

_XMP = (
    '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
    '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
    'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    '<rdf:Description xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" '
    'drone-dji:RelativeAltitude="+{rel:.2f}" drone-dji:GimbalYawDegree="+0.00" '
    'drone-dji:GimbalPitchDegree="-90.00" drone-dji:FlightYawDegree="+0.00"/>'
    '</rdf:RDF></x:xmpmeta><?xpacket end="w"?>'
)


def _dms(deg: float):
    d = int(deg)
    m = int((deg - d) * 60)
    s = round((deg - d - m / 60) * 3600 * 10000)
    return ((d, 1), (m, 1), (s, 10000))


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    side = int(N ** 0.5) or 1
    for i in range(N):
        lat = LAT0 + (i // side) * 0.00012
        lon = LON0 + (i % side) * 0.00012
        a = (np.random.rand(1200, 1600, 3) * 120 + 60).astype("uint8")
        a[520:700, 720:940] = (25, 25, 25)          # วัตถุเข้ม ๆ กลางภาพ
        gps = {
            piexif.GPSIFD.GPSLatitudeRef: "N", piexif.GPSIFD.GPSLatitude: _dms(lat),
            piexif.GPSIFD.GPSLongitudeRef: "E", piexif.GPSIFD.GPSLongitude: _dms(lon),
            piexif.GPSIFD.GPSAltitudeRef: 0, piexif.GPSIFD.GPSAltitude: (80, 1),
        }
        exif = piexif.dump({
            "0th": {piexif.ImageIFD.Make: b"DJI", piexif.ImageIFD.Model: b"FC-TEST"},
            "Exif": {piexif.ExifIFD.FocalLengthIn35mmFilm: 24},
            "GPS": gps, "1st": {}, "thumbnail": None,
        })
        Image.fromarray(a).save(
            os.path.join(OUT, f"DJI_{i:04d}.JPG"),
            exif=exif, xmp=_XMP.format(rel=REL_ALT).encode("utf-8"), quality=88,
        )
    print(f"เขียน {N} ภาพลง {OUT}/")


if __name__ == "__main__":
    main()
