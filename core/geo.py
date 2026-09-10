"""
ชั้นอ่านพิกัดภูมิศาสตร์จากไฟล์โดรน + ฉายพิกเซลในภาพเป็นพิกัดจริงบนพื้น

รองรับ 2 แหล่ง:
  - ภาพนิ่ง .JPG จากโดรน → EXIF GPS + XMP (drone-dji:*)
  - วิดีโอ + ไฟล์ .SRT (subtitle telemetry ของ DJI) → interpolate ตามเวลา

ความแม่น "ระดับกลาง": สมมติกล้องถ่ายดิ่งลง (nadir) + คำนวณ Ground Sampling Distance
จากความสูงเหนือจุดขึ้นบิน (rel_alt) และมุมมองภาพแนวนอน (HFOV)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# ---------------------------------------------------------------- constants

_M_PER_DEG_LAT = 111_320.0  # เมตรต่อองศาละติจูด (พอสำหรับสเกลพื้นที่บินโดรน)
_FULL_FRAME_MM = 36.0       # ความกว้างฟิล์ม 35mm ใช้แปลง focal(35mm-eq) เป็น HFOV

# HFOV โดยประมาณ (องศา) ของกล้องโดรนยอดนิยม — ใช้เมื่ออ่าน focal จากไฟล์ไม่ได้
CAMERA_PRESETS: dict[str, float | None] = {
    "อ่านจากไฟล์อัตโนมัติ": None,
    "DJI Mini 2 / Mini 2 SE": 71.0,
    "DJI Mini 3 / Mini 3 Pro": 73.7,
    "DJI Mini 4 Pro": 73.7,
    "DJI Air 2S": 76.0,
    "DJI Air 3 (มุมกว้าง)": 73.7,
    "DJI Mavic 3 (Hasselblad)": 74.0,
    "DJI Phantom 4 Pro": 73.7,
    "DJI Mavic 2 Pro": 68.0,
}

OBLIQUE_TOLERANCE_DEG = 20.0  # |gimbal_pitch + 90| เกินนี้ = ถือว่าเอียงมาก คลาดเคลื่อนสูง


@dataclass
class GeoPose:
    lat: float
    lon: float
    rel_alt: float | None = None          # ความสูงเหนือจุดขึ้นบิน (เมตร)
    abs_alt: float | None = None          # ความสูงเหนือระดับน้ำทะเล (เมตร)
    yaw: float | None = None              # ทิศที่ "หัวภาพ" ชี้ องศาจากทิศเหนือ (0..360)
    gimbal_pitch: float | None = None     # -90 = ดิ่งลง
    focal35: float | None = None          # focal length เทียบ 35mm (มม.)
    ts: float = 0.0                       # วินาทีนับจากต้นคลิป (ใช้กับ SRT)


# ---------------------------------------------------------------- EXIF / XMP (ภาพนิ่ง)

def _dms_to_deg(dms, ref) -> float:
    d, m, s = (float(x) for x in dms)
    val = d + m / 60.0 + s / 3600.0
    if str(ref).strip().upper() in ("S", "W"):
        val = -val
    return val


def _xmp_number(blob: str, *keys) -> float | None:
    for key in keys:
        # DJI เขียนได้ทั้งแบบ attribute  drone-dji:Key="+12.3"
        # และแบบ element           <drone-dji:Key>+12.3</drone-dji:Key>
        m = re.search(rf'{key}\s*=\s*"([+-]?[\d.]+)"', blob) or re.search(
            rf"<{key}>\s*([+-]?[\d.]+)\s*</{key}>", blob
        )
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def _read_xmp_blob(path: str) -> str:
    try:
        with open(path, "rb") as f:
            raw = f.read(512_000)  # XMP packet อยู่ต้นไฟล์ ไม่ต้องอ่านทั้งรูป
    except OSError:
        return ""
    start = raw.find(b"<x:xmpmeta")
    end = raw.find(b"</x:xmpmeta>")
    if start == -1 or end == -1:
        return ""
    return raw[start : end + 12].decode("utf-8", "ignore")


def photo_pose(path: str) -> GeoPose | None:
    """อ่าน pose จากภาพนิ่ง — คืน None ถ้าไม่มี GPS ในไฟล์"""
    from PIL import Image

    lat = lon = None
    focal35 = None
    gps_alt = None
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            gps = exif.get_ifd(0x8825) if exif else {}
            if gps and 2 in gps and 4 in gps:
                lat = _dms_to_deg(gps[2], gps.get(1, "N"))
                lon = _dms_to_deg(gps[4], gps.get(3, "E"))
                if 6 in gps:
                    try:
                        gps_alt = float(gps[6])
                        if gps.get(5) in (1, b"\x01"):
                            gps_alt = -gps_alt
                    except (TypeError, ValueError):
                        gps_alt = None
            # FocalLengthIn35mmFilm (0xA405) อยู่ใน Exif sub-IFD (0x8769)
            exif_ifd = exif.get_ifd(0x8769) if exif else {}
            ff = exif_ifd.get(0xA405) if exif_ifd else exif.get(0xA405)
            if ff is not None:
                focal35 = float(ff)
    except Exception:
        return None

    if lat is None or lon is None:
        return None

    xmp = _read_xmp_blob(path)
    rel_alt = _xmp_number(xmp, "drone-dji:RelativeAltitude", "RelativeAltitude")
    abs_alt = _xmp_number(xmp, "drone-dji:AbsoluteAltitude", "AbsoluteAltitude")
    g_yaw = _xmp_number(xmp, "drone-dji:GimbalYawDegree", "GimbalYawDegree")
    f_yaw = _xmp_number(xmp, "drone-dji:FlightYawDegree", "FlightYawDegree")
    g_pitch = _xmp_number(xmp, "drone-dji:GimbalPitchDegree", "GimbalPitchDegree")

    yaw = g_yaw if g_yaw is not None else f_yaw
    return GeoPose(
        lat=lat,
        lon=lon,
        rel_alt=rel_alt,
        abs_alt=abs_alt if abs_alt is not None else gps_alt,
        yaw=(yaw % 360.0) if yaw is not None else None,
        gimbal_pitch=g_pitch,
        focal35=focal35,
    )


# ---------------------------------------------------------------- SRT (วิดีโอ)

_SRT_TIME = re.compile(
    r"(\d\d):(\d\d):(\d\d)[,.](\d+)\s*-->\s*(\d\d):(\d\d):(\d\d)[,.](\d+)"
)


def _srt_seconds(h, m, s, ms) -> float:
    ms = float(ms)
    if ms >= 1:  # milliseconds เขียนเป็น 3 หลัก
        ms /= 1000.0 if ms < 1000 else 10 ** len(str(int(ms)))
    return int(h) * 3600 + int(m) * 60 + int(s) + ms


def _parse_srt_block(text: str) -> dict:
    out: dict[str, float] = {}

    # ---- ฟอร์แมตใหม่ (2020+):  [latitude: 13.7] [longitude: 100.5] [rel_alt: 45.2 abs_alt: 78.4] [gimbal_yaw: 12.3 ...]
    for key in ("latitude", "longitude", "rel_alt", "abs_alt",
                "gimbal_yaw", "gimbal_pitch", "drone_yaw", "flight_yaw", "focal_len"):
        m = re.search(rf"\[?{key}\s*:\s*([+-]?[\d.]+)", text)
        if m:
            out[key] = float(m.group(1))

    # ---- ฟอร์แมตเก่า:  GPS(100.523,13.736,20) ... D 0.00m,H 45.20m ...  หรือ  HOME(...)
    if "latitude" not in out:
        m = re.search(r"GPS\s*\(\s*([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)", text)
        if m:  # GPS(lon, lat, alt)
            out["longitude"] = float(m.group(1))
            out["latitude"] = float(m.group(2))
    if "rel_alt" not in out:
        m = re.search(r"\bH\s*([+-]?[\d.]+)\s*m", text)
        if m:
            out["rel_alt"] = float(m.group(1))

    return out


def parse_srt(path: str) -> list[GeoPose]:
    """อ่านไฟล์ .SRT ของ DJI เป็น list[GeoPose] เรียงตามเวลา"""
    try:
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
            content = f.read()
    except OSError:
        return []

    poses: list[GeoPose] = []
    # แยกเป็นบล็อกด้วยบรรทัดว่าง
    for block in re.split(r"\n\s*\n", content):
        tm = _SRT_TIME.search(block)
        if not tm:
            continue
        t0 = _srt_seconds(*tm.group(1, 2, 3, 4))
        t1 = _srt_seconds(*tm.group(5, 6, 7, 8))
        ts = (t0 + t1) / 2.0
        d = _parse_srt_block(block)
        if "latitude" not in d or "longitude" not in d:
            continue
        yaw = d.get("gimbal_yaw", d.get("drone_yaw", d.get("flight_yaw")))
        poses.append(
            GeoPose(
                lat=d["latitude"],
                lon=d["longitude"],
                rel_alt=d.get("rel_alt"),
                abs_alt=d.get("abs_alt"),
                yaw=(yaw % 360.0) if yaw is not None else None,
                gimbal_pitch=d.get("gimbal_pitch"),
                focal35=d.get("focal_len"),
                ts=ts,
            )
        )
    poses.sort(key=lambda p: p.ts)

    # SRT บางรุ่น (เช่น DJI Lito X1) ไม่มี gimbal_yaw → เดา "ทิศหัวภาพ" จากทิศที่โดรนบินไป
    # (โดรนบินหน้าตรง กล้องหันตามหัว → หัวภาพ ≈ ทิศเดินทาง)
    if poses and all(p.yaw is None for p in poses):
        _fill_yaw_from_track(poses)

    return poses


def _bearing(lat1, lon1, lat2, lon2) -> float:
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(
        math.radians(lat1)
    ) * math.cos(math.radians(lat2)) * math.cos(dlon)
    return math.degrees(math.atan2(y, x)) % 360.0


def _fill_yaw_from_track(poses: list[GeoPose], min_move_m: float = 0.7) -> None:
    last_yaw = 0.0
    for i, p in enumerate(poses):
        j = i + 1
        while j < len(poses) and haversine_m(p.lat, p.lon, poses[j].lat, poses[j].lon) < min_move_m:
            j += 1
        if j < len(poses):
            last_yaw = _bearing(p.lat, p.lon, poses[j].lat, poses[j].lon)
        p.yaw = last_yaw


def _lerp(a, b, f):
    if a is None or b is None:
        return a if b is None else b
    return a + (b - a) * f


def _lerp_angle(a, b, f):
    if a is None or b is None:
        return a if b is None else b
    diff = ((b - a + 180.0) % 360.0) - 180.0
    return (a + diff * f) % 360.0


def pose_at(samples: list[GeoPose], t: float) -> GeoPose | None:
    """interpolate pose ณ เวลา t (วินาที) จาก list ที่ parse_srt คืนมา"""
    if not samples:
        return None
    if t <= samples[0].ts:
        return samples[0]
    if t >= samples[-1].ts:
        return samples[-1]
    for i in range(1, len(samples)):
        p0, p1 = samples[i - 1], samples[i]
        if p0.ts <= t <= p1.ts:
            span = (p1.ts - p0.ts) or 1e-9
            f = (t - p0.ts) / span
            return GeoPose(
                lat=_lerp(p0.lat, p1.lat, f),
                lon=_lerp(p0.lon, p1.lon, f),
                rel_alt=_lerp(p0.rel_alt, p1.rel_alt, f),
                abs_alt=_lerp(p0.abs_alt, p1.abs_alt, f),
                yaw=_lerp_angle(p0.yaw, p1.yaw, f),
                gimbal_pitch=_lerp(p0.gimbal_pitch, p1.gimbal_pitch, f),
                focal35=p0.focal35 or p1.focal35,
                ts=t,
            )
    return samples[-1]


# ---------------------------------------------------------------- projection

def hfov_from_focal35(focal35: float, img_w: int, img_h: int) -> float:
    """HFOV (องศา) จาก focal length เทียบ 35mm — ปรับตามอัตราส่วนภาพจริง
    (36mm คือด้านยาวของฟิล์ม 35mm = แนวนอนของภาพ 3:2)"""
    base = 2.0 * math.degrees(math.atan(_FULL_FRAME_MM / (2.0 * focal35)))
    # ถ้าภาพไม่ใช่ 3:2 ให้ครอปตามแนวนอน (โดยประมาณ)
    ratio = (img_w / img_h) / (3.0 / 2.0) if img_h else 1.0
    if ratio < 1.0:
        base = 2.0 * math.degrees(math.atan(math.tan(math.radians(base / 2.0)) * ratio))
    return base


def resolve_hfov(
    focal35: float | None,
    preset_name: str | None,
    manual_hfov: float | None,
    img_w: int,
    img_h: int,
) -> float:
    """เลือก HFOV ตามลำดับ: ค่าที่กรอกเอง > preset รุ่น > อ่านจาก focal ในไฟล์ > 73.7 (ดีฟอลต์ DJI)"""
    if manual_hfov and manual_hfov > 1:
        return float(manual_hfov)
    if preset_name and CAMERA_PRESETS.get(preset_name):
        return float(CAMERA_PRESETS[preset_name])
    if focal35 and focal35 > 1:
        return hfov_from_focal35(focal35, img_w, img_h)
    return 73.7


def pixel_to_world(
    pose: GeoPose,
    px: float,
    py: float,
    img_w: int,
    img_h: int,
    hfov_deg: float,
) -> tuple[float, float, bool] | None:
    """แปลงพิกเซล (px, py) ในภาพ → (lat, lon, oblique_flag)

    ยิงรังสีจากกล้อง (pinhole) ผ่านพิกเซล ไปตัดพื้นราบที่ระดับจุดขึ้นบิน
    รองรับทั้งกล้องถ่ายดิ่ง (nadir) และเอียง (oblique) โดยใช้ gimbal_pitch/yaw
    คืน None ถ้าไม่รู้ความสูง หรือรังสีไม่ตัดพื้น (กล้องเงยขึ้นฟ้า)
    """
    h = pose.rel_alt
    if h is None or h <= 0:
        return None

    f = (img_w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)   # focal (พิกเซล)
    rx, ry, rz = px - img_w / 2.0, py - img_h / 2.0, f            # รังสีในกรอบกล้อง (x ขวา, y ลง, z แกนเลนส์)

    phi = math.radians(pose.gimbal_pitch if pose.gimbal_pitch is not None else -90.0)
    psi = math.radians(pose.yaw if pose.yaw is not None else 0.0)
    cph, sph = math.cos(phi), math.sin(phi)

    # หมุนรังสีเข้าโลก (E, N, U) ก่อนหมุน yaw:
    #   forward = (0, cosφ, sinφ)   → φ=-90° = ดิ่งลง, φ=0° = ระดับขอบฟ้าไปทิศเหนือ
    #   right   = (1, 0, 0)
    #   img_down= (0, sinφ, -cosφ)
    e = rx
    n = ry * sph + rz * cph
    u = -ry * cph + rz * sph

    # หมุน yaw (ตามเข็มนาฬิกาจากทิศเหนือ) รอบแกนขึ้น
    cps, sps = math.cos(psi), math.sin(psi)
    e2 = e * cps + n * sps
    n2 = -e * sps + n * cps

    if u >= 0:            # รังสีชี้ขึ้นฟ้า ไม่ตัดพื้น
        return None
    t = -h / u
    east, north = t * e2, t * n2

    dlat = north / _M_PER_DEG_LAT
    dlon = east / (_M_PER_DEG_LAT * math.cos(math.radians(pose.lat)))

    oblique = (
        pose.gimbal_pitch is not None
        and abs(pose.gimbal_pitch + 90.0) > OBLIQUE_TOLERANCE_DEG
    )
    return pose.lat + dlat, pose.lon + dlon, oblique


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
