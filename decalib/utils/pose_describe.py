import math
from typing import Tuple, Dict, Any, Optional, List

# -------- helpers --------
_SEVERITY_NAMES = ["frontal", "slight", "moderate", "strong", "extreme"]


def _severity_for_angle(abs_deg: float, bands: Tuple[float, float, float, float]) -> Tuple[int, str]:
    """Return (level, label) based on absolute angle in degrees."""
    b0, b1, b2, b3 = bands
    if abs_deg <= b0: return 0, _SEVERITY_NAMES[0]
    if abs_deg <= b1: return 1, _SEVERITY_NAMES[1]
    if abs_deg <= b2: return 2, _SEVERITY_NAMES[2]
    if abs_deg <= b3: return 3, _SEVERITY_NAMES[3]
    return 4, _SEVERITY_NAMES[4]


def _dir_word(v: float, pos: str, neg: str, zero: str, eps: float = 1.0) -> str:
    """Pick a direction word using a small deadzone (eps in degrees for angles)."""
    if v > eps:  return pos
    if v < -eps: return neg
    return zero


def _mag_adverb(v: float, small: float = 0.02, large: float = 0.06) -> Optional[str]:
    """Adverb for translation magnitude; thresholds are in model units."""
    av = abs(v)
    if av < 1e-6:
        return None
    if av < small:
        return "slightly"
    if av < large:
        return ""  # medium: no adverb
    return "noticeably"


def _fmt_abs(x: float, ndigits: int = 1) -> str:
    return f"{abs(x):.{ndigits}f}"


# -------- main API --------
def describe_pose(
        yaw_pitch_roll_deg: List[float],
        translation_xyz: Optional[List[float]] = None,
        rotvec_axis_angle: Optional[List[float]] = None,
        *,
        ndigits: int = 1,
) -> str:
    """
    Build a short, readable sentence describing pose from yaw/pitch/roll (deg) and translation (model units).
    If rotvec is provided, total rotation (deg) is appended for context.
    """
    yaw, pitch, roll = yaw_pitch_roll_deg
    yaw_dir = _dir_word(yaw, "right", "left", "straight")
    pitch_dir = _dir_word(pitch, "up", "down", "level")
    roll_dir = _dir_word(roll, "left", "right", "no tilt")

    parts = []
    # Orientation
    if yaw_dir == "straight" and pitch_dir == "level" and roll_dir == "no tilt":
        parts.append("Head is facing forward")
    else:
        parts.append(
            f"Head is turned {yaw_dir} (~{_fmt_abs(yaw, ndigits)}° yaw), "
            f"looking {pitch_dir} (~{_fmt_abs(pitch, ndigits)}° pitch), "
            f"and tilted {roll_dir} (~{_fmt_abs(roll, ndigits)}° roll)"
        )

    # Translation (optional)
    if translation_xyz is not None and len(translation_xyz) >= 3:
        tx, ty, tz = translation_xyz[:3]
        # X
        x_adv = _mag_adverb(tx)
        x_dir = "right" if tx > 0 else "left" if tx < 0 else None
        # Y
        y_adv = _mag_adverb(ty)
        y_dir = "up" if ty > 0 else "down" if ty < 0 else None
        # Z
        z_adv = _mag_adverb(tz)
        z_dir = "closer" if tz < 0 else "farther" if tz > 0 else None

        tr_parts = []
        if x_dir:
            tr_parts.append(f"{x_adv + ' ' if x_adv else ''}{x_dir} (X≈{tx:.3f})")
        if y_dir:
            tr_parts.append(f"{y_adv + ' ' if y_adv else ''}{y_dir} (Y≈{ty:.3f})")
        if z_dir:
            tr_parts.append(f"{z_adv + ' ' if z_adv else ''}{z_dir} (Z≈{tz:.3f})")

        if tr_parts:
            parts.append("positioned " + ", ".join(tr_parts))

    # Total rotation from axis-angle (optional)
    if rotvec_axis_angle is not None and len(rotvec_axis_angle) >= 3:
        rx, ry, rz = rotvec_axis_angle[:3]
        theta_deg = math.degrees(math.sqrt(rx * rx + ry * ry + rz * rz))
        parts.append(f"(total rotation ≈ {theta_deg:.1f}°)")

    return ". ".join(parts) + "."


def pose_severity_text(
        yaw_pitch_roll_deg: List[float],
        *,
        yaw_bands: Tuple[float, float, float, float] = (5, 15, 30, 45),
        pitch_bands: Tuple[float, float, float, float] = (5, 15, 30, 45),
        roll_bands: Tuple[float, float, float, float] = (5, 15, 25, 35),
        ndigits: int = 1,
) -> str:
    """
    Return a compact severity summary string for yaw/pitch/roll and overall severity.
    Bands are degree thresholds defining: frontal, slight, moderate, strong, extreme.
    """
    yaw, pitch, roll = yaw_pitch_roll_deg
    a_y, a_p, a_r = abs(yaw), abs(pitch), abs(roll)

    lvl_y, lab_y = _severity_for_angle(a_y, yaw_bands)
    lvl_p, lab_p = _severity_for_angle(a_p, pitch_bands)
    lvl_r, lab_r = _severity_for_angle(a_r, roll_bands)
    overall_lvl = max(lvl_y, lvl_p, lvl_r)
    overall_lab = _SEVERITY_NAMES[overall_lvl]

    yaw_dir = _dir_word(yaw, "right", "left", "straight")
    pitch_dir = _dir_word(pitch, "up", "down", "level")
    roll_dir = _dir_word(roll, "left", "right", "no tilt")

    return (
        f"Pose severity — "
        f"yaw: {lab_y} {yaw_dir} (~{_fmt_abs(yaw, ndigits)}°), "
        f"pitch: {lab_p} {pitch_dir} (~{_fmt_abs(pitch, ndigits)}°), "
        f"roll: {lab_r} {roll_dir} (~{_fmt_abs(roll, ndigits)}°). "
        f"Overall: {overall_lab}."
    )


def pose_overall_text(
        yaw_pitch_roll_deg: List[float],
        *,
        yaw_bands: Tuple[float, float, float, float] = (5, 15, 30, 45),
        pitch_bands: Tuple[float, float, float, float] = (5, 15, 30, 45),
        roll_bands: Tuple[float, float, float, float] = (5, 15, 25, 35),
) -> str:
    """
    Return a compact severity summary string for yaw/pitch/roll and overall severity.
    Bands are degree thresholds defining: frontal, slight, moderate, strong, extreme.
    """
    yaw, pitch, roll = yaw_pitch_roll_deg
    a_y, a_p, a_r = abs(yaw), abs(pitch), abs(roll)

    lvl_y, lab_y = _severity_for_angle(a_y, yaw_bands)
    lvl_p, lab_p = _severity_for_angle(a_p, pitch_bands)
    lvl_r, lab_r = _severity_for_angle(a_r, roll_bands)
    overall_lvl = max(lvl_y, lvl_p, lvl_r)
    overall_lab = _SEVERITY_NAMES[overall_lvl]

    return overall_lab


# -------- convenience to attach into your JSON dict --------
def attach_pose_descriptions(
        human_json: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Reads the keys you’re emitting in your human JSON:
      - pose.rotation_axis_angle
      - pose.yaw_pitch_roll_deg
      - pose.translation
    Writes:
      - pose_text
      - pose_severity_text
    Returns the updated dict (also mutates in place).
    """
    pose = (human_json or {}).get("pose", {})
    rot = pose.get("rotation_axis_angle")
    ypr = pose.get("yaw_pitch_roll_deg")
    trans = pose.get("translation")

    if ypr:
        human_json["pose_text"] = describe_pose(ypr, trans, rot)
        human_json["pose_severity_text"] = pose_severity_text(ypr)
        human_json["overall"] = pose_overall_text(ypr)
    return human_json
