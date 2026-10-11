import math
import os
import time
import urllib.request
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------
HOLD_SECONDS = 2.0            # how long a gesture must be held to activate
TRAIL_LIFETIME = 1.1          # seconds a glow-trail segment stays visible
DIM_FACTOR = 0.55             # background darkening while drawing
GLOW_COLOR = (255, 255, 0)    # BGR cyan
FONT = cv2.FONT_HERSHEY_SIMPLEX

CAPTURE_WIDTH = 1280           # requested webcam resolution (falls back if unsupported)
CAPTURE_HEIGHT = 720           # kept modest on purpose: the window still fills the
                                # screen (stretched), but every filter below only
                                # touches a small region, so a smaller source frame
                                # means noticeably less work per frame
FULLSCREEN = True              # start maximized/fullscreen; press 'f' to toggle

MISS_TOLERANCE_FRAMES = 10     # consecutive bad-detection frames tolerated during
                                # CHARGING before a hold timer resets
FINGER_MARGIN = 8              # pixels a tip must clear a pip by to count as "up"
POINT_SMOOTHING = 0.55         # base smoothing for gesture anchor points; combined
                                # with MOTION_GAIN below into an adaptive filter that
                                # barely smooths real, fast movement (so shapes track
                                # your fingers precisely) but still damps small
                                # camera-noise jitter when you're holding still
MOTION_GAIN = 1.0 / 22.0       # extra responsiveness added per pixel of frame-to-frame
                                # movement; higher = snappier / less lag on fast moves
REGION_PAD = 24                # pixels of margin around a gesture region for filters

PALM_CLOSE_DIST_MULT = 1.3     # palms count as "together" when closer than this x hand-scale
PALM_CLOSE_HOLD = 0.3           # seconds palms must stay together to close the frame
HANDS_LOST_TIMEOUT = 1.5        # seconds with no hands before an active frame auto-closes
PINKY_TAP_COOLDOWN = 0.5        # seconds between accepted pinky-tap filter cycles

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (one-time, ~10 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


# --------------------------------------------------------------------------
# Geometry / gesture helpers
# --------------------------------------------------------------------------
def to_px(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def fingers_up(pts):
    """pts: 21 (x, y) pixel points -> [thumb, index, middle, ring, pinky] bools.
    Thumb uses distance-from-index-MCP (robust to sideways thumb motion, unlike
    a wrist-based check). The other fingers need a pixel margin past their pip
    joint so the flag doesn't flicker when a tip sits almost level with it."""
    up = [False] * 5
    up[0] = dist(pts[4], pts[5]) > dist(pts[2], pts[5]) * 1.1
    tips, pips = [8, 12, 16, 20], [6, 10, 14, 18]
    for i, (t, p) in enumerate(zip(tips, pips)):
        up[i + 1] = pts[t][1] < pts[p][1] - FINGER_MARGIN
    return up


def is_l_pose(up):
    return up[0] and up[1] and not up[2] and not up[3] and not up[4]


def is_pointer_pose(up):
    # thumb state is ignored on purpose: a natural pointing hand often rests
    # its thumb loosely rather than tucking it tightly into the palm.
    return up[1] and not up[2] and not up[3] and not up[4]


def is_single_digit_pose(up):
    thumb_only = up[0] and not up[1] and not up[2] and not up[3] and not up[4]
    index_only = up[1] and not up[0] and not up[2] and not up[3] and not up[4]
    return thumb_only or index_only


def single_digit_index(up):
    return 4 if up[0] else 8


def is_fist(up):
    return not any(up)


def hand_scale(pts):
    return max(dist(pts[0], pts[9]), 1.0)


def palm_center(pts):
    return ((pts[0][0] + pts[9][0]) // 2, (pts[0][1] + pts[9][1]) // 2)


def palms_together(hands_data):
    if len(hands_data) != 2:
        return False
    c0 = palm_center(hands_data[0]["pts"])
    c1 = palm_center(hands_data[1]["pts"])
    scale = (hand_scale(hands_data[0]["pts"]) + hand_scale(hands_data[1]["pts"])) / 2.0
    return dist(c0, c1) < scale * PALM_CLOSE_DIST_MULT


def ordered_polygon(points):
    """Convex-hull-order an arbitrary set of 3+ points so they can be used
    directly with fillConvexPoly / polylines, whether they form a triangle
    or a quadrilateral."""
    pts = np.array(points, dtype=np.float32)
    hull = cv2.convexHull(pts)
    return hull.reshape(-1, 2).astype(np.float32)


def smooth_point(new_pt, prev_pt, alpha=POINT_SMOOTHING):
    if prev_pt is None:
        return new_pt
    return (int(prev_pt[0] + (new_pt[0] - prev_pt[0]) * alpha),
            int(prev_pt[1] + (new_pt[1] - prev_pt[1]) * alpha))


def smooth_polygon(new_poly, prev_poly, alpha=POINT_SMOOTHING):
    if prev_poly is None or len(prev_poly) != len(new_poly):
        return new_poly
    return prev_poly + (new_poly - prev_poly) * alpha


# --------------------------------------------------------------------------
# Two-hand frame gesture: which config is currently being formed?
# --------------------------------------------------------------------------
def get_frame_candidate(hands_data):
    """Returns {'A': (hand_index, [landmark_ids]), 'B': (...)} for a valid
    quad (2+2 points) or triangle (2+1 points) frame gesture, or None."""
    if len(hands_data) != 2:
        return None
    h0, h1 = hands_data
    l0, l1 = is_l_pose(h0["up"]), is_l_pose(h1["up"])
    s0, s1 = is_single_digit_pose(h0["up"]), is_single_digit_pose(h1["up"])
    if l0 and l1:
        return {"A": (0, [4, 8]), "B": (1, [4, 8])}
    if l0 and s1:
        return {"A": (0, [4, 8]), "B": (1, [single_digit_index(h1["up"])])}
    if l1 and s0:
        return {"A": (1, [4, 8]), "B": (0, [single_digit_index(h0["up"])])}
    return None


def points_from_config(hands_data, config):
    pts = []
    for role in ("A", "B"):
        hidx, lidxs = config[role]
        for li in lidxs:
            pts.append(hands_data[hidx]["pts"][li])
    return pts


# --------------------------------------------------------------------------
# Region filters (MONO / NEON / HEAT / GRAINY)
# --------------------------------------------------------------------------
def make_gray_clahe(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def filter_mono(frame):
    gray3 = cv2.cvtColor(make_gray_clahe(frame), cv2.COLOR_GRAY2BGR).astype(np.int16)
    grain = np.random.normal(0, 16, gray3.shape).astype(np.int16)
    noisy = np.clip(gray3 + grain, 0, 255).astype(np.uint8)
    return cv2.addWeighted(noisy, 0.85, frame, 0.15, 0)


def filter_grainy(frame):
    gray3 = cv2.cvtColor(make_gray_clahe(frame), cv2.COLOR_GRAY2BGR).astype(np.float32)
    grain = np.random.normal(0, 40, gray3.shape).astype(np.float32)
    noisy = np.clip(gray3 + grain, 0, 255)
    noisy[..., 0] *= 0.85  # B
    noisy[..., 1] *= 0.95  # G
    noisy[..., 2] *= 1.15  # R  (subtle warm/sepia bias)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return cv2.addWeighted(noisy, 0.85, frame, 0.15, 0)


def filter_neon(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    colored = np.zeros_like(frame)
    colored[edges > 0] = GLOW_COLOR
    glow = cv2.GaussianBlur(colored, (0, 0), 6)
    darker = (frame.astype(np.float32) * 0.3).astype(np.uint8)
    return cv2.add(darker, cv2.add(glow, colored))


def filter_heat(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(heat, 0.9, frame, 0.1, 0)


FILTER_ORDER = ["MONO", "NEON", "HEAT", "GRAINY"]
FILTER_FUNCS = {"MONO": filter_mono, "NEON": filter_neon, "HEAT": filter_heat, "GRAINY": filter_grainy}


def apply_region_filter(frame, poly_float, filter_name):
    h, w = frame.shape[:2]
    poly_int = poly_float.astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly_int, 255)
    mask_blur = cv2.GaussianBlur(mask, (31, 31), 0).astype(np.float32) / 255.0
    if mask_blur.max() <= 0:
        return frame
    filtered_full = FILTER_FUNCS[filter_name](frame)
    alpha3 = cv2.merge([mask_blur] * 3)
    out = frame.astype(np.float32) * (1 - alpha3) + filtered_full.astype(np.float32) * alpha3
    return out.astype(np.uint8)


def draw_region_overlay(layer, poly_float, filter_name):
    poly_int = poly_float.astype(np.int32)
    cv2.polylines(layer, [poly_int], True, (0, 200, 255), 1, cv2.LINE_AA)
    corner_colors = [(255, 220, 0), (255, 160, 0), (255, 120, 0), (255, 80, 0)]
    for i, pt in enumerate(poly_int):
        cv2.circle(layer, tuple(pt), 5, corner_colors[i % len(corner_colors)], -1, cv2.LINE_AA)
    xs, ys = poly_int[:, 0], poly_int[:, 1]
    box_w, box_h = int(xs.max() - xs.min()), int(ys.max() - ys.min())
    label = f"{filter_name} {box_w}x{box_h}"
    top_pt = poly_int[np.argmin(ys)]
    cv2.putText(layer, label, (int(top_pt[0]) - 40, int(top_pt[1]) - 16),
                FONT, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


def estimate_rpy(poly_float, frame_w, frame_h):
    """Stylized (not physically exact) roll/pitch/yaw readout for flavor."""
    order_by_y = poly_float[np.argsort(poly_float[:, 1])]
    top2 = order_by_y[:2]
    roll = np.degrees(np.arctan2(top2[1][1] - top2[0][1], top2[1][0] - top2[0][0]))
    cx, cy = float(poly_float[:, 0].mean()), float(poly_float[:, 1].mean())
    yaw = (cx - frame_w / 2) / (frame_w / 2) * 45
    pitch = (cy - frame_h / 2) / (frame_h / 2) * -45
    return roll, pitch, yaw


# --------------------------------------------------------------------------
# Visual effect: fading neon glow trail (draw mode)
# --------------------------------------------------------------------------
def render_glow_trail(frame, trail, now, color=GLOW_COLOR):
    glow = np.zeros_like(frame)
    for i in range(1, len(trail)):
        p0, _ = trail[i - 1]
        p1, t1 = trail[i]
        age = now - t1
        alpha = max(0.0, 1 - age / TRAIL_LIFETIME)
        if alpha <= 0:
            continue
        thickness = max(1, int(5 * alpha))
        cv2.line(glow, p0, p1, tuple(int(c * alpha) for c in color), thickness, cv2.LINE_AA)
    halo = cv2.GaussianBlur(glow, (0, 0), 9)
    combined = cv2.add(halo, glow)
    return cv2.add(frame, combined)


# --------------------------------------------------------------------------
# Hand joint skeleton
# --------------------------------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
FINGERTIPS = {4, 8, 12, 16, 20}


def draw_hand_skeleton(layer, pts, line_color=(80, 220, 120), joint_color=(60, 255, 255),
                        tip_color=(255, 255, 255)):
    for a, b in HAND_CONNECTIONS:
        cv2.line(layer, pts[a], pts[b], line_color, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        if i in FINGERTIPS:
            cv2.circle(layer, p, 5, tip_color, -1, cv2.LINE_AA)
        else:
            cv2.circle(layer, p, 3, joint_color, -1, cv2.LINE_AA)


# --------------------------------------------------------------------------
# UI chrome: brackets, REC bar, bottom info bar, progress rings
# --------------------------------------------------------------------------
def draw_frame_brackets(frame, length=26, color=(255, 255, 255), thickness=2):
    h, w = frame.shape[:2]
    for (x, y, dx, dy) in [(12, 12, 1, 1), (w - 12, 12, -1, 1),
                            (12, h - 12, 1, -1), (w - 12, h - 12, -1, -1)]:
        cv2.line(frame, (x, y), (x + dx * length, y), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x, y + dy * length), color, thickness, cv2.LINE_AA)


def draw_top_bar(frame):
    h, w = frame.shape[:2]
    if int(time.time() * 2) % 2 == 0:
        cv2.circle(frame, (26, 26), 6, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(frame, "REC", (40, 32), FONT, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    ts = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    size = cv2.getTextSize(ts, FONT, 0.55, 1)[0]
    cv2.putText(frame, ts, (w - size[0] - 16, 30), FONT, 0.55, (0, 255, 255), 1, cv2.LINE_AA)


def draw_bottom_bar(frame, state_label, fps, foto_count, filter_name=None):
    h = frame.shape[0]
    text = f"{state_label} | {filter_name or 'MONO'} | FPS {fps:.0f} | FOTO {foto_count}"
    cv2.putText(frame, text, (14, h - 16), FONT, 0.55, (0, 255, 255), 1, cv2.LINE_AA)


def draw_rpy_bar(frame, roll, pitch, yaw):
    h, w = frame.shape[:2]
    text = f"ROLL:{roll:+.0f}  PITCH:{pitch:+.0f}  YAW:{yaw:+.0f}"
    size = cv2.getTextSize(text, FONT, 0.5, 1)[0]
    cv2.putText(frame, text, (w - size[0] - 14, h - 16), FONT, 0.5, (0, 160, 255), 1, cv2.LINE_AA)


def draw_progress_ring(layer, center, radius, progress, color=(0, 255, 255)):
    cv2.circle(layer, center, radius, (255, 255, 255), 1, cv2.LINE_AA)
    end_angle = -90 + int(360 * progress)
    cv2.ellipse(layer, center, (radius, radius), 0, -90, end_angle, color, 3, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Main application
# --------------------------------------------------------------------------
def main():
    ensure_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    window_name = "AHGI - AssistantHandGestureInterface "
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    fullscreen = FULLSCREEN
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                           cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        running_mode=mp_vision.RunningMode.VIDEO,
    )

    # ---- mono/triangle frame state ----
    mono_hold_start = None
    mono_active = False
    mono_miss = MISS_TOLERANCE_FRAMES + 1
    last_frame_points = None      # smoothed polygon while charging
    smoothed_frame_poly = None
    current_filter_idx = 0

    role_offsets = {}             # role -> [(dx,dy), ...] from that hand's wrist
    role_last_wrist = {}          # role -> last known wrist point
    role_last_points = {}         # role -> last known absolute anchor point(s)
    role_prev_pinky = {}          # role -> pinky-up bool last frame
    last_pinky_tap_time = 0.0
    last_any_hand_seen_time = 0.0
    palms_close_start = None

    # ---- draw mode state ----
    draw_hold_start = None
    draw_active = False
    draw_miss = MISS_TOLERANCE_FRAMES + 1
    last_pointer_pt = None
    smoothed_pointer = None
    trail = []

    foto_count = 0
    prev_time = time.time()
    start_time = prev_time
    fps = 0.0

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = fps * 0.9 + (1.0 / dt) * 0.1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((now - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            hands_data = []
            if result.hand_landmarks:
                for lm_list in result.hand_landmarks:
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
                    hands_data.append({"pts": pts, "up": fingers_up(pts)})

            live_layer = np.zeros((h, w, 3), dtype=np.uint8)

            # always show the tracked joints
            for hd in hands_data:
                draw_hand_skeleton(live_layer, hd["pts"])

            config = None if mono_active else get_frame_candidate(hands_data)

            # ============= MONO / TRIANGLE FRAME =============
            if not mono_active:
                if config is not None:
                    mono_miss = 0
                    poly_now = ordered_polygon(points_from_config(hands_data, config))
                    smoothed_frame_poly = smooth_polygon(poly_now, smoothed_frame_poly)
                    last_frame_points = smoothed_frame_poly
                else:
                    mono_miss += 1

                mono_tolerated = last_frame_points is not None and mono_miss <= MISS_TOLERANCE_FRAMES
                if mono_tolerated:
                    poly = last_frame_points
                    if mono_hold_start is None:
                        mono_hold_start = now
                    progress = min(1.0, (now - mono_hold_start) / HOLD_SECONDS)
                    center = tuple(poly.mean(axis=0).astype(int))
                    draw_progress_ring(live_layer, center, 34, progress)
                    shape_label = "MODE MONO" if len(poly) == 4 else "MODE MONO (TRI)"
                    cv2.putText(live_layer, shape_label, (center[0] - 60, center[1] - 46),
                                FONT, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

                    if progress >= 1.0 and config is not None:
                        mono_active = True
                        foto_count += 1
                        mono_hold_start = None
                        current_filter_idx = 0
                        role_offsets, role_last_wrist, role_last_points, role_prev_pinky = {}, {}, {}, {}
                        for role in ("A", "B"):
                            hidx, lidxs = config[role]
                            wrist = hands_data[hidx]["pts"][0]
                            role_last_wrist[role] = wrist
                            role_offsets[role] = [(hands_data[hidx]["pts"][li][0] - wrist[0],
                                                    hands_data[hidx]["pts"][li][1] - wrist[1])
                                                   for li in lidxs]
                            role_last_points[role] = [hands_data[hidx]["pts"][li] for li in lidxs]
                            role_prev_pinky[role] = hands_data[hidx]["up"][4]
                        last_any_hand_seen_time = now
                        palms_close_start = None
                else:
                    mono_hold_start = None
            else:
                # ACTIVE: track each hand by wrist position; ignore exact finger pose
                mapping = {}
                if len(hands_data) == 2:
                    d1 = dist(hands_data[0]["pts"][0], role_last_wrist["A"]) + \
                         dist(hands_data[1]["pts"][0], role_last_wrist["B"])
                    d2 = dist(hands_data[0]["pts"][0], role_last_wrist["B"]) + \
                         dist(hands_data[1]["pts"][0], role_last_wrist["A"])
                    mapping = {"A": 0, "B": 1} if d1 <= d2 else {"A": 1, "B": 0}
                elif len(hands_data) == 1:
                    dA = dist(hands_data[0]["pts"][0], role_last_wrist["A"])
                    dB = dist(hands_data[0]["pts"][0], role_last_wrist["B"])
                    mapping = {"A": 0} if dA <= dB else {"B": 0}

                if mapping:
                    last_any_hand_seen_time = now
                for role, hidx in mapping.items():
                    hd = hands_data[hidx]
                    wrist = hd["pts"][0]
                    role_last_wrist[role] = wrist
                    role_last_points[role] = [(wrist[0] + dx, wrist[1] + dy) for dx, dy in role_offsets[role]]
                    cur_pinky = hd["up"][4]
                    if role_prev_pinky.get(role, False) and not cur_pinky:
                        if now - last_pinky_tap_time > PINKY_TAP_COOLDOWN:
                            current_filter_idx = (current_filter_idx + 1) % len(FILTER_ORDER)
                            last_pinky_tap_time = now
                    role_prev_pinky[role] = cur_pinky

                poly = ordered_polygon(role_last_points["A"] + role_last_points["B"])
                filt_name = FILTER_ORDER[current_filter_idx]
                frame = apply_region_filter(frame, poly, filt_name)
                draw_region_overlay(live_layer, poly, filt_name)

                should_close = False
                if len(hands_data) == 2:
                    if palms_together(hands_data):
                        if palms_close_start is None:
                            palms_close_start = now
                        elif now - palms_close_start > PALM_CLOSE_HOLD:
                            should_close = True
                    else:
                        palms_close_start = None
                if now - last_any_hand_seen_time > HANDS_LOST_TIMEOUT:
                    should_close = True
                if should_close:
                    mono_active = False
                    last_frame_points = None
                    smoothed_frame_poly = None
                    mono_hold_start = None

            # ============= GAMBAR / GLOW DRAW =============
            if not mono_active and config is None:
                pointer_hands = [hd for hd in hands_data if is_pointer_pose(hd["up"])]
                single_pointer_raw = pointer_hands[0]["pts"][8] if pointer_hands else None
                fist_present = any(is_fist(hd["up"]) for hd in hands_data) if hands_data else False

                if single_pointer_raw is not None:
                    smoothed_pointer = smooth_point(single_pointer_raw, smoothed_pointer)
                    last_pointer_pt = smoothed_pointer
                    draw_miss = 0
                else:
                    draw_miss += 1
                pointer_tolerated = last_pointer_pt is not None and draw_miss <= MISS_TOLERANCE_FRAMES

                if pointer_tolerated:
                    pt = last_pointer_pt
                    if not draw_active:
                        if draw_hold_start is None:
                            draw_hold_start = now
                        progress = min(1.0, (now - draw_hold_start) / HOLD_SECONDS)
                        draw_progress_ring(live_layer, pt, 26, progress, GLOW_COLOR)
                        cv2.putText(live_layer, "MODE GAMBAR", (pt[0] - 60, pt[1] - 40),
                                    FONT, 0.55, GLOW_COLOR, 1, cv2.LINE_AA)
                        if progress >= 1.0:
                            draw_active = True
                            draw_hold_start = None
                            trail = []
                    elif single_pointer_raw is not None:
                        trail.append((pt, now))
                else:
                    draw_hold_start = None
                    if draw_active:
                        draw_active = False
                        trail = []

                if fist_present and draw_active:
                    draw_active = False
                    trail = []

                trail = [(p, t) for (p, t) in trail if now - t < TRAIL_LIFETIME]
                if draw_active:
                    frame = (frame.astype(np.float32) * DIM_FACTOR).astype(np.uint8)
                    frame = render_glow_trail(frame, trail, now)
            else:
                draw_hold_start = None

            # ---------------- composite + chrome ----------------
            live_mask = cv2.cvtColor(live_layer, cv2.COLOR_BGR2GRAY) > 0
            frame[live_mask] = live_layer[live_mask]

            draw_frame_brackets(frame)
            draw_top_bar(frame)
            state_label = "GAMBAR" if draw_active else "LENSA"
            active_filter_name = FILTER_ORDER[current_filter_idx] if mono_active else None
            draw_bottom_bar(frame, state_label, fps, foto_count, active_filter_name)
            if mono_active:
                roll, pitch, yaw = estimate_rpy(poly, w, h)
                draw_rpy_bar(frame, roll, pitch, yaw)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('c'):
                trail = []
                foto_count = 0
                mono_active = False
                draw_active = False
                last_frame_points = None
                smoothed_frame_poly = None
            if key == ord('f'):
                fullscreen = not fullscreen
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                       cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()