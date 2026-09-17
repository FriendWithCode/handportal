"""
=============================================================================
 HAND PORTAL REAL-TIME FX - MEDIAPIPE TASKS & OPENCV (STABLE DUAL HAND)
=============================================================================
Perbaikan Lengkap:
 1. Kamera Stabil (CAP_DSHOW):
    - Membuka kamera dengan backend DirectShow yang stabil tanpa error MSMF grabFrame.
 2. Deteksi 2 Tangan Stabil & Kuat:
    - RunningMode.VIDEO dengan MediaPipe Tasks HandLandmarker.
    - Confidence threshold optimal (0.30) agar mengunci kedua tangan dengan cepat
      dan tidak mendeteksi objek palsu pada pakaian.
 3. Dual Hand Persistence (Anti-Flicker):
    - Jika kedua tangan sedang memegang portal dan salah satu tangan sempat hilang
      sekejap (1-8 frame karena sudut miring atau gerakan cepat), posisinya
      dipertahankan (hold) agar portal 2 tangan tidak putus/hilang!
 4. Pemisahan Kiri & Kanan (Spatial Slotting):
    - Tangan di sisi kiri layar otomatis menjadi Hand Left, dan di sisi kanan Hand Right.
 5. 4 Jari (2D Quad) Persis Foto Referensi:
    - 4 Sudut: Telunjuk Kiri [8], Telunjuk Kanan [8], Jempol Kanan [4], Jempol Kiri [4].
    - Sisi kanan meruncing lancip saat mencubit (seperti di foto referensi pengguna).
 6. Ganti Filter via Pinch & Keyboard:
    - Gesture PINCH (cubit jempol-telunjuk) langsung berganti 1x seketika.
    - Schmitt Trigger Hysteresis: Filter terkunci stabil dan TIDAK AKAN SPAM
      berganti puluhan kali saat menahan cubitan!
    - Tombol Keyboard: 'N' (next), 'P' (prev), SPASI, dan angka 1-9.
=============================================================================
"""

import os
import sys
import time
import math
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =============================================================================
# MODEL ASSET AUTO-DOWNLOADER
# =============================================================================
MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

def ensure_model_exists():
    """Memastikan model hand_landmarker.task tersedia secara otomatis."""
    if not os.path.exists(MODEL_FILENAME):
        print(f"[INFO] Mengunduh '{MODEL_FILENAME}' dari Google...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_FILENAME)
            print(f"[SUCCESS] Model siap digunakan ({os.path.getsize(MODEL_FILENAME)} bytes).")
        except Exception as e:
            print(f"[ERROR] Gagal mengunduh model: {e}")
            sys.exit(1)


# =============================================================================
# KONEKSI SKELETON TULANG TANGAN (21 TITIK)
# =============================================================================
HAND_CONNECTIONS = [
    # Ibu Jari
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Jari Telunjuk
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Jari Tengah
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Jari Manis
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Jari Kelingking
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Pangkal Telapak
    (0, 17)
]


# =============================================================================
# 1. ADAPTIVE LANDMARK SMOOTHER (ZERO-LAG)
# =============================================================================
class AdaptiveLandmarkSmoother:
    def __init__(self, min_alpha=0.52, max_alpha=0.92, vel_threshold=14.0):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.vel_threshold = vel_threshold
        self.prev_landmarks = {}

    def smooth(self, slot_id, current_pts):
        if slot_id not in self.prev_landmarks or self.prev_landmarks[slot_id] is None:
            self.prev_landmarks[slot_id] = current_pts.copy()
            return current_pts

        prev = self.prev_landmarks[slot_id]
        velocity = float(np.mean(np.linalg.norm(current_pts[:, :2] - prev[:, :2], axis=1)))
        factor = min(1.0, velocity / self.vel_threshold)
        alpha = self.min_alpha + factor * (self.max_alpha - self.min_alpha)

        smoothed = alpha * current_pts + (1.0 - alpha) * prev
        self.prev_landmarks[slot_id] = smoothed
        return smoothed

    def reset(self, slot_id=None):
        if slot_id:
            self.prev_landmarks.pop(slot_id, None)
        else:
            self.prev_landmarks.clear()


# =============================================================================
# 2. GESTURE DETECTOR DENGAN SCHMITT TRIGGER (ANTI-CHATTER)
# =============================================================================
class GestureDetector:
    def __init__(self, enter_pinch_ratio=0.45, exit_pinch_ratio=0.60):
        self.enter_pinch_ratio = enter_pinch_ratio
        self.exit_pinch_ratio = exit_pinch_ratio
        self.pinch_states = {}  # state per slot ('left', 'right')

    def analyze_hand(self, slot_id, landmarks_3d, frame_w, frame_h):
        wrist = landmarks_3d[0, :2]
        thumb_tip = landmarks_3d[4, :2]
        index_tip = landmarks_3d[8, :2]
        middle_mcp = landmarks_3d[9, :2]

        palm_scale = float(np.linalg.norm(middle_mcp - wrist))
        palm_scale = max(palm_scale, 20.0)

        # Jarak cubitan jempol-telunjuk dinormalisasi
        dist_4_8 = float(np.linalg.norm(thumb_tip - index_tip))
        pinch_ratio = dist_4_8 / palm_scale

        # Schmitt Trigger Hysteresis
        curr_state = self.pinch_states.get(slot_id, False)
        just_triggered = False

        if not curr_state and pinch_ratio < self.enter_pinch_ratio:
            self.pinch_states[slot_id] = True
            just_triggered = True  # Baru saja mencubit
        elif curr_state and pinch_ratio > self.exit_pinch_ratio:
            self.pinch_states[slot_id] = False  # Jari sudah terbuka kembali

        # Fist detection
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        fist_votes = 0
        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            d_tip = np.linalg.norm(landmarks_3d[tip_idx, :2] - wrist)
            d_pip = np.linalg.norm(landmarks_3d[pip_idx, :2] - wrist)
            if d_tip < d_pip:
                fist_votes += 1

        is_fist = (fist_votes >= 3)

        return {
            'slot_id': slot_id,
            'is_pinch': self.pinch_states.get(slot_id, False),
            'just_pinched': just_triggered,
            'is_fist': is_fist,
            'pinch_ratio': pinch_ratio,
            'wrist': tuple(wrist.astype(int)),
            'thumb_tip': tuple(thumb_tip.astype(int)),
            'index_tip': tuple(index_tip.astype(int)),
            'palm_center': tuple(landmarks_3d[[0, 5, 9, 13, 17], :2].mean(axis=0).astype(int)),
            'landmarks_3d': landmarks_3d
        }


# =============================================================================
# 3. FILTER ENGINE (PRE-WARMED UNTUK ZERO DELAY)
# =============================================================================
class FilterEngine:
    _duotone_lut = None

    @classmethod
    def _get_duotone_lut(cls):
        if cls._duotone_lut is None:
            lut = np.zeros((256, 1, 3), dtype=np.uint8)
            for i in range(256):
                t = i / 255.0
                b = int((1.0 - t) * 80 + t * 240)
                g = int((1.0 - t) * 10 + t * 60)
                r = int((1.0 - t) * 20 + t * 255)
                lut[i, 0] = [np.clip(b, 0, 255), np.clip(g, 0, 255), np.clip(r, 0, 255)]
            cls._duotone_lut = lut
        return cls._duotone_lut

    @classmethod
    def warmup_filters(cls):
        """Memanaskan semua filter saat startup."""
        dummy = np.zeros((80, 80, 3), dtype=np.uint8)
        cls.filter_invert(dummy)
        cls.filter_rainbow_wave(dummy, 1.0)
        cls.filter_pixelate(dummy, 20)
        cls.filter_cartoon(dummy)
        cls.filter_dual_tone(dummy)
        cls.filter_glitch(dummy, 1.0)
        cls.filter_thermal(dummy)
        cls.filter_edge(dummy)
        cls.filter_blur(dummy, 21)

    @staticmethod
    def filter_rainbow_wave(frame, t_sec, speed=85.0, freq=0.007):
        """1. RAINBOW-WAVE"""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        x_indices = np.arange(w, dtype=np.float32)
        hue_shift = ((t_sec * speed) + (x_indices * freq * 180.0)) % 180.0
        h_channel = (hsv[:, :, 0].astype(np.int32) + hue_shift[np.newaxis, :].astype(np.uint8)) % 180
        hsv[:, :, 0] = h_channel.astype(np.uint8)
        hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 35)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def filter_invert(frame):
        """2. INVERT (Default persis referensi foto)"""
        return cv2.bitwise_not(frame)

    @staticmethod
    def filter_pixelate(frame, pixel_size=20):
        """3. PIXELATE"""
        h, w = frame.shape[:2]
        if h <= pixel_size or w <= pixel_size:
            return frame
        small = cv2.resize(frame, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def filter_cartoon(frame):
        """4. CARTOON"""
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_LINEAR)
        color_small = cv2.medianBlur(small, 7)
        color = cv2.resize(color_small, (w, h), interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(color, edges_3ch)

    @classmethod
    def filter_dual_tone(cls, frame):
        """5. DUAL-TONE"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lut = cls._get_duotone_lut()
        return cv2.LUT(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), lut)

    @staticmethod
    def filter_glitch(frame, t_sec):
        """6. GLITCH"""
        h, w = frame.shape[:2]
        glitched = frame.copy()
        shift = int(8 * math.sin(t_sec * 12.0))
        if shift != 0:
            glitched[:, :, 2] = np.roll(frame[:, :, 2], shift, axis=1)
            glitched[:, :, 0] = np.roll(frame[:, :, 0], -shift, axis=1)

        if int(t_sec * 8) % 3 == 0 and h > 40:
            slice_y = int((math.sin(t_sec * 7.0) * 0.5 + 0.5) * (h - 30))
            slice_h = min(20, h - slice_y)
            slice_shift = int(math.cos(t_sec * 20.0) * 18)
            glitched[slice_y:slice_y + slice_h, :, :] = np.roll(
                glitched[slice_y:slice_y + slice_h, :, :], slice_shift, axis=1
            )
        return glitched

    @staticmethod
    def filter_thermal(frame):
        """7. THERMAL"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    @staticmethod
    def filter_edge(frame):
        """8. EDGE"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 140)
        edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        h, w = frame.shape[:2]
        neon = np.zeros((h, w, 3), dtype=np.uint8)
        neon[edges_dilated > 0] = [255, 230, 20]
        neon[edges > 0] = [255, 100, 255]
        darkened = cv2.multiply(frame, 0.25).astype(np.uint8)
        return cv2.add(darkened, neon)

    @staticmethod
    def filter_blur(frame, ksize=29):
        """(+ BLUR)"""
        k = ksize if ksize % 2 == 1 else ksize + 1
        return cv2.GaussianBlur(frame, (k, k), 0)


# =============================================================================
# 4. PORTAL GEOMETRY (4 JARI / 2D QUAD PERSIS FOTO REFERENSI)
# =============================================================================
class PortalGeometry:
    @staticmethod
    def build_2d_quad(hand_left, hand_right):
        p_l_top = hand_left['index_tip']
        p_r_top = hand_right['index_tip']
        p_r_bot = hand_right['thumb_tip']
        p_l_bot = hand_left['thumb_tip']
        return np.array([p_l_top, p_r_top, p_r_bot, p_l_bot], dtype=np.int32)

    @staticmethod
    def build_2d_bowtie(hand_left, hand_right):
        p_l_top = hand_left['index_tip']
        p_l_bot = hand_left['thumb_tip']
        p_r_top = hand_right['index_tip']
        p_r_bot = hand_right['thumb_tip']
        return np.array([p_l_top, p_r_bot, p_r_top, p_l_bot], dtype=np.int32)

    @staticmethod
    def build_3d_mesh_data(hand_left, hand_right):
        l_lms = hand_left['landmarks_3d']
        r_lms = hand_right['landmarks_3d']

        key_indices = [0, 4, 8, 12, 16, 20]
        pts_left = l_lms[key_indices, :2].astype(np.int32)
        pts_right = r_lms[key_indices, :2].astype(np.int32)
        boundary_pts = np.vstack([pts_left, pts_right[::-1]])

        mesh_edges = []
        for idx in key_indices:
            pt_l = tuple(l_lms[idx, :2].astype(int))
            pt_r = tuple(r_lms[idx, :2].astype(int))
            z_diff = abs(float(l_lms[idx, 2] - r_lms[idx, 2]))
            mesh_edges.append((pt_l, pt_r, z_diff))

        cross_pairs = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 0)]
        for i1, i2 in cross_pairs:
            pt_l = tuple(l_lms[i1, :2].astype(int))
            pt_r = tuple(r_lms[i2, :2].astype(int))
            z_diff = abs(float(l_lms[i1, 2] - r_lms[i2, 2]))
            mesh_edges.append((pt_l, pt_r, z_diff))

        return boundary_pts, mesh_edges

    @staticmethod
    def build_single_hand_portal(hand_info, radius=110):
        center = hand_info['palm_center']
        num_pts = 6
        angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
        pts = []
        for a in angles:
            x = int(center[0] + radius * math.cos(a))
            y = int(center[1] + radius * math.sin(a))
            pts.append([x, y])
        return np.array(pts, dtype=np.int32)


# =============================================================================
# 5. VISUALISASI SKELETON JARI KAMERA (TITIK MERAH MEDIAPIPE PERSIS FOTO)
# =============================================================================
class HandSkeletonVisualizer:
    @staticmethod
    def draw_skeleton(frame, hands_list, opacity=1.0):
        if opacity <= 0.04 or not hands_list:
            return

        line_col = (int(220 * opacity), int(220 * opacity), int(220 * opacity))
        dot_col = (0, 0, int(255 * opacity))
        ring_col = (int(255 * opacity), int(255 * opacity), int(255 * opacity))

        for hand in hands_list:
            lms_3d = hand['landmarks_3d']
            pts_2d = lms_3d[:, :2].astype(int)

            # Garis tulang tipis
            for p1_idx, p2_idx in HAND_CONNECTIONS:
                pt1 = tuple(pts_2d[p1_idx])
                pt2 = tuple(pts_2d[p2_idx])
                cv2.line(frame, pt1, pt2, line_col, 1, cv2.LINE_AA)

            # Titik merah di setiap 21 sendi persis foto referensi
            for pt in pts_2d:
                cv2.circle(frame, tuple(pt), 4, dot_col, -1, cv2.LINE_AA)
                cv2.circle(frame, tuple(pt), 4, ring_col, 1, cv2.LINE_AA)


# =============================================================================
# 6. COMPOSITOR DENGAN OUTLINE PUTIH TEGAS (DENGAN SMOOTH FADE)
# =============================================================================
class Compositor:
    @staticmethod
    def composite_portal_roi(frame, filter_func, polygon_pts, mesh_edges=None, feather_ksize=5, draw_border=True, border_color=(255, 255, 255), opacity=1.0):
        h, w = frame.shape[:2]
        if polygon_pts is None or len(polygon_pts) < 3 or opacity <= 0.02:
            return frame

        pad = 12
        x_min = max(0, int(np.min(polygon_pts[:, 0])) - pad)
        y_min = max(0, int(np.min(polygon_pts[:, 1])) - pad)
        x_max = min(w, int(np.max(polygon_pts[:, 0])) + pad)
        y_max = min(h, int(np.max(polygon_pts[:, 1])) + pad)

        if x_max <= x_min or y_max <= y_min:
            return frame

        roi_frame = frame[y_min:y_max, x_min:x_max]
        roi_h, roi_w = roi_frame.shape[:2]

        poly_local = polygon_pts - np.array([x_min, y_min], dtype=np.int32)

        mask_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.fillPoly(mask_roi, [poly_local], 255)

        filtered_roi = filter_func(roi_frame)

        if feather_ksize > 1:
            k = feather_ksize if feather_ksize % 2 == 1 else feather_ksize + 1
            mask_float = cv2.GaussianBlur(mask_roi, (k, k), 0).astype(np.float32) / 255.0
        else:
            mask_float = mask_roi.astype(np.float32) / 255.0

        if opacity < 1.0:
            mask_float = mask_float * float(np.clip(opacity, 0.0, 1.0))

        mask_3ch = np.repeat(mask_float[:, :, np.newaxis], 3, axis=2)

        blended_roi = (roi_frame.astype(np.float32) * (1.0 - mask_3ch) + filtered_roi.astype(np.float32) * mask_3ch).astype(np.uint8)

        if mesh_edges is not None:
            for pt1, pt2, z_diff in mesh_edges:
                p1_loc = (pt1[0] - x_min, pt1[1] - y_min)
                p2_loc = (pt2[0] - x_min, pt2[1] - y_min)
                intensity = int(np.clip((255 - z_diff * 400) * opacity, 50, 255))
                edge_color = (intensity, int(255 * opacity), int(intensity * 0.7))
                cv2.line(blended_roi, p1_loc, p2_loc, edge_color, 1, cv2.LINE_AA)

        if draw_border and opacity > 0.05:
            b_c = tuple(int(c * opacity) for c in border_color)
            cv2.polylines(blended_roi, [poly_local], isClosed=True, color=b_c, thickness=2, lineType=cv2.LINE_AA)

        frame[y_min:y_max, x_min:x_max] = blended_roi
        return frame


# =============================================================================
# 7. CLEAN HUD (PERSIS FOTO: TEXT KUNING DI POJOK KIRI ATAS)
# =============================================================================
class CleanHUD:
    @staticmethod
    def draw_hud(frame, current_mode, geo_mode, switched_flash=False):
        mode_str = f"MODE: {geo_mode} [Key 'C' / Dual Fist]"
        filter_str = f"FILTER: {current_mode} [Pinch / Key 'N'/'P']"

        text_color = (0, 255, 255)
        if switched_flash:
            filter_str += "  << SWITCHED! >>"
            text_color = (0, 255, 120)

        cv2.putText(frame, mode_str, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, lineType=cv2.LINE_AA)
        cv2.putText(frame, mode_str, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 1, lineType=cv2.LINE_AA)

        cv2.putText(frame, filter_str, (18, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, lineType=cv2.LINE_AA)
        cv2.putText(frame, filter_str, (18, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.58, text_color, 1, lineType=cv2.LINE_AA)


# =============================================================================
# 8. DUAL HAND PERSISTENCE TRACKER (MENCEGAH KEDUA TANGAN PUTUS-PUTUS)
# =============================================================================
class DualHandPersistenceTracker:
    def __init__(self, max_hold_frames=15):
        self.max_hold_frames = max_hold_frames
        self.cached_left = None
        self.cached_right = None
        self.left_lost_count = 999
        self.right_lost_count = 999

    def update(self, detected_left, detected_right):
        if detected_left is not None:
            self.cached_left = detected_left
            self.left_lost_count = 0
        else:
            self.left_lost_count += 1
            if self.left_lost_count > self.max_hold_frames:
                self.cached_left = None

        if detected_right is not None:
            self.cached_right = detected_right
            self.right_lost_count = 0
        else:
            self.right_lost_count += 1
            if self.right_lost_count > self.max_hold_frames:
                self.cached_right = None

        return self.cached_left, self.cached_right


# =============================================================================
# 9. MAIN APPLICATION CONTROLLER
# =============================================================================
class HandPortalApp:
    def __init__(self, camera_index=0, width=1280, height=720):
        ensure_model_exists()
        FilterEngine.warmup_filters()

        self.camera_index = camera_index
        self.target_w = width
        self.target_h = height

        # Filter List
        self.modes = [
            "INVERT",
            "RAINBOW-WAVE",
            "PIXELATE",
            "CARTOON",
            "DUAL-TONE",
            "GLITCH",
            "THERMAL",
            "EDGE",
            "BLUR"
        ]
        self.mode_index = 0

        # Mode Geometri (DEFAULT KE 2D QUAD: 4 JARI PERSIS FOTO)
        self.geo_modes = [
            "2D Quad",
            "2D Bowtie",
            "3D Mesh"
        ]
        self.geo_index = 0

        # Gesture & Control
        self.last_pinch_switch_time = 0.0
        self.pinch_switch_cooldown = 0.65
        self.flash_until_time = 0.0

        # Smoothing & Detection
        self.smoothing_enabled = True
        self.border_enabled = True
        self.show_skeleton = True
        self.smoother = AdaptiveLandmarkSmoother(min_alpha=0.52, max_alpha=0.92, vel_threshold=14.0)
        self.gesture_detector = GestureDetector(enter_pinch_ratio=0.35, exit_pinch_ratio=0.50)
        self.persistence_tracker = DualHandPersistenceTracker(max_hold_frames=8)
        self.portal_alpha = 0.0
        self.last_polygon_pts = None
        self.last_mesh_edges = None

        # Inisialisasi MediaPipe Tasks HandLandmarker (RunningMode.VIDEO)
        print("[INFO] Memuat MediaPipe HandLandmarker (Deteksi 2 Tangan)...")
        base_options = python.BaseOptions(model_asset_path=MODEL_FILENAME)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.30,
            min_hand_presence_confidence=0.30,
            min_tracking_confidence=0.30,
            running_mode=vision.RunningMode.VIDEO
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        print("[SUCCESS] HandLandmarker siap beroperasi.")

    def next_mode(self):
        self.mode_index = (self.mode_index + 1) % len(self.modes)
        self.flash_until_time = time.time() + 0.45
        print(f"[FILTER BERGANTI] -> {self.modes[self.mode_index]}")

    def prev_mode(self):
        self.mode_index = (self.mode_index - 1) % len(self.modes)
        self.flash_until_time = time.time() + 0.45
        print(f"[FILTER BERGANTI] -> {self.modes[self.mode_index]}")

    def next_geo_mode(self):
        self.geo_index = (self.geo_index + 1) % len(self.geo_modes)
        print(f"[GEOMETRY BERGANTI] -> {self.geo_modes[self.geo_index]}")

    def get_current_filter_func(self, t_sec):
        current = self.modes[self.mode_index]
        if current == "INVERT":
            return FilterEngine.filter_invert
        elif current == "RAINBOW-WAVE":
            return lambda img: FilterEngine.filter_rainbow_wave(img, t_sec)
        elif current == "PIXELATE":
            return lambda img: FilterEngine.filter_pixelate(img, pixel_size=20)
        elif current == "CARTOON":
            return FilterEngine.filter_cartoon
        elif current == "DUAL-TONE":
            return FilterEngine.filter_dual_tone
        elif current == "GLITCH":
            return lambda img: FilterEngine.filter_glitch(img, t_sec)
        elif current == "THERMAL":
            return FilterEngine.filter_thermal
        elif current == "EDGE":
            return FilterEngine.filter_edge
        elif current == "BLUR":
            return lambda img: FilterEngine.filter_blur(img, ksize=29)
        return lambda img: img

    def run(self):
        print(f"[INFO] Membuka kamera {self.camera_index}...")
        # Buka kamera dengan DirectShow yang stabil di Windows
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[WARN] Kamera {self.camera_index} via DSHOW gagal. Mencoba default backend...")
            cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            print("[ERROR] Kamera tidak ditemukan.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        window_name = "EchoRegion"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1120, 630)

        print("\n" + "=" * 65)
        print(" ECHO REGION / HAND PORTAL AKTIF (STABLE DUAL-HAND)!")
        print(" - Deteksi 2 Tangan Stabil (Video Tracking + Persistence)")
        print(" - MODE: 2D Quad (4 Jari persis foto referensi)")
        print(" - GANTI FILTER: Cubit jari (Pinch) atau tekan 'N' / 'P' / SPASI")
        print(" - TEKAN '1'-'9': Pilih filter langsung")
        print(" - Tekan 'c' (Ganti Geometri) | 'q' (Keluar)")
        print("=" * 65 + "\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.005)
                    continue

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                curr_time = time.time()

                # Inferensi 640x360 Full-Frame untuk kedua tangan
                infer_w = 640
                infer_h = int(infer_w * h / w)
                infer_frame = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
                rgb_infer = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_infer)

                timestamp_ms = int(curr_time * 1000)
                detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

                # Ekstraksi dan pengelompokan spasial tangan kiri & kanan
                detected_left = None
                detected_right = None

                if detection_result and detection_result.hand_landmarks:
                    raw_hands = []
                    for hand_lms in detection_result.hand_landmarks:
                        pts_3d = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in hand_lms], dtype=np.float32)
                        palm_cx = float(np.mean(pts_3d[[0, 5, 9, 13, 17], 0]))
                        raw_hands.append({'pts': pts_3d, 'cx': palm_cx})

                    # Urutkan berdasarkan posisi X horizontal
                    raw_hands.sort(key=lambda item: item['cx'])

                    if len(raw_hands) >= 2:
                        left_pts = raw_hands[0]['pts']
                        right_pts = raw_hands[-1]['pts']

                        if self.smoothing_enabled:
                            left_pts = self.smoother.smooth('left', left_pts)
                            right_pts = self.smoother.smooth('right', right_pts)

                        detected_left = self.gesture_detector.analyze_hand('left', left_pts, w, h)
                        detected_right = self.gesture_detector.analyze_hand('right', right_pts, w, h)

                    elif len(raw_hands) == 1:
                        pts = raw_hands[0]['pts']
                        cx = raw_hands[0]['cx']
                        slot = 'left' if cx < w / 2 else 'right'

                        if self.smoothing_enabled:
                            pts = self.smoother.smooth(slot, pts)

                        analysis = self.gesture_detector.analyze_hand(slot, pts, w, h)
                        if slot == 'left':
                            detected_left = analysis
                        else:
                            detected_right = analysis

                # Persistence Tracker: Menahan posisi sesaat jika salah satu tangan sempat drop
                active_left, active_right = self.persistence_tracker.update(detected_left, detected_right)

                # TRIGGER GANTI FILTER DENGAN PINCH (ANTI-CHATTER / ZERO SPAM)
                active_hands = [h for h in [active_left, active_right] if h is not None]
                any_just_pinched = any(h.get('just_pinched', False) for h in active_hands)

                if any_just_pinched and (curr_time - self.last_pinch_switch_time >= self.pinch_switch_cooldown):
                    self.next_mode()
                    self.last_pinch_switch_time = curr_time

                # Hitung Poligon Geometri Portal
                polygon_pts = None
                mesh_edges = None
                current_geo = self.geo_modes[self.geo_index]

                has_hands = (active_left is not None or active_right is not None)

                # Transisi Mulus Fade-In / Fade-Out Opacity Portal (Lebih Halus Saat Tangan Hilang)
                if has_hands:
                    self.portal_alpha = min(1.0, self.portal_alpha + 0.25)
                else:
                    self.portal_alpha = max(0.0, self.portal_alpha - 0.20)

                if active_left is not None and active_right is not None:
                    # Kedua tangan aktif stabil
                    if current_geo == "2D Quad":
                        # 4 Sudut (Telunjuk Kiri, Telunjuk Kanan, Jempol Kanan, Jempol Kiri)
                        polygon_pts = PortalGeometry.build_2d_quad(active_left, active_right)
                    elif current_geo == "2D Bowtie":
                        polygon_pts = PortalGeometry.build_2d_bowtie(active_left, active_right)
                    elif current_geo == "3D Mesh":
                        polygon_pts, mesh_edges = PortalGeometry.build_3d_mesh_data(active_left, active_right)
                    self.last_polygon_pts = polygon_pts
                    self.last_mesh_edges = mesh_edges

                elif active_left is not None:
                    polygon_pts = PortalGeometry.build_single_hand_portal(active_left)
                    self.last_polygon_pts = polygon_pts
                    self.last_mesh_edges = None
                elif active_right is not None:
                    polygon_pts = PortalGeometry.build_single_hand_portal(active_right)
                    self.last_polygon_pts = polygon_pts
                    self.last_mesh_edges = None
                elif self.portal_alpha > 0.02 and self.last_polygon_pts is not None:
                    # Saat tangan baru saja hilang, portal memudar halus (fade-out) tanpa patah-patah
                    polygon_pts = self.last_polygon_pts
                    mesh_edges = self.last_mesh_edges
                else:
                    self.last_polygon_pts = None
                    self.last_mesh_edges = None
                    self.smoother.reset()

                # Compositing Portal Instan (Crop-ROI dengan Smooth Fade)
                output_frame = frame.copy()

                if polygon_pts is not None and len(polygon_pts) >= 3 and self.portal_alpha > 0.02:
                    filter_func = self.get_current_filter_func(curr_time)
                    output_frame = Compositor.composite_portal_roi(
                        output_frame,
                        filter_func,
                        polygon_pts,
                        mesh_edges=mesh_edges,
                        feather_ksize=5,
                        draw_border=self.border_enabled,
                        border_color=(255, 255, 255),
                        opacity=self.portal_alpha
                    )

                # Skeleton Jari (Titik Merah MediaPipe memudar halus bersamaan)
                if self.show_skeleton:
                    HandSkeletonVisualizer.draw_skeleton(output_frame, active_hands, opacity=self.portal_alpha)

                # HUD Bersih di Pojok Kiri Atas
                is_flashing = curr_time < self.flash_until_time
                CleanHUD.draw_hud(
                    output_frame,
                    current_mode=self.modes[self.mode_index],
                    geo_mode=current_geo,
                    switched_flash=is_flashing
                )

                # Tampilkan Window
                cv2.imshow(window_name, output_frame)

                # Kontrol Keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('c') or key == ord('g') or key == ord('m'):
                    self.next_geo_mode()
                elif key == ord('n') or key == 32:
                    self.next_mode()
                elif key == ord('p'):
                    self.prev_mode()
                elif key == ord('s'):
                    self.show_skeleton = not self.show_skeleton
                elif key == ord('b'):
                    self.border_enabled = not self.border_enabled
                elif ord('1') <= key <= ord('9'):
                    idx = key - ord('1')
                    if idx < len(self.modes):
                        self.mode_index = idx
                        self.flash_until_time = time.time() + 0.45
                        print(f"[FILTER BERGANTI] -> {self.modes[self.mode_index]}")

        except KeyboardInterrupt:
            print("\n[INFO] Dihentikan oleh pengguna (Ctrl+C).")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.landmarker.close()
            print("[INFO] Aplikasi ditutup dengan bersih.")


if __name__ == "__main__":
    app = HandPortalApp(camera_index=0, width=1280, height=720)
    app.run()
