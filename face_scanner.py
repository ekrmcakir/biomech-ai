import os
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

try:
    import mediapipe as mp
    try:
        mp_face_mesh = mp.solutions.face_mesh
    except AttributeError:
        from mediapipe.python.solutions import face_mesh as mp_face_mesh
    HAS_FACEMESH = True
except (ImportError, AttributeError):
    HAS_FACEMESH = False


class FaceScanner:
    """
    Analyzes facial geometry (MediaPipe 468 3D Mesh);
    computes Jawline taper sharpness, cheek fullness (Buccal Fat), fWHR,
    and estimates body fat percentage exclusively from facial anthropometrics.
    """

    JAWLINE_INDICES = [
        234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
        377, 400, 378, 379, 365, 397, 288, 361, 323, 454
    ]

    def __init__(self, min_detection_confidence=0.6, min_tracking_confidence=0.6):
        self.has_facemesh = HAS_FACEMESH
        if self.has_facemesh:
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        else:
            self.face_mesh = None

        self.font_path = "/System/Library/Fonts/Helvetica.ttc"
        if not os.path.exists(self.font_path):
            self.font_path = "/System/Library/Fonts/SFNS.ttf"

        try:
            self.font_title = ImageFont.truetype(self.font_path, 19)
            self.font_badge = ImageFont.truetype(self.font_path, 13)
            self.font_metric_big = ImageFont.truetype(self.font_path, 28)
            self.font_sub = ImageFont.truetype(self.font_path, 14)
            self.font_label = ImageFont.truetype(self.font_path, 12)
            self.font_tiny = ImageFont.truetype(self.font_path, 11)
        except Exception:
            self.font_title = ImageFont.load_default()
            self.font_badge = ImageFont.load_default()
            self.font_metric_big = ImageFont.load_default()
            self.font_sub = ImageFont.load_default()
            self.font_label = ImageFont.load_default()
            self.font_tiny = ImageFont.load_default()

    def calculate_distance(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def calculate_angle(self, p_left, p_vertex, p_right):
        v1 = (p_left[0] - p_vertex[0], p_left[1] - p_vertex[1])
        v2 = (p_right[0] - p_vertex[0], p_right[1] - p_vertex[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.hypot(v1[0], v1[1])
        mag2 = math.hypot(v2[0], v2[1])
        if mag1 * mag2 == 0:
            return 90.0
        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))

    def draw_rounded_rect(self, img, pt1, pt2, color, radius=12, alpha=0.84, border_color=(60, 75, 90)):
        x1, y1 = pt1
        x2, y2 = pt2
        overlay = img.copy()
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        if border_color:
            cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), border_color, 1, cv2.LINE_AA)
            cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), border_color, 1, cv2.LINE_AA)
            cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), border_color, 1, cv2.LINE_AA)
            cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), border_color, 1, cv2.LINE_AA)

    def draw_corner_brackets(self, img, x1, y1, x2, y2, color, length=12, thickness=2):
        cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness, cv2.LINE_AA)

    def draw_radial_gauge(self, img, cx, cy, radius, value, max_val, color, thickness=6):
        cv2.circle(img, (cx, cy), radius, (35, 42, 50), thickness, cv2.LINE_AA)
        pct = max(0.0, min(1.0, value / max_val))
        sweep = int(pct * 360)
        if sweep > 0:
            cv2.ellipse(img, (cx, cy), (radius, radius), -90, 0, sweep, color, thickness, cv2.LINE_AA)

    def process_frame(self, frame_bgr):
        h, w, _ = frame_bgr.shape
        metrics = {
            "detected": False,
            "landmarks": None,
            "jawline_points": [],
            "face_width": 0,
            "face_height": 0,
            "jaw_width": 0,
            "cheek_width": 0,
            "jaw_angle": 0.0,
            "cheek_to_jaw_ratio": 1.0,
            "fwhr": 0.0,
            "jawline_sharpness": 0,
            "facial_fat_range": (0, 0),
            "facial_adiposity_status": "Computing...",
            "face_shape": "Unknown",
            "facial_symmetry": 0,
            "status_message": "Please center your face in camera",
            "recommendation": ""
        }

        if not self.has_facemesh:
            metrics["status_message"] = "MediaPipe FaceMesh is not installed!"
            return metrics

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            return metrics

        face_landmarks = results.multi_face_landmarks[0]
        metrics["detected"] = True

        coords = {}
        for idx, lm in enumerate(face_landmarks.landmark):
            coords[idx] = (int(lm.x * w), int(lm.y * h), lm.z)
        metrics["landmarks"] = coords

        jawline_pts = [coords[idx][:2] for idx in self.JAWLINE_INDICES if idx in coords]
        metrics["jawline_points"] = jawline_pts

        chin = coords[152][:2]
        forehead = coords[10][:2]
        left_cheek = coords[234][:2]
        right_cheek = coords[454][:2]
        left_jaw = coords[132][:2]
        right_jaw = coords[361][:2]
        left_eye = coords[33][:2]
        right_eye = coords[263][:2]
        nose = coords[1][:2]

        face_width = self.calculate_distance(left_cheek, right_cheek)
        face_height = self.calculate_distance(forehead, chin)
        jaw_width = self.calculate_distance(left_jaw, right_jaw)

        metrics["face_width"] = round(face_width, 1)
        metrics["face_height"] = round(face_height, 1)
        metrics["jaw_width"] = round(jaw_width, 1)
        metrics["cheek_width"] = round(face_width, 1)

        if face_height < h * 0.28:
            metrics["status_message"] = "Please move closer to camera"
        elif face_height > h * 0.85:
            metrics["status_message"] = "Please step back slightly"
        else:
            metrics["status_message"] = "Face Alignment Optimal ✓"

        jaw_angle = self.calculate_angle(left_jaw, chin, right_jaw)
        metrics["jaw_angle"] = round(jaw_angle, 1)

        cjr = face_width / max(1.0, jaw_width)
        metrics["cheek_to_jaw_ratio"] = round(cjr, 2)

        mid_brow_y = (left_eye[1] + right_eye[1]) / 2.0
        upper_face_height = max(1.0, abs(chin[1] - mid_brow_y))
        fwhr = face_width / upper_face_height
        metrics["fwhr"] = round(fwhr, 2)

        left_dist = self.calculate_distance(nose, left_cheek)
        right_dist = self.calculate_distance(nose, right_cheek)
        sym_diff = abs(left_dist - right_dist) / max(1.0, (left_dist + right_dist) / 2.0)
        metrics["facial_symmetry"] = max(0, min(100, int(100 - (sym_diff * 140))))

        angle_penalty = max(0.0, (jaw_angle - 68.0) * 1.5)
        sharpness = max(20, min(98, int(98 - angle_penalty)))
        metrics["jawline_sharpness"] = sharpness

        if sharpness >= 82 and jaw_angle <= 78.0:
            metrics["facial_fat_range"] = (9, 13)
            metrics["facial_adiposity_status"] = "Lean & Chiseled Jawline"
            metrics["face_shape"] = "Chiseled V-Line"
            metrics["recommendation"] = "Sharp bone structure with high mandibular definition. Maintain hydration to preserve skin elasticity."
        elif sharpness >= 68 and jaw_angle <= 86.0:
            metrics["facial_fat_range"] = (13, 16)
            metrics["facial_adiposity_status"] = "Athletic & Balanced Features"
            metrics["face_shape"] = "Balanced Oval / Athletic"
            metrics["recommendation"] = "Healthy and balanced facial composition. Jawline is defined with minimal subcutaneous facial fat."
        elif sharpness >= 52:
            metrics["facial_fat_range"] = (16, 20)
            metrics["facial_adiposity_status"] = "Moderate Fullness / Soft Contour"
            metrics["face_shape"] = "Oval Contour"
            metrics["recommendation"] = "Standard facial adiposity. Reducing dietary sodium and optimizing water intake can help eliminate facial puffiness."
        else:
            metrics["facial_fat_range"] = (21, 26)
            metrics["facial_adiposity_status"] = "Heavy / Rounded Buccal Volume"
            metrics["face_shape"] = "Round / Full Cheek"
            metrics["recommendation"] = "Noticeable buccal fat volume. Achieving an overall caloric deficit will rapidly sharpen mandibular definition."

        return metrics

    def draw_face_hud(self, frame, metrics, show_mesh=True):
        """Draws high-tech frosted glass HUD for facial analysis."""
        h, w, _ = frame.shape
        display = frame.copy()

        # Top Navigation Bar
        bar_w = w - 60
        self.draw_rounded_rect(display, (30, 14), (30 + bar_w, 66), (14, 18, 24), radius=16, alpha=0.90, border_color=(45, 60, 75))
        self.draw_corner_brackets(display, 30, 14, 30 + bar_w, 66, (0, 230, 255), length=10)

        # Mode Badge
        self.draw_rounded_rect(display, (w - 300, 22), (w - 50, 58), (22, 28, 38), radius=10, alpha=0.92, border_color=(0, 230, 255))
        cv2.circle(display, (w - 285, 40), 5, (0, 230, 255), -1)

        # Status & Distance Message
        dist_color = (0, 255, 120) if "Optimal" in metrics["status_message"] else (0, 180, 255)
        self.draw_rounded_rect(display, (w // 2 - 180, 74), (w // 2 + 180, 108), (18, 22, 30), radius=12, alpha=0.85, border_color=dist_color)

        # Face & Jawline Neon Glow Path
        if metrics["detected"] and metrics["landmarks"]:
            coords = metrics["landmarks"]
            jaw_pts = metrics["jawline_points"]

            if len(jaw_pts) > 1:
                pts_arr = np.array(jaw_pts, np.int32).reshape((-1, 1, 2))
                overlay = display.copy()
                cv2.polylines(overlay, [pts_arr], False, (0, 230, 255), 7, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.45, display, 0.55, 0, display)
                cv2.polylines(display, [pts_arr], False, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.polylines(display, [pts_arr], False, (0, 210, 255), 3, cv2.LINE_AA)

                for pt in jaw_pts[::2]:
                    cv2.circle(display, pt, 4, (0, 255, 200), -1)

            chin = coords[152][:2]
            left_jaw = coords[132][:2]
            right_jaw = coords[361][:2]
            left_cheek = coords[234][:2]
            right_cheek = coords[454][:2]

            cv2.circle(display, chin, 8, (0, 120, 255), -1)
            cv2.circle(display, chin, 12, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(display, left_jaw, 6, (0, 255, 120), -1)
            cv2.circle(display, right_jaw, 6, (0, 255, 120), -1)

            cv2.line(display, left_jaw, chin, (0, 255, 200), 2, cv2.LINE_AA)
            cv2.line(display, right_jaw, chin, (0, 255, 200), 2, cv2.LINE_AA)
            cv2.line(display, left_cheek, right_cheek, (255, 160, 0), 1, cv2.LINE_AA)

        # Left Glass Panel
        panel_w = 340
        panel_h = 390
        p_x1, p_y1 = 30, 120
        p_x2, p_y2 = p_x1 + panel_w, p_y1 + panel_h

        self.draw_rounded_rect(display, (p_x1, p_y1), (p_x2, p_y2), (15, 18, 26), radius=16, alpha=0.86, border_color=(50, 65, 80))
        self.draw_corner_brackets(display, p_x1, p_y1, p_x2, p_y2, (0, 230, 255), length=12)

        # Right Glass Panel
        r_x1 = w - panel_w - 30
        r_y1 = 120
        r_x2 = r_x1 + panel_w
        r_y2 = r_y1 + panel_h

        self.draw_rounded_rect(display, (r_x1, r_y1), (r_x2, r_y2), (15, 18, 26), radius=16, alpha=0.86, border_color=(50, 65, 80))
        self.draw_corner_brackets(display, r_x1, r_y1, r_x2, r_y2, (0, 200, 255), length=12)

        # Bottom Floating Dock
        b_y1 = h - 68
        self.draw_rounded_rect(display, (30, b_y1), (30 + bar_w, h - 14), (14, 18, 24), radius=14, alpha=0.92, border_color=(45, 60, 75))

        # Radial Gauges (Jawline Sharpness & Facial Symmetry)
        gauge_y = p_y1 + 175
        g1_cx = p_x1 + 85
        g2_cx = p_x1 + 250

        sharpness = metrics.get("jawline_sharpness", 75)
        symmetry = metrics.get("facial_symmetry", 80)

        self.draw_radial_gauge(display, g1_cx, gauge_y, radius=32, value=sharpness, max_val=100, color=(0, 230, 255), thickness=6)
        self.draw_radial_gauge(display, g2_cx, gauge_y, radius=32, value=symmetry, max_val=100, color=(255, 170, 0), thickness=6)

        # Anti-aliased Pillow typography
        rgb_img = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        draw = ImageDraw.Draw(pil_img)

        draw.text((50, 27), "AI FACIAL ADIPOSITY", font=self.font_title, fill=(0, 230, 255))
        draw.text((255, 33), "JAWLINE & FACE FAT ENGINE", font=self.font_label, fill=(160, 180, 200))
        draw.text((w - 270, 31), "FACE SCAN MODE", font=self.font_badge, fill=(240, 240, 255))

        draw.text((w // 2 - 150, 82), f">> {metrics['status_message']}", font=self.font_sub, fill=(255, 255, 255))

        # Left Panel Texts
        draw.text((p_x1 + 20, p_y1 + 18), "FACIAL ANATOMY & JAWLINE", font=self.font_title, fill=(0, 230, 255))
        draw.line([(p_x1 + 20, p_y1 + 46), (p_x2 - 20, p_y1 + 46)], fill=(50, 65, 80), width=1)

        draw.text((p_x1 + 20, p_y1 + 55), "JAWLINE V-ANGLE (TAPER)", font=self.font_label, fill=(160, 175, 190))
        jaw_ang = metrics["jaw_angle"]
        ang_col = (0, 255, 120) if jaw_ang <= 78.0 else ((0, 240, 255) if jaw_ang <= 86.0 else (255, 180, 0))
        draw.text((p_x1 + 20, p_y1 + 75), f"{jaw_ang:.1f}°", font=self.font_metric_big, fill=ang_col)
        draw.text((p_x1 + 105, p_y1 + 86), "Sharp & Defined" if jaw_ang <= 78.0 else "Balanced Jaw", font=self.font_label, fill=(180, 200, 210))

        draw.text((g1_cx - 32, gauge_y + 40), f"Sharpness: %{sharpness}", font=self.font_label, fill=(0, 230, 255))
        draw.text((g2_cx - 30, gauge_y + 40), f"Symmetry: %{symmetry}", font=self.font_label, fill=(255, 180, 0))
        draw.text((g1_cx - 10, gauge_y - 8), str(sharpness), font=self.font_label, fill=(255, 255, 255))
        draw.text((g2_cx - 10, gauge_y - 8), str(symmetry), font=self.font_label, fill=(255, 255, 255))

        draw.text((p_x1 + 20, p_y1 + 250), f"Cheek / Jaw Ratio (CJR): {metrics['cheek_to_jaw_ratio']}", font=self.font_label, fill=(200, 220, 230))
        draw.text((p_x1 + 20, p_y1 + 280), f"Facial Ratio (fWHR): {metrics['fwhr']}", font=self.font_label, fill=(180, 195, 210))
        draw.text((p_x1 + 20, p_y1 + 310), f"Face Dimensions: {metrics['face_height']}x{metrics['face_width']} px", font=self.font_label, fill=(180, 195, 210))

        # Right Panel Texts
        draw.text((r_x1 + 20, r_y1 + 18), "FACIAL BODY FAT %", font=self.font_title, fill=(0, 200, 255))
        draw.line([(r_x1 + 20, r_y1 + 46), (r_x2 - 20, r_y1 + 46)], fill=(50, 65, 80), width=1)

        low_fat, high_fat = metrics["facial_fat_range"]
        fat_text = f"%{low_fat} - %{high_fat}" if low_fat > 0 else "Scanning..."
        draw.text((r_x1 + 20, r_y1 + 60), "ESTIMATED BODY FAT RANGE", font=self.font_label, fill=(160, 200, 220))
        draw.text((r_x1 + 20, r_y1 + 82), fat_text, font=self.font_metric_big, fill=(255, 255, 255))

        draw.text((r_x1 + 20, r_y1 + 150), "ADIPOSITY STATUS", font=self.font_label, fill=(160, 175, 190))
        draw.text((r_x1 + 20, r_y1 + 172), metrics["facial_adiposity_status"][:26], font=self.font_sub, fill=(255, 180, 0))

        draw.text((r_x1 + 20, r_y1 + 215), "JAWLINE STRUCTURE", font=self.font_label, fill=(160, 175, 190))
        draw.text((r_x1 + 20, r_y1 + 237), metrics["face_shape"], font=self.font_sub, fill=(0, 255, 180))

        draw.text((r_x1 + 20, r_y1 + 280), "ANALYSIS MODEL", font=self.font_label, fill=(160, 175, 190))
        draw.text((r_x1 + 20, r_y1 + 302), "MediaPipe 468 3D Mesh", font=self.font_sub, fill=(0, 230, 255))

        # Bottom Dock Texts
        if metrics.get("recommendation"):
            draw.text((45, b_y1 + 12), f"💡 COACH TIP: {metrics['recommendation'][:95]}", font=self.font_label, fill=(220, 245, 235))

        controls_txt = "[SPACE] Face Report   |   [F] Switch to Body Mode   |   [S] Save   |   [Q] Quit"
        draw.text((45, b_y1 + 34), controls_txt, font=self.font_tiny, fill=(150, 175, 195))

        final_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return final_frame
