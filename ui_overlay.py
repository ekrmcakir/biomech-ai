import os
import time
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class UIOverlay:
    """
    High-end Futuristic Cyberpunk / Medical Glassmorphism HUD Rendering Engine.
    Features:
      - Glassmorphic rounded frosted glass panels
      - TrueType anti-aliased English typography (Apple Helvetica / SF)
      - Apple Watch style circular radial gauges
      - Dual-layer glowing neon skeletal & spinal lines
      - Holographic corner brackets and telemetry status bars
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.theme_color = self.config.get("theme_color", (0, 255, 180))     # Neon Mint / Cyan (BGR)
        self.accent_color = self.config.get("accent_color", (255, 130, 0))   # Neon Blue / Orange (BGR)
        self.warning_color = self.config.get("warning_color", (0, 180, 255)) # Orange / Amber
        self.danger_color = self.config.get("danger_color", (0, 50, 255))    # Red / Coral
        self.success_color = self.config.get("success_color", (0, 255, 120)) # Emerald Green
        self.laser_pos = 100
        self.laser_dir = 1

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

    def draw_rounded_rect(self, img, pt1, pt2, color, radius=12, alpha=0.82, border_color=(60, 75, 90)):
        """Draws frosted glass rounded rectangle."""
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
        """Draws sci-fi targeting corner brackets."""
        cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness, cv2.LINE_AA)

    def draw_radial_gauge(self, img, cx, cy, radius, value, max_val, color, thickness=6):
        """Draws Apple Watch style radial gauge ring."""
        cv2.circle(img, (cx, cy), radius, (35, 42, 50), thickness, cv2.LINE_AA)
        pct = max(0.0, min(1.0, value / max_val))
        sweep_angle = int(pct * 360)
        if sweep_angle > 0:
            cv2.ellipse(img, (cx, cy), (radius, radius), -90, 0, sweep_angle, color, thickness, cv2.LINE_AA)

    def draw_glowing_line(self, img, pt1, pt2, color, core_thickness=2, glow_thickness=7):
        """Draws dual-layer glowing neon line."""
        overlay = img.copy()
        cv2.line(overlay, pt1, pt2, color, glow_thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.40, img, 0.60, 0, img)
        cv2.line(img, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(img, pt1, pt2, color, core_thickness, cv2.LINE_AA)

    def apply_segmentation_effects(self, frame, mask, bg_mode=0):
        """Applies silhouette and laser visual effects."""
        if mask is None or bg_mode == 0:
            return frame

        h, w, _ = frame.shape
        condition = np.stack((mask,) * 3, axis=-1) > 0.45

        if bg_mode == 1:
            dark_bg = cv2.GaussianBlur(frame, (25, 25), 0)
            dark_bg = (dark_bg * 0.22).astype(np.uint8)
            output = np.where(condition, frame, dark_bg)

            mask_uint8 = (mask * 255).astype(np.uint8)
            edges = cv2.Canny(mask_uint8, 80, 180)
            edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
            output[edges_dilated > 0] = self.theme_color
            return output

        elif bg_mode == 2:
            cyber_bg = np.zeros_like(frame)
            cyber_bg[:] = (16, 12, 10)

            for gx in range(0, w, 50):
                cv2.line(cyber_bg, (gx, 0), (gx, h), (35, 25, 20), 1)
            for gy in range(0, h, 50):
                cv2.line(cyber_bg, (0, gy), (w, gy), (35, 25, 20), 1)

            output = np.where(condition, frame, cyber_bg)

            mask_uint8 = (mask * 255).astype(np.uint8)
            edges = cv2.Canny(mask_uint8, 80, 180)
            edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
            output[edges_dilated > 0] = self.theme_color

            self.laser_pos += 10 * self.laser_dir
            if self.laser_pos >= h - 90:
                self.laser_dir = -1
            elif self.laser_pos <= 90:
                self.laser_dir = 1

            laser_y = int(self.laser_pos)
            cv2.line(output, (40, laser_y), (w - 40, laser_y), (0, 255, 255), 3, cv2.LINE_AA)
            cv2.line(output, (50, laser_y - 2), (w - 50, laser_y - 2), (255, 255, 255), 1, cv2.LINE_AA)
            return output

        return frame

    def draw_hud(self, frame, metrics, show_landmarks=True, scan_active=False, bg_mode=0, guided_info=None, voice_enabled=True):
        """Draws complete modern HUD dashboard."""
        h, w, _ = frame.shape
        display = frame.copy()
        view_mode = metrics.get("view_mode", "front")

        # 1. Top Navigation Bar (Floating Pill)
        bar_w = w - 60
        self.draw_rounded_rect(display, (30, 14), (30 + bar_w, 66), (14, 18, 24), radius=16, alpha=0.90, border_color=(45, 60, 75))
        self.draw_corner_brackets(display, 30, 14, 30 + bar_w, 66, self.theme_color, length=10, thickness=2)

        if view_mode == "back":
            badge_title = "BACK PROFILE (SPINE & SHOULDERS)"
            badge_color = (255, 80, 190)
        elif view_mode == "side":
            badge_title = "SIDE PROFILE (KYPHOSIS & NECK)"
            badge_color = (0, 200, 255)
        else:
            badge_title = "FRONT PROFILE (V-TAPER & FITNESS)"
            badge_color = self.theme_color

        r_box_w = 280
        self.draw_rounded_rect(display, (w - 340, 22), (w - 340 + r_box_w, 58), (22, 28, 38), radius=10, alpha=0.92, border_color=badge_color)
        cv2.circle(display, (w - 325, 40), 5, badge_color, -1)

        # 2. Distance & Alignment Badge
        dist_color = self.success_color if "Optimal" in metrics["distance_status"] else self.warning_color
        self.draw_rounded_rect(display, (w // 2 - 190, 74), (w // 2 + 190, 108), (18, 22, 30), radius=12, alpha=0.85, border_color=dist_color)

        # 2.5 Guided Scan Stage Banner & Circular Countdown
        if guided_info and guided_info.get("active"):
            stage_msg = guided_info.get("message", "")
            countdown = guided_info.get("countdown", 0)

            self.draw_rounded_rect(display, (w // 2 - 280, 116), (w // 2 + 280, 160), (32, 22, 50), radius=12, alpha=0.95, border_color=(0, 240, 255))
            self.draw_corner_brackets(display, w // 2 - 280, 116, w // 2 + 280, 160, (0, 255, 255), length=12)

            if countdown > 0:
                cx, cy = w // 2, h // 2
                cv2.circle(display, (cx, cy), 70, (15, 20, 30), -1)
                cv2.circle(display, (cx, cy), 70, (0, 255, 200), 3, cv2.LINE_AA)
                cv2.circle(display, (cx, cy), 78, (0, 255, 200), 1, cv2.LINE_AA)

        # 3. Glowing Skeletal Lines & Telemetry Points
        if show_landmarks and metrics["detected"] and metrics["landmarks"]:
            lm = metrics["landmarks"]

            if view_mode in ["front", "back"]:
                p_ls, p_rs = (lm[11][0], lm[11][1]), (lm[12][0], lm[12][1])
                p_lh, p_rh = (lm[23][0], lm[23][1]), (lm[24][0], lm[24][1])

                sh_color = self.theme_color if metrics["shoulder_drop_angle"] < 2.0 else self.warning_color
                self.draw_glowing_line(display, p_ls, p_rs, sh_color, core_thickness=3, glow_thickness=8)
                self.draw_glowing_line(display, p_lh, p_rh, self.accent_color, core_thickness=3, glow_thickness=8)

                for p in [p_ls, p_rs]:
                    cv2.circle(display, p, 8, sh_color, -1)
                    cv2.circle(display, p, 12, (255, 255, 255), 1, cv2.LINE_AA)
                for p in [p_lh, p_rh]:
                    cv2.circle(display, p, 7, self.accent_color, -1)
                    cv2.circle(display, p, 11, (255, 255, 255), 1, cv2.LINE_AA)

                if view_mode == "back":
                    mid_sh = ((p_ls[0] + p_rs[0]) // 2, (p_ls[1] + p_rs[1]) // 2)
                    mid_h = ((p_lh[0] + p_rh[0]) // 2, (p_lh[1] + p_rh[1]) // 2)
                    self.draw_glowing_line(display, mid_sh, mid_h, (255, 100, 220), core_thickness=3, glow_thickness=9)

                    y_ref = min(p_ls[1], p_rs[1])
                    cv2.line(display, (min(p_ls[0], p_rs[0]) - 35, y_ref),
                             (max(p_ls[0], p_rs[0]) + 35, y_ref), (160, 160, 160), 1, cv2.LINE_AA)
                else:
                    self.draw_glowing_line(display, p_ls, p_lh, (100, 255, 255), core_thickness=2, glow_thickness=6)
                    self.draw_glowing_line(display, p_rs, p_rh, (100, 255, 255), core_thickness=2, glow_thickness=6)

            elif view_mode == "side":
                facing = metrics.get("side_facing", "left")
                ear_idx = 7 if facing == "left" else 8
                sh_idx = 11 if facing == "left" else 12
                hip_idx = 23 if facing == "left" else 24

                p_ear = (lm[ear_idx][0], lm[ear_idx][1])
                p_sh = (lm[sh_idx][0], lm[sh_idx][1])
                p_hip = (lm[hip_idx][0], lm[hip_idx][1])

                self.draw_glowing_line(display, p_ear, p_sh, (0, 240, 255), core_thickness=3, glow_thickness=7)
                self.draw_glowing_line(display, p_sh, p_hip, self.theme_color, core_thickness=3, glow_thickness=7)
                cv2.circle(display, p_ear, 8, (0, 240, 255), -1)
                cv2.circle(display, p_sh, 8, self.theme_color, -1)
                cv2.circle(display, p_hip, 8, self.accent_color, -1)

        # 4. Left Glass Panel
        panel_w = 340
        panel_h = 390
        p_x1, p_y1 = 30, 120
        p_x2, p_y2 = p_x1 + panel_w, p_y1 + panel_h

        self.draw_rounded_rect(display, (p_x1, p_y1), (p_x2, p_y2), (15, 18, 26), radius=16, alpha=0.86, border_color=(50, 65, 80))
        self.draw_corner_brackets(display, p_x1, p_y1, p_x2, p_y2, self.theme_color, length=12)

        # 5. Right Glass Panel
        r_x1 = w - panel_w - 30
        r_y1 = 120
        r_x2 = r_x1 + panel_w
        r_y2 = r_y1 + panel_h

        self.draw_rounded_rect(display, (r_x1, r_y1), (r_x2, r_y2), (15, 18, 26), radius=16, alpha=0.86, border_color=(50, 65, 80))
        self.draw_corner_brackets(display, r_x1, r_y1, r_x2, r_y2, (0, 200, 255), length=12)

        # 6. Bottom Floating Dock
        b_y1 = h - 68
        self.draw_rounded_rect(display, (30, b_y1), (30 + bar_w, h - 14), (14, 18, 24), radius=14, alpha=0.92, border_color=(45, 60, 75))

        # Radial Gauges
        gauge_y = p_y1 + 175
        g1_cx = p_x1 + 85
        g2_cx = p_x1 + 250

        posture_score = metrics.get("posture_score", 80)
        symmetry_score = metrics.get("body_symmetry", 75)

        self.draw_radial_gauge(display, g1_cx, gauge_y, radius=32, value=posture_score, max_val=100, color=self.theme_color, thickness=6)
        self.draw_radial_gauge(display, g2_cx, gauge_y, radius=32, value=symmetry_score, max_val=100, color=(255, 170, 0), thickness=6)

        # High-res anti-aliased Pillow typography
        rgb_img = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        draw = ImageDraw.Draw(pil_img)

        # 1. Header Texts
        draw.text((50, 27), "AI BODY SCANNER", font=self.font_title, fill=(0, 255, 180))
        draw.text((225, 33), "360 BIOMETRICS & POSTURE ENGINE", font=self.font_label, fill=(160, 180, 200))
        draw.text((w - 310, 31), badge_title, font=self.font_badge, fill=(240, 240, 255))

        # 2. Distance Text
        draw.text((w // 2 - 160, 82), f">> {metrics['distance_status']}", font=self.font_sub, fill=(255, 255, 255))

        # 2.5 Guided Scan Text
        if guided_info and guided_info.get("active"):
            stage_msg = guided_info.get("message", "")
            draw.text((w // 2 - 250, 128), stage_msg, font=self.font_title, fill=(0, 255, 255))

            countdown = guided_info.get("countdown", 0)
            if countdown > 0:
                cx, cy = w // 2, h // 2
                draw.text((cx - 15, cy - 25), str(countdown), font=self.font_metric_big, fill=(0, 255, 220))

        # 4. Left Panel Texts
        draw.text((p_x1 + 20, p_y1 + 18), "BIOMETRIC TELEMETRY", font=self.font_title, fill=(0, 255, 180))
        draw.line([(p_x1 + 20, p_y1 + 46), (p_x2 - 20, p_y1 + 46)], fill=(50, 65, 80), width=1)

        if view_mode == "back":
            draw.text((p_x1 + 20, p_y1 + 60), "SHOULDER BALANCE & DROP", font=self.font_label, fill=(160, 175, 190))
            drop_txt = metrics.get("shoulder_drop_status", "Balanced")
            drop_col = (0, 255, 120) if "Balanced" in drop_txt else (255, 180, 0)
            draw.text((p_x1 + 20, p_y1 + 80), drop_txt[:26], font=self.font_sub, fill=drop_col)

            draw.text((p_x1 + 20, p_y1 + 108), f"Delta: ~{metrics['shoulder_drop_cm']} cm ({metrics['shoulder_drop_angle']}°)",
                      font=self.font_label, fill=(200, 220, 230))

            draw.text((p_x1 + 20, p_y1 + 225), "SPINE ALIGNMENT (SCOLIOSIS)", font=self.font_label, fill=(160, 175, 190))
            spine_txt = metrics.get("spine_status", "Straight")
            spine_col = (0, 255, 120) if "Straight" in spine_txt else (255, 180, 0)
            draw.text((p_x1 + 20, p_y1 + 245), spine_txt[:26], font=self.font_sub, fill=spine_col)

            draw.text((p_x1 + 20, p_y1 + 285), f"Pelvis Tilt: {metrics.get('pelvis_status', 'Balanced')}", font=self.font_label, fill=(180, 195, 210))
            draw.text((p_x1 + 20, p_y1 + 315), f"Shoulder: {metrics['shoulder_width']} px  |  Hips: {metrics['hip_width']} px", font=self.font_label, fill=(180, 195, 210))

        elif view_mode == "side":
            draw.text((p_x1 + 20, p_y1 + 60), "THORACIC KYPHOSIS ANGLE", font=self.font_label, fill=(160, 175, 190))
            draw.text((p_x1 + 20, p_y1 + 80), f"{metrics.get('kyphosis_angle', 170.0):.1f}° - {metrics.get('kyphosis_status', '')}",
                      font=self.font_sub, fill=(0, 255, 180))

            draw.text((p_x1 + 20, p_y1 + 108), "FORWARD HEAD (TEXT NECK)", font=self.font_label, fill=(160, 175, 190))
            neck_col = (0, 255, 120) if metrics.get("head_tilt_angle", 10.0) < 14 else (255, 180, 0)
            draw.text((p_x1 + 20, p_y1 + 128), f"{metrics.get('head_tilt_angle', 10.0):.1f}° - {metrics.get('neck_status', '')}",
                      font=self.font_sub, fill=neck_col)

            draw.text((p_x1 + 20, p_y1 + 245), "Spine Posture & Alignment", font=self.font_label, fill=(160, 175, 190))
            draw.text((p_x1 + 20, p_y1 + 285), "Rx: Chin-Tucks & Face-Pulls", font=self.font_sub, fill=(0, 220, 255))
            draw.text((p_x1 + 20, p_y1 + 325), f"Torso Length: {metrics['torso_height']} px", font=self.font_label, fill=(180, 195, 210))

        else:
            # Front Profile
            draw.text((p_x1 + 20, p_y1 + 55), "V-TAPER RATIO (SHOULDER/WAIST)", font=self.font_label, fill=(160, 175, 190))
            vt = metrics["v_taper_ratio"]
            vt_col = (0, 255, 120) if vt >= 1.40 else ((0, 240, 255) if vt >= 1.25 else (255, 180, 0))
            draw.text((p_x1 + 20, p_y1 + 75), f"{vt:.2f}", font=self.font_metric_big, fill=vt_col)
            draw.text((p_x1 + 90, p_y1 + 86), "Aesthetic V-Shape" if vt >= 1.40 else "Balanced Athletic", font=self.font_label, fill=(180, 200, 210))

            draw.text((g1_cx - 28, gauge_y + 40), f"Posture: {posture_score}%", font=self.font_label, fill=(0, 255, 180))
            draw.text((g2_cx - 30, gauge_y + 40), f"Symmetry: {symmetry_score}%", font=self.font_label, fill=(255, 180, 0))

            draw.text((g1_cx - 10, gauge_y - 8), str(posture_score), font=self.font_label, fill=(255, 255, 255))
            draw.text((g2_cx - 10, gauge_y - 8), str(symmetry_score), font=self.font_label, fill=(255, 255, 255))

            draw.text((p_x1 + 20, p_y1 + 250), f"Shoulders: {metrics.get('shoulder_drop_status', 'Balanced')[:24]}", font=self.font_label, fill=(200, 220, 230))
            draw.text((p_x1 + 20, p_y1 + 285), f"Shoulders: {metrics['shoulder_width']} px   Hips: {metrics['hip_width']} px", font=self.font_label, fill=(180, 195, 210))
            draw.text((p_x1 + 20, p_y1 + 315), f"Torso Length: {metrics['torso_height']} px", font=self.font_label, fill=(180, 195, 210))

        # 5. Right Panel Texts
        draw.text((r_x1 + 20, r_y1 + 18), "BODY COMPOSITION", font=self.font_title, fill=(0, 240, 255))
        draw.line([(r_x1 + 20, r_y1 + 46), (r_x2 - 20, r_y1 + 46)], fill=(50, 65, 80), width=1)

        low_fat, high_fat = metrics["estimated_body_fat_range"]
        fat_text = f"{low_fat}% - {high_fat}%" if low_fat > 0 else "Scanning..."
        draw.text((r_x1 + 20, r_y1 + 60), "ESTIMATED BODY FAT RANGE", font=self.font_label, fill=(160, 200, 220))
        draw.text((r_x1 + 20, r_y1 + 82), fat_text, font=self.font_metric_big, fill=(255, 255, 255))

        draw.text((r_x1 + 20, r_y1 + 150), "SOMATOTYPE / BUILD", font=self.font_label, fill=(160, 175, 190))
        draw.text((r_x1 + 20, r_y1 + 172), metrics["body_type"][:26], font=self.font_sub, fill=(255, 180, 0))

        draw.text((r_x1 + 20, r_y1 + 215), "FITNESS CATEGORY", font=self.font_label, fill=(160, 175, 190))
        draw.text((r_x1 + 20, r_y1 + 237), metrics["fitness_level"], font=self.font_sub, fill=(0, 255, 180))

        bg_names = ["Normal Camera", "Dark Studio Mode", "Holographic Cyber Laser"]
        draw.text((r_x1 + 20, r_y1 + 280), "SILHOUETTE MODE [B]", font=self.font_label, fill=(160, 175, 190))
        draw.text((r_x1 + 20, r_y1 + 302), bg_names[bg_mode % 3], font=self.font_sub, fill=(0, 230, 255))

        # 6. Bottom Dock Texts
        if metrics.get("recommendation"):
            draw.text((45, b_y1 + 12), f"💡 COACH TIP: {metrics['recommendation'][:95]}", font=self.font_label, fill=(220, 245, 235))

        voice_status = "Voice: ON" if voice_enabled else "Voice: OFF"
        controls_txt = f"[T / SPACE] Voice 360° Scan   |   [V] {voice_status}   |   [F] Face Mode   |   [B] Cyber Laser   |   [S] Save   |   [Q] Quit"
        draw.text((45, b_y1 + 34), controls_txt, font=self.font_tiny, fill=(150, 175, 195))

        final_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return final_frame
