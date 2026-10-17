import math
import numpy as np
import cv2

try:
    import mediapipe as mp
    try:
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
    except AttributeError:
        from mediapipe.python.solutions import pose as mp_pose
        from mediapipe.python.solutions import drawing_utils as mp_drawing
        from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
    HAS_MEDIAPIPE = True
except (ImportError, AttributeError):
    HAS_MEDIAPIPE = False


class BodyScanner:
    """
    Real-time body tracking, 360° posture analysis, and anthropometric modeling.
    Modes:
      - Front Profile : V-Taper Ratio, Shoulder Width, Body Fat Estimation
      - Side Profile  : Forward Head (Text Neck), Thoracic Kyphosis (Hunchback)
      - Back Profile  : Left/Right Shoulder Drop (cm/deg), Lateral Spine Deviation (Scoliosis indicator)
    """

    def __init__(self, min_detection_confidence=0.6, min_tracking_confidence=0.6):
        self.has_mediapipe = HAS_MEDIAPIPE
        if self.has_mediapipe:
            self.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            self.pose = None

    def calculate_distance(self, p1, p2):
        """Calculates 2D Euclidean distance between two points."""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def calculate_angle(self, a, b, c):
        """Calculates angle at vertex b in degrees."""
        radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
        angle = abs(radians * 180.0 / math.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def process_frame(self, frame_bgr, user_height_cm=170):
        """Processes webcam frame and computes biometric & postural telemetry."""
        h, w, _ = frame_bgr.shape
        metrics = {
            "detected": False,
            "landmarks": None,
            "raw_landmarks": None,
            "segmentation_mask": None,
            "view_mode": "front",             # 'front', 'side', 'back'
            "side_facing": "none",
            
            # Shoulder Drop & Symmetry
            "shoulder_drop_status": "Balanced & Symmetrical ✓",
            "shoulder_drop_side": "none",     # 'Left', 'Right', 'none'
            "shoulder_drop_angle": 0.0,
            "shoulder_drop_cm": 0.0,
            
            # Front Profile
            "shoulder_width": 0,
            "hip_width": 0,
            "torso_height": 0,
            "v_taper_ratio": 0.0,
            "posture_score": 0,
            "body_symmetry": 0,
            
            # Side Profile (Kyphosis / Neck)
            "head_tilt_angle": 0.0,
            "neck_status": "Unknown",
            "kyphosis_angle": 0.0,
            "kyphosis_status": "Unknown",
            
            # Back Profile (Spine / Scoliosis / Pelvis)
            "spine_lateral_offset_cm": 0.0,
            "spine_status": "Spine Line Straight & Optimal ✓",
            "pelvis_tilt_angle": 0.0,
            "pelvis_status": "Balanced",
            
            # Global Evaluation
            "distance_status": "Please step into camera frame",
            "body_type": "Detecting...",
            "body_shape": "Unknown",
            "estimated_body_fat_range": (0, 0),
            "muscle_score": 0,
            "fitness_level": "Initializing",
            "recommendation": ""
        }

        if not self.has_mediapipe:
            metrics["distance_status"] = "MediaPipe is not installed!"
            return metrics

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.pose.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if results.segmentation_mask is not None:
            metrics["segmentation_mask"] = results.segmentation_mask

        if not results.pose_landmarks:
            return metrics

        lm = results.pose_landmarks.landmark
        metrics["detected"] = True
        metrics["raw_landmarks"] = results.pose_landmarks

        coords = {}
        for idx, landmark in enumerate(lm):
            coords[idx] = (int(landmark.x * w), int(landmark.y * h), landmark.visibility, landmark.z)
        metrics["landmarks"] = coords

        nose = coords[0]
        left_ear, right_ear = coords[7], coords[8]
        left_shoulder, right_shoulder = coords[11], coords[12]
        left_hip, right_hip = coords[23], coords[24]

        shoulder_width = self.calculate_distance(left_shoulder, right_shoulder)
        hip_width = self.calculate_distance(left_hip, right_hip)

        mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
        mid_hip = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
        torso_height = self.calculate_distance(mid_shoulder, mid_hip)

        metrics["shoulder_width"] = round(shoulder_width, 1)
        metrics["hip_width"] = round(hip_width, 1)
        metrics["torso_height"] = round(torso_height, 1)

        # Scale pixels to real centimeters based on user height
        torso_real_cm = max(40.0, user_height_cm * 0.30)
        px_per_cm = max(0.5, torso_height / torso_real_cm)

        if torso_height < h * 0.18:
            metrics["distance_status"] = "Please step closer to camera"
        elif torso_height > h * 0.70:
            metrics["distance_status"] = "Please step back a bit"
        else:
            metrics["distance_status"] = "Distance & Alignment Optimal ✓"

        # --- VIEW ORIENTATION CLASSIFICATION (Front, Side, Back) ---
        shoulder_span_x = abs(left_shoulder[0] - right_shoulder[0])
        is_side = (shoulder_span_x < torso_height * 0.38)

        # Back view detection based on facial visibility and 3D Z depth
        mid_sh_z = (left_shoulder[3] + right_shoulder[3]) / 2.0
        is_back = (not is_side) and (nose[2] < 0.40 or nose[3] > mid_sh_z + 0.04)

        if is_side:
            metrics["view_mode"] = "side"
        elif is_back:
            metrics["view_mode"] = "back"
        else:
            metrics["view_mode"] = "front"

        # --- SHOULDER DROP / ASYMMETRY CALCULATION ---
        dy_shoulder = left_shoulder[1] - right_shoulder[1]
        dx_shoulder = max(1, abs(left_shoulder[0] - right_shoulder[0]))
        sh_drop_angle = math.degrees(math.atan2(abs(dy_shoulder), dx_shoulder))
        sh_drop_cm = abs(dy_shoulder) / px_per_cm

        metrics["shoulder_drop_angle"] = round(sh_drop_angle, 1)
        metrics["shoulder_drop_cm"] = round(sh_drop_cm, 1)

        dropped_side = "Left" if dy_shoulder > 0 else "Right"

        if sh_drop_angle < 1.6:
            metrics["shoulder_drop_status"] = "Balanced & Symmetrical ✓"
            metrics["shoulder_drop_side"] = "none"
        elif sh_drop_angle < 3.8:
            metrics["shoulder_drop_status"] = f"Mild {dropped_side} Shoulder Drop ({sh_drop_cm:.1f} cm / {sh_drop_angle:.1f}°)"
            metrics["shoulder_drop_side"] = dropped_side
        else:
            metrics["shoulder_drop_status"] = f"Noticeable {dropped_side} Shoulder Drop ({sh_drop_cm:.1f} cm / {sh_drop_angle:.1f}° ⚠)"
            metrics["shoulder_drop_side"] = dropped_side

        # --- PROFILE-SPECIFIC METRICS ---
        if metrics["view_mode"] == "back":
            dx_spine = abs(mid_shoulder[0] - mid_hip[0])
            spine_offset_cm = dx_spine / px_per_cm
            metrics["spine_lateral_offset_cm"] = round(spine_offset_cm, 1)

            if spine_offset_cm < 1.4:
                metrics["spine_status"] = "Spine Line Straight & Optimal ✓"
            elif spine_offset_cm < 3.0:
                metrics["spine_status"] = f"Mild Lateral Spine Deviation ({spine_offset_cm:.1f} cm)"
            else:
                metrics["spine_status"] = f"Noticeable Spine Curvature ({spine_offset_cm:.1f} cm ⚠)"

            dy_hip = abs(left_hip[1] - right_hip[1])
            dx_hip = max(1, abs(left_hip[0] - right_hip[0]))
            pelvis_angle = math.degrees(math.atan2(dy_hip, dx_hip))
            metrics["pelvis_tilt_angle"] = round(pelvis_angle, 1)
            metrics["pelvis_status"] = "Balanced" if pelvis_angle < 2.0 else f"Tilted ({pelvis_angle:.1f}°)"

            back_posture_score = max(0, min(100, int(100 - (sh_drop_angle * 8 + spine_offset_cm * 8))))
            metrics["posture_score"] = back_posture_score
            metrics["body_symmetry"] = max(0, min(100, int(100 - sh_drop_angle * 12)))

            metrics["body_type"] = f"Back Profile // {metrics['shoulder_drop_status']}"
            metrics["body_shape"] = metrics["spine_status"]
            metrics["recommendation"] = "Perform unilateral pulling exercises (Single-Arm Dumbbell Row) to balance shoulder symmetry."

        elif metrics["view_mode"] == "side":
            facing = "left" if left_ear[2] > right_ear[2] else "right"
            metrics["side_facing"] = facing
            active_ear = left_ear if facing == "left" else right_ear
            active_shoulder = left_shoulder if facing == "left" else right_shoulder
            active_hip = left_hip if facing == "left" else right_hip

            dx_head = abs(active_ear[0] - active_shoulder[0])
            dy_head = max(1, abs(active_ear[1] - active_shoulder[1]))
            head_angle = math.degrees(math.atan2(dx_head, dy_head))
            metrics["head_tilt_angle"] = round(head_angle, 1)

            if head_angle < 12.0:
                metrics["neck_status"] = "Optimal / Upright Neck ✓"
            elif head_angle < 22.0:
                metrics["neck_status"] = "Mild Forward Head (Text Neck)"
            else:
                metrics["neck_status"] = "Severe Forward Head Posture ⚠"

            kyphosis_angle = self.calculate_angle(active_ear, active_shoulder, active_hip)
            metrics["kyphosis_angle"] = round(kyphosis_angle, 1)

            if kyphosis_angle >= 165.0:
                metrics["kyphosis_status"] = "Optimal Upright Spine ✓"
                metrics["posture_score"] = 95
            elif kyphosis_angle >= 150.0:
                metrics["kyphosis_status"] = "Mild Thoracic Kyphosis"
                metrics["posture_score"] = 78
            else:
                metrics["kyphosis_status"] = "Thoracic Kyphosis (Hunchback) ⚠"
                metrics["posture_score"] = 55

            metrics["body_type"] = f"Side Profile // {metrics['kyphosis_status']}"
            metrics["body_shape"] = f"Neck: {metrics['neck_status']}"
            metrics["recommendation"] = "Perform Chin-Tucks and Face-Pulls daily to strengthen deep cervical flexors and upper back."

        else:
            # Front Profile
            v_taper = (shoulder_width / hip_width) if hip_width > 0 else 1.0
            metrics["v_taper_ratio"] = round(v_taper, 2)

            torso_dx = abs(mid_shoulder[0] - mid_hip[0])
            torso_dy = max(1, abs(mid_shoulder[1] - mid_hip[1]))
            torso_lean_angle = math.degrees(math.atan2(torso_dx, torso_dy))
            metrics["posture_score"] = max(0, min(100, int(100 - (torso_lean_angle * 5 + sh_drop_angle * 5))))
            metrics["body_symmetry"] = max(0, min(100, int(100 - sh_drop_angle * 10)))

            if v_taper >= 1.45:
                metrics["body_type"] = "Muscular & V-Shape (Inverted Triangle)"
                metrics["body_shape"] = "V-Taper (Athletic/Mesomorph)"
                metrics["fitness_level"] = "Advanced Athletic / Fit"
                metrics["estimated_body_fat_range"] = (10, 14)
                metrics["muscle_score"] = min(98, int(75 + (v_taper - 1.45) * 50))
                metrics["recommendation"] = "Aesthetic V-Taper form! Turn around to check spine symmetry and shoulder balance."
            elif v_taper >= 1.30:
                metrics["body_type"] = "Athletic & Balanced Build"
                metrics["body_shape"] = "Athletic Frame"
                metrics["fitness_level"] = "Intermediate Athletic Fit"
                metrics["estimated_body_fat_range"] = (13, 17)
                metrics["muscle_score"] = int(65 + (v_taper - 1.30) * 60)
                metrics["recommendation"] = "Balanced upper torso. Focus on lateral deltoids and latissimus dorsi to enhance V-Taper."
            elif v_taper >= 1.15:
                metrics["body_type"] = "Lean & Agile Build (Ectomorph)"
                metrics["body_shape"] = "H-Shape (Lean / Agile)"
                metrics["fitness_level"] = "Standard Fit / Agile"
                metrics["estimated_body_fat_range"] = (16, 21)
                metrics["muscle_score"] = int(50 + (v_taper - 1.15) * 60)
                metrics["recommendation"] = "Prioritize hypertrophy and progressive overload to build shoulder and back volume."
            else:
                metrics["body_type"] = "Heavy Frame / Bulking Phase (Endomorph)"
                metrics["body_shape"] = "Oval / Heavy Frame"
                metrics["fitness_level"] = "Bulking / Development Phase"
                metrics["estimated_body_fat_range"] = (21, 26)
                metrics["muscle_score"] = int(40 + max(0, v_taper * 20))
                metrics["recommendation"] = "Increase cardio output and maintain caloric deficit to reveal underlying muscle definition."

        return metrics
