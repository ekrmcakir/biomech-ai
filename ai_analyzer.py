import os
import datetime
from typing import Dict, Any, Optional
import cv2
from PIL import Image

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


class AIAnalyzer:
    """
    Biometric body composition, fat percentage, muscle build, 360° posture analysis,
    and facial adiposity / jawline assessment engine.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.client = None
        if self.api_key and HAS_GENAI:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[Warning] Gemini Client could not be initialized: {e}")

    def generate_local_report(
        self,
        metrics: Dict[str, Any],
        user_config: Optional[Dict[str, Any]] = None,
        front_metrics: Optional[Dict[str, Any]] = None,
        side_metrics: Optional[Dict[str, Any]] = None,
        back_metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generates comprehensive 360° posture and body composition report in English."""
        user_cfg = user_config or {}
        height = user_cfg.get("height_cm")
        weight = user_cfg.get("weight_kg")
        age = user_cfg.get("age")
        gender = user_cfg.get("gender")

        height_str = f"{height} cm" if height else "Not specified"
        weight_str = f"{weight} kg" if weight else "Not specified"
        age_str = f"{age}" if age else "Not specified"
        gender_str = gender.capitalize() if gender else "Not specified"

        if height and weight:
            try:
                bmi_val = round(float(weight) / ((float(height) / 100) ** 2), 1)
                bmi_str = f"{bmi_val}"
            except Exception:
                bmi_str = "N/A"
        else:
            bmi_str = "N/A (Provide in config.py)"

        f_met = front_metrics or (metrics if metrics.get("view_mode") == "front" else {})
        s_met = side_metrics or (metrics if metrics.get("view_mode") == "side" else {})
        b_met = back_metrics or (metrics if metrics.get("view_mode") == "back" else {})

        v_taper = f_met.get("v_taper_ratio", metrics.get("v_taper_ratio", 1.25))
        low_fat, high_fat = f_met.get("estimated_body_fat_range", metrics.get("estimated_body_fat_range", (12, 16)))
        body_type = f_met.get("body_type", metrics.get("body_type", "Athletic & Fit Build"))
        muscle_score = f_met.get("muscle_score", metrics.get("muscle_score", 70))

        sh_drop_status = b_met.get("shoulder_drop_status", metrics.get("shoulder_drop_status", "Balanced & Symmetrical ✓"))
        sh_drop_angle = b_met.get("shoulder_drop_angle", metrics.get("shoulder_drop_angle", 0.0))
        sh_drop_cm = b_met.get("shoulder_drop_cm", metrics.get("shoulder_drop_cm", 0.0))

        neck_status = s_met.get("neck_status", metrics.get("neck_status", "Optimal Upright Neck ✓"))
        neck_angle = s_met.get("head_tilt_angle", metrics.get("head_tilt_angle", 10.0))
        kyphosis_status = s_met.get("kyphosis_status", metrics.get("kyphosis_status", "Optimal Upright Spine ✓"))
        kyphosis_angle = s_met.get("kyphosis_angle", metrics.get("kyphosis_angle", 168.0))

        spine_status = b_met.get("spine_status", metrics.get("spine_status", "Spine Line Straight & Optimal ✓"))
        spine_offset = b_met.get("spine_lateral_offset_cm", metrics.get("spine_lateral_offset_cm", 0.0))
        pelvis_status = b_met.get("pelvis_status", metrics.get("pelvis_status", "Balanced"))

        posture_score = metrics.get("posture_score", 80)
        symmetry_score = metrics.get("body_symmetry", 75)

        report = f"""
=============================================================
   BIOMECHAI // 360° BIOMETRICS & POSTURE REPORT
=============================================================
Timestamp    : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
User Profile : {gender_str} | Age: {age_str} | Height: {height_str} | Weight: {weight_str} | BMI: {bmi_str}

[ 1. BODY COMPOSITION & AESTHETICS (FRONT PROFILE) ]
-------------------------------------------------------------
- Body Classification   : {body_type}
- Estimated Body Fat    : {low_fat}% - {high_fat}%
- V-Taper Ratio         : {v_taper:.2f} (Shoulder / Waist Width)
  * Reference Standards:
    - > 1.45 : Aesthetic Inverted Triangle / High V-Taper
    - 1.30 - 1.45 : Balanced Athletic / Fit Frame
    - < 1.20 : Rectangular / Heavier Torso Frame
- Muscle & Form Score   : {muscle_score} / 100

[ 2. SHOULDER SYMMETRY & SPINE BIOMECHANICS (BACK PROFILE) ]
-------------------------------------------------------------
- Shoulder Balance      : {sh_drop_status}
  * Tilt Angle          : {sh_drop_angle:.1f}° (Delta: ~{sh_drop_cm:.1f} cm)
  * < 1.6°: Symmetrical & Balanced
  * 1.6° - 3.8°: Mild Asymmetry / Lateral Drop
  * > 3.8°: Noticeable Shoulder Drop (Corrective exercise advised)
- Spine Alignment       : {spine_status} (Lateral Offset: {spine_offset:.1f} cm)
- Pelvis Alignment      : {pelvis_status}
- Overall Symmetry Score: {symmetry_score} / 100

[ 3. SPINAL HEALTH & CERVICAL POSTURE (SIDE PROFILE) ]
-------------------------------------------------------------
- Thoracic Kyphosis     : {kyphosis_angle:.1f}° -> Status: {kyphosis_status}
  * Optimal: 165° - 180° upright spinal neutral axis.
- Forward Head Angle    : {neck_angle:.1f}° -> Status: {neck_status}
  * Optimal: < 14° (Ear canal aligned directly above the acromion process).
  * 15° - 25°: Screen / Smartphone induced forward head posture (Text Neck).
- Posture Health Score  : {posture_score} / 100

[ 4. PERSONALIZED CORRECTIVE EXERCISE & COACHING PLAN ]
-------------------------------------------------------------
"""
        if sh_drop_angle >= 1.6:
            report += f"""* Shoulder Symmetry Correction:
  - {sh_drop_status} detected. Avoid carrying backpacks or shoulder bags unilaterally on one side.
  - Stretch the Upper Trapezius and Levator Scapulae on the dropped side.
  - Perform Single-Arm Dumbbell Overhead Presses and Single-Arm Dumbbell Rows
    to equalize neuromuscular stability across both shoulders.\n"""
        else:
            report += """* Shoulder Balance:
  - Your shoulders exhibit excellent bilateral symmetry on the horizontal plane. Maintain this form.\n"""

        if neck_angle > 14.0 or "Kyphosis" in kyphosis_status or "Hunchback" in kyphosis_status:
            report += """* Cervical & Thoracic Posture (Spinal Hygiene):
  - Chin Tucks: 3 sets of 12 reps daily; retract the chin backward to strengthen deep cervical flexors.
  - Face Pulls & Band Pull-Aparts: Strengthen rhomboids and rear deltoids to prevent rounded shoulders.
  - Wall Angels: 2 sets of 10 reps against a wall to improve thoracic mobility.\n"""

        if v_taper >= 1.40:
            report += """* Hypertrophy & Fitness Architecture:
  - Your V-Taper ratio is exemplary (aesthetic inverted triangle form).
  - Maintain lateral deltoid and latissimus dorsi development while balancing chest and leg volume.
  - Nutrition: High-protein clean nutrition (Lean Bulk or Maintenance)."""
        else:
            report += """* Hypertrophy & Fitness Architecture:
  - To increase V-Taper ratio: Add Pull-ups, Overhead Barbell Presses, and Dumbbell Lateral Raises.
  - Tighten core circumference through caloric balance to enhance shoulder-to-waist contrast."""

        report += "\n=============================================================\n"
        return report

    def generate_face_report(
        self,
        face_metrics: Dict[str, Any],
        user_config: Optional[Dict[str, Any]] = None
    ) -> str:
        user_cfg = user_config or {}
        height = user_cfg.get("height_cm")
        weight = user_cfg.get("weight_kg")
        age = user_cfg.get("age")
        gender = user_cfg.get("gender")

        height_str = f"{height} cm" if height else "Not specified"
        weight_str = f"{weight} kg" if weight else "Not specified"
        age_str = f"{age}" if age else "Not specified"
        gender_str = gender.capitalize() if gender else "Not specified"

        low_fat, high_fat = face_metrics.get("facial_fat_range", (12, 16))
        status = face_metrics.get("facial_adiposity_status", "Athletic")
        jaw_angle = face_metrics.get("jaw_angle", 72.0)
        sharpness = face_metrics.get("jawline_sharpness", 80)
        symmetry = face_metrics.get("facial_symmetry", 85)
        cjr = face_metrics.get("cheek_to_jaw_ratio", 1.15)
        fwhr = face_metrics.get("fwhr", 1.85)

        report = f"""
=============================================================
   BIOMECHAI // FACIAL ADIPOSITY & JAWLINE REPORT
=============================================================
Timestamp    : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
User Profile : {gender_str} | Age: {age_str} | Height: {height_str} | Weight: {weight_str}

[ 1. ESTIMATED BODY FAT FROM FACIAL ANTHROPOMETRICS ]
-------------------------------------------------------------
- Estimated Body Fat    : {low_fat}% - {high_fat}%
- Facial Adiposity      : {status}
- Jawline Structure     : {face_metrics.get('face_shape', 'Chiseled V-Line')}

[ 2. JAWLINE & CRANIOFACIAL METRICS ]
-------------------------------------------------------------
- Jawline V-Angle       : {jaw_angle:.1f}°
  * < 76° : Chiseled & Sharp Mandibular Taper (Low Body Fat / Chiseled)
  * 76° - 86° : Balanced & Athletic Jaw Contour
  * > 86° : Rounded / Softer Lower Facial Angle
- Jawline Sharpness     : {sharpness}% / 100
- Facial Symmetry Score : {symmetry}% / 100
- Cheek / Jaw Ratio     : {cjr:.2f} (Cheekbone width to gonial jaw angle)
- Facial Ratio (fWHR)   : {fwhr:.2f} (Bizygomatic width to upper face height)

[ 3. AESTHETIC & POSTURAL COACHING ADVICE ]
-------------------------------------------------------------
"""
        if sharpness >= 75:
            report += """* Facial Definition & Posture:
  - Subcutaneous facial fat is minimal; your mandibular border and zygomatic bones are prominent.
  - Maintain proper tongue posture (Mewing) with the tongue pressed flush against the palate.
  - Keep hydration optimal (2.5 - 3.0 liters daily) to preserve skin elasticity."""
        elif sharpness >= 55:
            report += """* Facial Definition & Posture:
  - Balanced and athletic facial volume.
  - To further accentuate bone definition, balance dietary sodium intake and drink adequate water to minimize fluid retention.
  - Chewing resistant foods or mastication training helps define the masseter corners."""
        else:
            report += """* Facial Definition & Posture:
  - Buccal fat volume is moderately soft.
  - As overall systemic body fat is reduced via caloric deficit and cardio, facial fat pads shrink rapidly.
  - Limiting late-night sodium and refined carbohydrates will eliminate morning facial puffiness."""

        report += "\n=============================================================\n"
        return report

    def analyze_with_gemini(self, frame_bgr, metrics, mode="body"):
        """Sends frame to Gemini Multimodal Vision API for expert fitness assessment in English."""
        if not self.client:
            return "Gemini API key is not configured. Local biometric telemetry active."

        try:
            rgb_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_img)

            if mode == "face":
                prompt = f"""
You are a top facial plastic surgeon, aesthetician, and fitness biomechanics specialist.
Analyze the person's face in the attached photo for facial adiposity, mandibular definition (jawline), buccal fat fullness, and facial symmetry.

Computed Biometric Measurements:
- Jawline V-Angle: {metrics.get('jaw_angle', 75)}°
- Jawline Sharpness: {metrics.get('jawline_sharpness', 80)}%
- Estimated Facial Body Fat Range: {metrics.get('facial_fat_range', (12, 16))[0]}% - {metrics.get('facial_fat_range', (12, 16))[1]}%
- Facial Symmetry: {metrics.get('facial_symmetry', 85)}%

Please provide an objective, professional, and encouraging evaluation in English covering:
1. Facial Fat Distribution & Bone Prominence
2. Estimated Total Body Fat Assessment
3. Practical Tips (Proper tongue posture/mewing, sodium/water balance, and skin health)
"""
            else:
                prompt = f"""
You are an elite sports physical therapist, biomechanics researcher, and fitness coach.
Analyze the person's upper body posture, shoulder symmetry, spinal alignment, and muscular aesthetics from the photo.

Computed Biometric Measurements:
- View Mode: {metrics.get('view_mode', 'front')}
- Shoulder Drop: {metrics.get('shoulder_drop_status', '-')} (Angle: {metrics.get('shoulder_drop_angle', 0)}°, Delta: {metrics.get('shoulder_drop_cm', 0)} cm)
- V-Taper Ratio: {metrics.get('v_taper_ratio', 1.25):.2f}
- Forward Head Angle: {metrics.get('head_tilt_angle', 10.0):.1f}° ({metrics.get('neck_status', '-')})
- Thoracic Kyphosis: {metrics.get('kyphosis_angle', 170.0):.1f}° ({metrics.get('kyphosis_status', '-')})
- Symmetry Score: {metrics.get('body_symmetry', 80)}/100

Please provide a detailed, professional evaluation in English covering:
1. Shoulder Symmetry & Spinal Line Assessment
2. Kyphosis & Forward Head Posture Evaluation
3. Corrective Exercise Prescription (Strengthen vs. Stretch)
4. Hypertrophy & Training Advice
"""
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[pil_image, prompt],
            )
            return response.text
        except Exception as e:
            return f"Gemini API Analysis Error: {str(e)}"
