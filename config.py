"""
BiomechAI - Configuration File
"""

# User Anthropometric Parameters (Optional: for enhanced metric precision)
USER_CONFIG = {
    "gender": "",          # 'male' or 'female'
    "height_cm": None,     # Height in cm
    "weight_kg": None,     # Weight in kg
    "age": None,           # Age
}

# Camera Settings
CAMERA_INDEX = 1          # 1 for MacBook built-in FaceTime HD camera, 0 for iPhone/External
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# UI & Visualization Configuration
UI_CONFIG = {
    "show_landmarks": True,         # Show skeleton and face mesh points
    "show_hud": True,               # Show overlay telemetry panels
    "hud_opacity": 0.86,            # Frosted glass panel opacity (0.0 - 1.0)
    "theme_color": (0, 255, 180),   # Neon Mint / Cyan (BGR)
    "accent_color": (255, 130, 0),  # Neon Blue / Orange (BGR)
    "warning_color": (0, 180, 255), # Orange / Amber
    "danger_color": (0, 50, 255),   # Red / Coral
    "success_color": (0, 255, 120), # Emerald Green
}

# Google Gemini API Key (Optional: for deep vision AI fitness & posture coaching)
# Can also be provided as environment variable: GEMINI_API_KEY
GEMINI_API_KEY = ""
