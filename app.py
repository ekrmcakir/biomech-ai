#!/usr/bin/env python3
"""
BiomechAI // v5.0 (English Edition)
===================================
1. Voice-Guided 360° Body & Posture Scanner:
   - Front Profile : V-Taper Ratio, Shoulder Width, Body Fat Estimation
   - Back Profile  : Left/Right Shoulder Drop (cm/deg), Spinal Curvature
   - Side Profile  : Thoracic Kyphosis, Forward Head Angle (Text Neck)
2. Facial Adiposity & Jawline Taper Scanner ([F] Key).
3. Holographic Silhouette Segmentation & Cyber Laser ([B] Key).

Key Shortcuts:
    [T] / [SPACE] : START VOICE-GUIDED 360° SEQUENTIAL SCAN
    [V]           : Toggle Voice Coach (ON / OFF)
    [F]           : Switch Mode (360° BODY <-> FACIAL FAT)
    [B]           : Toggle Silhouette / Cyber Laser Mode
    [S]           : Save Screenshot & Telemetry Report to Disk
    [H]           : Toggle HUD / Skeleton Lines
    [Q/ESC]       : Quit
"""

import os
import sys
import time
import datetime
import warnings

# Suppress upstream TensorFlow and Google Protobuf deprecation warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2

from config import USER_CONFIG, UI_CONFIG, CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, GEMINI_API_KEY
from scanner import BodyScanner
from face_scanner import FaceScanner
from ui_overlay import UIOverlay
from ai_analyzer import AIAnalyzer
from voice_coach import VoiceCoach


def main():
    print("=" * 76)
    print("    BIOMECHAI 5.0 // 360° BIOMECHANICAL POSTURE & FACIAL ADIPOSITY ENGINE    ")
    print("=" * 76)
    print("Initializing camera feed...")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        for alt_idx in [1, 2]:
            cap = cv2.VideoCapture(alt_idx)
            if cap.isOpened():
                break

    if not cap.isOpened():
        print("[Error] No camera device found. Please verify camera permissions.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    user_height = USER_CONFIG.get("height_cm", 170)
    body_scanner = BodyScanner()
    face_scanner = FaceScanner()
    overlay = UIOverlay(UI_CONFIG)
    analyzer = AIAnalyzer(api_key=GEMINI_API_KEY)
    voice = VoiceCoach(enabled=True)

    app_mode = "body"      # 'body' or 'face'
    show_landmarks = UI_CONFIG.get("show_landmarks", True)
    bg_mode = 0
    scan_flash_until = 0
    last_report_text = ""

    # Multi-Angle Biometric Cache
    cached_front_metrics = None
    cached_side_metrics = None
    cached_back_metrics = None

    # Guided Scan State Machine: 0: Idle, 1: Front, 2: Back, 3: Side
    guided_stage = 0
    stage_target_detected_time = 0
    countdown_duration = 3.0

    window_name = "BiomechAI // 360 Biomechanics HUD"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, FRAME_WIDTH, FRAME_HEIGHT)

    print("\n✓ System Ready!")
    print("  * Press [T] or [SPACE] to initiate the VOICE-GUIDED 360° SCAN.")
    print("    - Sequence: 1. FRONT -> 2. BACK -> 3. SIDE profile scans.")
    print("  * Press [V] to toggle Voice Coach (ON / OFF).")
    print("  * Press [F] to switch between BODY SCAN and FACIAL FAT & JAWLINE mode.")
    print("  * Press [B] to toggle Cyber Laser / Dark Studio silhouette background.")
    print("  * Press [Q] or [ESC] to quit.\n")

    retry_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            retry_count += 1
            time.sleep(0.1)
            if retry_count == 5:
                print("[Info] Awaiting camera feed stream...")
            if retry_count > 35:
                print("[Error] Camera feed disconnected. Reinitializing...")
                cap.release()
                time.sleep(0.5)
                cap = cv2.VideoCapture(CAMERA_INDEX)
                retry_count = 0
                if not cap.isOpened():
                    break
            continue
        retry_count = 0

        frame = cv2.flip(frame, 1)
        now = time.time()
        is_flashing = now < scan_flash_until

        guided_info = {"active": False, "message": "", "countdown": 0}

        if app_mode == "body":
            metrics = body_scanner.process_frame(frame, user_height_cm=user_height)

            if metrics["detected"]:
                mode = metrics["view_mode"]
                if mode == "front":
                    cached_front_metrics = metrics.copy()
                elif mode == "side":
                    cached_side_metrics = metrics.copy()
                elif mode == "back":
                    cached_back_metrics = metrics.copy()

            # --- VOICE-GUIDED SEQUENTIAL SCANNER ENGINE ---
            if guided_stage == 1:
                guided_info["active"] = True
                if metrics["detected"] and metrics["view_mode"] == "front" and "Optimal" in metrics["distance_status"]:
                    if stage_target_detected_time == 0:
                        stage_target_detected_time = now

                    elapsed = now - stage_target_detected_time
                    remaining = max(0, int(countdown_duration - elapsed) + 1)
                    guided_info["countdown"] = remaining
                    guided_info["message"] = f"STAGE 1/3: FRONT SCAN - Stand Still [{remaining}]"

                    if elapsed >= countdown_duration:
                        cached_front_metrics = metrics.copy()
                        scan_flash_until = now + 0.5
                        guided_stage = 2
                        stage_target_detected_time = 0
                        guided_info["countdown"] = 0
                        print("\n✓ 1/3 Front Profile Captured -> Please turn around to show your back!")
                        voice.announce_front_captured()
                else:
                    stage_target_detected_time = 0
                    guided_info["message"] = "STAGE 1/3: Please stand facing camera directly"

            elif guided_stage == 2:
                guided_info["active"] = True
                if metrics["detected"] and metrics["view_mode"] == "back":
                    if stage_target_detected_time == 0:
                        stage_target_detected_time = now

                    elapsed = now - stage_target_detected_time
                    remaining = max(0, int(countdown_duration - elapsed) + 1)
                    guided_info["countdown"] = remaining
                    guided_info["message"] = f"STAGE 2/3: BACK SCAN - Hold Still [{remaining}]"

                    if elapsed >= countdown_duration:
                        cached_back_metrics = metrics.copy()
                        scan_flash_until = now + 0.5
                        guided_stage = 3
                        stage_target_detected_time = 0
                        guided_info["countdown"] = 0
                        print("✓ 2/3 Back Profile Captured -> Please turn 90° to the side!")
                        voice.announce_back_captured()
                else:
                    stage_target_detected_time = 0
                    guided_info["message"] = "STAGE 2/3: Please turn your BACK to the camera"

            elif guided_stage == 3:
                guided_info["active"] = True
                if metrics["detected"] and metrics["view_mode"] == "side":
                    if stage_target_detected_time == 0:
                        stage_target_detected_time = now

                    elapsed = now - stage_target_detected_time
                    remaining = max(0, int(countdown_duration - elapsed) + 1)
                    guided_info["countdown"] = remaining
                    guided_info["message"] = f"STAGE 3/3: SIDE SCAN - Hold Still [{remaining}]"

                    if elapsed >= countdown_duration:
                        cached_side_metrics = metrics.copy()
                        scan_flash_until = now + 0.8
                        guided_stage = 0
                        stage_target_detected_time = 0
                        guided_info["active"] = False

                        print("\n" + "=" * 68)
                        print(">> VOICE-GUIDED 360° SCAN COMPLETED SUCCESSFULLY! <<")
                        print("=" * 68)
                        local_report = analyzer.generate_local_report(
                            metrics, USER_CONFIG,
                            front_metrics=cached_front_metrics,
                            side_metrics=cached_side_metrics,
                            back_metrics=cached_back_metrics
                        )
                        last_report_text = local_report
                        print(local_report)

                        v_tap = cached_front_metrics.get("v_taper_ratio", 1.4) if cached_front_metrics else 1.4
                        drop_sd = cached_back_metrics.get("shoulder_drop_side", "none") if cached_back_metrics else "none"
                        drop_cm = cached_back_metrics.get("shoulder_drop_cm", 0.0) if cached_back_metrics else 0.0
                        voice.announce_all_completed(v_taper=v_tap, drop_side=drop_sd, drop_cm=drop_cm)

                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(f"guided_scan_{ts}.png", frame)
                        with open(f"guided_report_{ts}.txt", "w", encoding="utf-8") as f:
                            f.write(local_report)
                        print(f"✓ Report and snapshot saved: guided_report_{ts}.txt")
                else:
                    stage_target_detected_time = 0
                    guided_info["message"] = "STAGE 3/3: Please turn 90 degrees to the SIDE"

            rendered_frame = overlay.apply_segmentation_effects(
                frame, metrics.get("segmentation_mask"), bg_mode=bg_mode
            )

            display_frame = overlay.draw_hud(
                rendered_frame, metrics,
                show_landmarks=show_landmarks,
                scan_active=is_flashing,
                bg_mode=bg_mode,
                guided_info=guided_info,
                voice_enabled=voice.enabled
            )

            cv2.putText(display_frame, "[F] SWITCH TO FACE MODE", (FRAME_WIDTH - 250, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)

        else:
            # --- FACIAL ADIPOSITY & JAWLINE SCANNER MODE ---
            face_metrics = face_scanner.process_frame(frame)
            display_frame = face_scanner.draw_face_hud(frame, face_metrics, show_mesh=show_landmarks)

            cv2.putText(display_frame, "[F] SWITCH TO BODY MODE", (FRAME_WIDTH - 250, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)

        if is_flashing:
            white_layer = display_frame.copy()
            white_layer[:] = (255, 255, 255)
            cv2.addWeighted(white_layer, 0.35, display_frame, 0.65, 0, display_frame)

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF

        # [Q] or [ESC] -> Quit
        if key == ord('q') or key == ord('Q') or key == 27:
            print("\nApplication terminated. Stay healthy and fit!")
            break

        # [V] -> Toggle Voice Coach
        elif key == ord('v') or key == ord('V'):
            voice.enabled = not voice.enabled
            status = "ON" if voice.enabled else "OFF"
            print(f"[Voice Coach] Status: {status}")
            if voice.enabled:
                voice.speak("Voice coach activated.")

        # [F] -> Switch Mode (Body <-> Face)
        elif key == ord('f') or key == ord('F'):
            app_mode = "face" if app_mode == "body" else "body"
            guided_stage = 0
            mode_name = "FACIAL ADIPOSITY & JAWLINE MODE" if app_mode == "face" else "360° BODY & POSTURE MODE"
            print(f"\n[Mode Switched] Active Mode: >> {mode_name} <<")
            if app_mode == "face":
                if voice.enabled:
                    voice.speak("Switched to facial scan mode. Please center your face.")
            else:
                if voice.enabled:
                    voice.speak("Switched to body scan mode.")

        # [B] -> Toggle Silhouette / Laser Background
        elif key == ord('b') or key == ord('B'):
            bg_mode = (bg_mode + 1) % 3
            mod_names = ["Normal Camera", "Dark Studio Mode (Neon Outline)", "Holographic Cyber Laser"]
            print(f"[Silhouette Mode] Active: {mod_names[bg_mode]}")

        # [H] -> Toggle Landmarks / Skeleton
        elif key == ord('h') or key == ord('H'):
            show_landmarks = not show_landmarks
            print(f"[UI] Overlay lines: {'ON' if show_landmarks else 'OFF'}")

        # [T] or [SPACE] -> Start Voice 360° Scan (or Face Snapshot)
        elif key == ord('t') or key == ord('T') or key == 32:
            if app_mode == "body":
                if guided_stage == 0:
                    guided_stage = 1
                    stage_target_detected_time = 0
                    print("\n>> VOICE-GUIDED 360° SCAN INITIATED <<")
                    voice.announce_scan_start()
                else:
                    guided_stage = 0
                    stage_target_detected_time = 0
                    print("[Info] Guided scan aborted.")
            else:
                scan_flash_until = now + 0.6
                face_report = analyzer.generate_face_report(face_metrics, USER_CONFIG)
                last_report_text = face_report
                print("\n" + "=" * 65)
                print(">> FACIAL ADIPOSITY & JAWLINE REPORT <<")
                print("=" * 65)
                print(face_report)
                if voice.enabled:
                    low_f, high_f = face_metrics.get("facial_fat_range", (12, 16))
                    voice.speak(f"Facial scan complete. Estimated body fat is between {low_f} and {high_f} percent.")

        # [S] -> Save Snapshot & Telemetry Report
        elif key == ord('s') or key == ord('S'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = "face_scan" if app_mode == "face" else "manual_body_scan"
            img_path = f"{prefix}_snapshot_{timestamp}.png"
            rep_path = f"{prefix}_report_{timestamp}.txt"

            cv2.imwrite(img_path, display_frame)
            report_to_save = last_report_text if last_report_text else analyzer.generate_local_report(
                metrics, USER_CONFIG,
                front_metrics=cached_front_metrics,
                side_metrics=cached_side_metrics,
                back_metrics=cached_back_metrics
            )
            with open(rep_path, "w", encoding="utf-8") as f:
                f.write(report_to_save)

            print(f"\n✓ Successfully Saved:")
            print(f"  * Snapshot: {os.path.abspath(img_path)}")
            print(f"  * Report  : {os.path.abspath(rep_path)}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
