import logging
import threading
import time
from pathlib import Path

import av
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import VideoProcessorBase

from detectors.squat import SquatDetector
from detectors.pushup import PushUpDetector
from detectors.biecep_curls import BicepsCurlDetector
from detectors.shoulder_press import ShoulderPressDetector
from detectors.lunges import LungesDetector
from services.config.workout_config import POSE_CONNECTIONS

# TEMPORARY: makes our INFO logs show in the terminal. Remove when finished.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("exercise_processor")

# Set to False to test the webcam through your real app WITHOUT MediaPipe.
POSE_ENABLED = True

DRAW_MIN_VISIBILITY = 0.5
MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# services/vision/this_file.py -> parents[2] is the project root,
# so the path works no matter where you launch the app from.
MODEL_PATH = Path(__file__).resolve().parents[2] / "ml_models" / "pose_landmarker_full.task"


def _vis(landmark):
    """Visibility as a number; if the model gives None, treat it as visible."""
    v = getattr(landmark, "visibility", None)
    return 1.0 if v is None else v


class VideoProcessorClass(VideoProcessorBase):
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_metrics = None
        self._exercise_type = "Squats"

        self._landmarker = None
        self._detectors = {}
        self._init_error = None

        self._t0 = time.monotonic()
        self._last_ts_ms = -1
        self._frame_count = 0
        self._error_count = 0
        self._last_pose_found = False
        self._last_detect_ms = 0.0

        if POSE_ENABLED:
            self._init_pose()
        else:
            logger.info("POSE_ENABLED is False: camera-only mode.")

    def _init_pose(self):
        try:
            logger.info("Loading pose model from: %s", MODEL_PATH)
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(f"Pose model not found at {MODEL_PATH}")

            base_option = python.BaseOptions(model_asset_path=str(MODEL_PATH))
            options = vision.PoseLandmarkerOptions(
                base_options=base_option,
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_pose_presence_confidence=MIN_PRESENCE_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
                output_segmentation_masks=False,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)

            self._detectors = {
                "Squats": SquatDetector(),
                "Push-ups": PushUpDetector(),
                "Biceps Curls (Dumbbell)": BicepsCurlDetector(),
                "Shoulder Press": ShoulderPressDetector(),
                "Lunges": LungesDetector(),
            }
            logger.info("Pose landmarker and detectors loaded OK.")
        except Exception as e:
            self._landmarker = None
            self._init_error = str(e)
            logger.exception("Pose initialization FAILED. Streaming camera only.")

    def on_ended(self):
        try:
            if self._landmarker is not None:
                self._landmarker.close()
        except Exception:
            logger.exception("Error closing landmarker")

    def _next_timestamp_ms(self):
        """Real elapsed milliseconds, guaranteed to strictly increase."""
        now_ms = int((time.monotonic() - self._t0) * 1000)
        if now_ms <= self._last_ts_ms:
            now_ms = self._last_ts_ms + 1
        self._last_ts_ms = now_ms
        return now_ms

    def _log_error(self, where):
        """Log the first few errors, then every 100th, so the log is not flooded."""
        self._error_count += 1
        if self._error_count <= 3 or self._error_count % 100 == 0:
            logger.exception("Error in %s (error #%d)", where, self._error_count)

    def set_latest_metrics(self, metrics):
        with self._lock:
            self._latest_metrics = metrics.copy()

    def get_latest_metrics(self):
        with self._lock:
            return None if self._latest_metrics is None else self._latest_metrics.copy()

    def set_exercise(self, exercise_type):
        with self._lock:
            self._exercise_type = exercise_type

    def get_exercise(self):
        with self._lock:
            return self._exercise_type

    def _draw_skeleton(self, img, landmarks):
        h, w = img.shape[:2]

        for start_idx, end_idx in POSE_CONNECTIONS:
            p1 = landmarks[start_idx]
            p2 = landmarks[end_idx]

            if _vis(p1) > DRAW_MIN_VISIBILITY and _vis(p2) > DRAW_MIN_VISIBILITY:
                cv2.line(
                    img,
                    (int(p1.x * w), int(p1.y * h)),
                    (int(p2.x * w), int(p2.y * h)),
                    (0, 255, 0),
                    8,
                )

        for lm in landmarks:
            if _vis(lm) > DRAW_MIN_VISIBILITY:
                cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 8, (255, 0, 0), -1)

    def _draw_no_pose_warnings(self, img):
        cv2.putText(img, "NO POSE DETECTED", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, "PLEASE FACE THE CAMERA", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    def _draw_overlays(self, img, metrics, ex_type):
        if ex_type == "Squats":
            self._draw_squats_overlays(img, metrics)
        elif ex_type == "Push-ups":
            self._draw_pushup_overlays(img, metrics)
        elif ex_type == "Biceps Curls (Dumbbell)":
            self._draw_curl_overlays(img, metrics)
        elif ex_type == "Shoulder Press":
            self._draw_press_overlays(img, metrics)
        elif ex_type == "Lunges":
            self._draw_lunge_overlays(img, metrics)

    def _draw_squats_overlays(self, img, metrics):
        h, _ = img.shape[:2]
        cv2.putText(img, f"DEPTH: {metrics['depth_status']}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def _draw_pushup_overlays(self, img, metrics):
        h, _ = img.shape[:2]
        cv2.putText(img, f"BODY: {metrics['body_alignment']} | HIP: {metrics['hip_status']}",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def _draw_curl_overlays(self, img, metrics):
        h, _ = img.shape[:2]
        cv2.putText(img, f"SWING: {metrics['swing_status']}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def _draw_press_overlays(self, img, metrics):
        h, _ = img.shape[:2]
        cv2.putText(img, f"EXT: {metrics['extension_status']} | BACK: {metrics['back_arch_status']}",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def _draw_lunge_overlays(self, img, metrics):
        h, _ = img.shape[:2]
        cv2.putText(img, f"BALANCE: {metrics['balance_status']}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def _process_frame(self, image):
        """Draws on `image` in place. Any error here is caught by recv()."""
        if not POSE_ENABLED:
            cv2.putText(image, "CAMERA ONLY (pose off)", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return

        if self._landmarker is None:
            cv2.putText(image, "POSE MODEL NOT LOADED - see terminal", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return

        # OpenCV frames are BGR; MediaPipe SRGB images need RGB.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        t_start = time.perf_counter()
        result = self._landmarker.detect_for_video(mp_image, self._next_timestamp_ms())
        self._last_detect_ms = (time.perf_counter() - t_start) * 1000

        if result.pose_landmarks:
            self._last_pose_found = True
            landmarks = result.pose_landmarks[0]

            # Skeleton first, so it still shows even if a detector fails.
            self._draw_skeleton(image, landmarks)

            try:
                ex_type = self.get_exercise()
                detector = self._detectors.get(ex_type)
                if detector:
                    metrics = detector.process(landmarks)
                    metrics["pose_detected"] = True
                    self._draw_overlays(image, metrics, ex_type)
                    self.set_latest_metrics(metrics)
            except Exception:
                self._log_error("exercise detector / overlay")
        else:
            self._last_pose_found = False
            self._draw_no_pose_warnings(image)
            with self._lock:
                if self._latest_metrics is not None:
                    self._latest_metrics["pose_detected"] = False
                else:
                    self._latest_metrics = {"pose_detected": False}

    def recv(self, frame):
        self._frame_count += 1

        try:
            image = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
        except Exception:
            self._log_error("reading the camera frame")
            return frame  # never leave the stream without a frame

        try:
            self._process_frame(image)
        except Exception:
            self._log_error("processing the frame")
            cv2.putText(image, "PROCESSING ERROR - see terminal", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if self._frame_count % 60 == 0:
            logger.info("frame %d | pose=%s | detect=%.1f ms",
                        self._frame_count, self._last_pose_found, self._last_detect_ms)

        return av.VideoFrame.from_ndarray(image, format="bgr24")