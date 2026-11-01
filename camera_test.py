import logging
import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("camera_test")
frame_count = 0


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    global frame_count
    frame_count += 1
    try:
        image = frame.to_ndarray(format="bgr24")
        cv2.putText(
            image, f"CAMERA IS LIVE - frame {frame_count}", (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
        )
        if frame_count % 30 == 0:
            logger.info("processed frame %d", frame_count)
        return av.VideoFrame.from_ndarray(image, format="bgr24")
    except Exception:
        logger.exception("callback failed on frame %d", frame_count)
        return frame


st.title("Camera test")
webrtc_streamer(
    key="camera-test",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)