"""Validated label maps, numerical schema, and user-facing feature metadata."""

from __future__ import annotations

from typing import Final

STRESS_CLASSES: Final[list[str]] = [
    "Healthy",
    "Mild_Stress",
    "Moderate_Stress",
    "Severe_Stress",
]

FACE_EMOTIONS: Final[list[str]] = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

SPEECH_EMOTIONS: Final[list[str]] = [
    "Neutral",
    "Calm",
    "Happy",
    "Sad",
    "Angry",
    "Fearful",
    "Disgust",
    "Surprised",
]

# These mappings intentionally differ for Angry and Disgust and must not be unified.
FACE_EMOTION_TO_STRESS: Final[dict[str, str]] = {
    "Happy": "Healthy",
    "Neutral": "Healthy",
    "Sad": "Mild_Stress",
    "Surprise": "Mild_Stress",
    "Fear": "Moderate_Stress",
    "Disgust": "Moderate_Stress",
    "Angry": "Severe_Stress",
}

SPEECH_EMOTION_TO_STRESS: Final[dict[str, str]] = {
    "Neutral": "Healthy",
    "Calm": "Healthy",
    "Happy": "Healthy",
    "Sad": "Mild_Stress",
    "Surprised": "Mild_Stress",
    "Fearful": "Moderate_Stress",
    "Angry": "Moderate_Stress",
    "Disgust": "Severe_Stress",
}

SPEECH_EMOTION_IDS: Final[dict[str, str]] = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised",
}

NUMERICAL_FEATURES: Final[list[str]] = [
    "Sleep_Quality",
    "Social_Engagement",
    "Daily_App_Usage_Min",
    "Typing_Speed_WPM",
    "Session_Frequency",
    "Idle_Time_Min",
    "Facial_Emotion_Variance",
    "Eye_Blink_Rate",
    "Smile_Intensity",
    "Head_Motion_Index",
    "MFCC_Mean",
    "MFCC_Variance",
    "Pitch_Mean",
    "Speech_Rate",
    "Heart_Rate_BPM",
    "HRV_Index",
    "Skin_Temperature",
    "GSR_Level",
]

CLASSIFICATION_TARGET: Final[str] = "Mental_Health_Status"
REGRESSION_TARGETS: Final[list[str]] = [
    "Depression_Score",
    "Anxiety_Score",
    "Stress_Score",
]
SCORE_RANGES: Final[dict[str, tuple[float, float]]] = {
    "Depression_Score": (0.0, 34.0),
    "Anxiety_Score": (0.0, 24.0),
    "Stress_Score": (0.0, 39.0),
}

# Limits are broad plausibility/UX limits, not clinical reference intervals.
FEATURE_METADATA: Final[dict[str, dict[str, object]]] = {
    "Sleep_Quality": {"min": 1, "max": 5, "step": 0.1, "unit": "/ 5", "demo": 3.2, "description": "Self-reported sleep quality"},
    "Social_Engagement": {"min": 1, "max": 5, "step": 0.1, "unit": "/ 5", "demo": 3.0, "description": "Self-reported social activity"},
    "Daily_App_Usage_Min": {"min": 0, "max": 1440, "step": 1, "unit": "min/day", "demo": 245, "description": "Daily app usage"},
    "Typing_Speed_WPM": {"min": 0, "max": 250, "step": 0.1, "unit": "WPM", "demo": 48, "description": "Typing speed"},
    "Session_Frequency": {"min": 0, "max": 300, "step": 1, "unit": "sessions/day", "demo": 28, "description": "Daily digital sessions"},
    "Idle_Time_Min": {"min": 0, "max": 1440, "step": 1, "unit": "min/day", "demo": 310, "description": "Daily idle time"},
    "Facial_Emotion_Variance": {"min": 0, "max": 100, "step": 0.01, "unit": "index", "demo": 0.42, "description": "Facial-expression variability"},
    "Eye_Blink_Rate": {"min": 0, "max": 100, "step": 0.1, "unit": "blinks/min", "demo": 18, "description": "Eye blink rate"},
    "Smile_Intensity": {"min": 0, "max": 100, "step": 0.01, "unit": "index", "demo": 0.55, "description": "Average smile intensity"},
    "Head_Motion_Index": {"min": 0, "max": 100, "step": 0.01, "unit": "index", "demo": 0.34, "description": "Head movement index"},
    "MFCC_Mean": {"min": -1000, "max": 1000, "step": 0.01, "unit": "coefficient", "demo": -18.5, "description": "Mean MFCC value"},
    "MFCC_Variance": {"min": 0, "max": 100000, "step": 0.01, "unit": "variance", "demo": 42, "description": "MFCC variance"},
    "Pitch_Mean": {"min": 20, "max": 1200, "step": 0.1, "unit": "Hz", "demo": 165, "description": "Average fundamental frequency"},
    "Speech_Rate": {"min": 0, "max": 15, "step": 0.01, "unit": "words/s", "demo": 2.4, "description": "Speech rate"},
    "Heart_Rate_BPM": {"min": 20, "max": 240, "step": 0.1, "unit": "BPM", "demo": 78, "description": "Average heart rate"},
    "HRV_Index": {"min": 0, "max": 500, "step": 0.1, "unit": "index", "demo": 54, "description": "Heart-rate variability index"},
    "Skin_Temperature": {"min": 20, "max": 45, "step": 0.1, "unit": "°C", "demo": 36.4, "description": "Skin temperature"},
    "GSR_Level": {"min": 0, "max": 1000, "step": 0.01, "unit": "level", "demo": 4.8, "description": "Galvanic skin response"},
}

