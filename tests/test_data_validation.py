from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mindfuse.constants import FEATURE_METADATA, NUMERICAL_FEATURES
from mindfuse.data.numerical import validate_batch_frame, validate_numerical_payload
from mindfuse.data.speech import parse_speech_filename


def valid_profile() -> dict[str, float]:
    return {feature: float(FEATURE_METADATA[feature]["demo"]) for feature in NUMERICAL_FEATURES}


def test_speech_filename_parser_extracts_actor_emotion_and_sex() -> None:
    result = parse_speech_filename("03-01-06-01-02-01-12.wav")
    assert result["emotion"] == "Fearful"
    assert result["actor"] == 12
    assert result["sex"] == "female"


@pytest.mark.parametrize("filename", ["bad.wav", "03-01-99-01-01-01-01.wav", "03-01-06-01-01-01-25.wav"])
def test_speech_filename_parser_rejects_malformed_names(filename: str) -> None:
    with pytest.raises(ValueError):
        parse_speech_filename(filename)


def test_numerical_schema_returns_ordered_finite_floats() -> None:
    result = validate_numerical_payload(valid_profile())
    assert list(result) == NUMERICAL_FEATURES
    assert all(np.isfinite(list(result.values())))


def test_numerical_schema_rejects_missing_invalid_and_out_of_range() -> None:
    payload = valid_profile(); payload.pop("Sleep_Quality")
    with pytest.raises(ValueError, match="Missing"):
        validate_numerical_payload(payload)
    payload = valid_profile(); payload["Heart_Rate_BPM"] = "fast"
    with pytest.raises(ValueError, match="numeric"):
        validate_numerical_payload(payload)
    payload = valid_profile(); payload["Sleep_Quality"] = 99
    with pytest.raises(ValueError, match="between"):
        validate_numerical_payload(payload)


def test_batch_schema_rejects_nan_and_missing_columns() -> None:
    profile = valid_profile()
    frame = pd.DataFrame([profile])
    frame.loc[0, "MFCC_Mean"] = np.nan
    with pytest.raises(ValueError, match="Non-numeric"):
        validate_batch_frame(frame)
    with pytest.raises(ValueError, match="Missing"):
        validate_batch_frame(pd.DataFrame([{"Sleep_Quality": 3}]))

