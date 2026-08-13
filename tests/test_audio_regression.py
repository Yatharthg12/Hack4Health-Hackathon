from __future__ import annotations

import io
import math
import struct
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import mindfuse.data.speech as speech_module
from mindfuse.data.speech import AudioProcessingError, extract_audio_features, load_audio


def _pcm16_wav(sample_rate: int = 16_000, channels: int = 1, seconds: float = 0.32) -> bytes:
    time = np.arange(int(sample_rate * seconds), dtype=np.float64) / sample_rate
    left = 0.42 * np.sin(2 * math.pi * 220 * time)
    if channels == 1:
        samples = left[:, None]
    else:
        samples = np.column_stack([
            (0.42 - index * 0.04) * np.sin(
                2 * math.pi * (220 + 55 * index) * time + index * 0.2
            )
            for index in range(channels)
        ])
    pcm = np.int16(np.clip(samples, -1, 1) * 32767)
    output = io.BytesIO()
    with wave.open(output, "wb") as destination:
        destination.setnchannels(channels)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate)
        destination.writeframes(pcm.tobytes())
    return output.getvalue()


def _soundfile_wav(subtype: str) -> bytes:
    rate = 11_025
    time = np.arange(int(rate * 0.35), dtype=np.float32) / rate
    samples = 0.35 * np.sin(2 * np.pi * 260 * time)
    output = io.BytesIO()
    sf.write(output, samples, rate, format="WAV", subtype=subtype)
    return output.getvalue()


def _with_metadata_chunk(payload: bytes, chunk_id: bytes = b"JUNK") -> bytes:
    chunk_payload = b"MindFuse metadata"
    if len(chunk_payload) % 2:
        chunk_payload += b"\0"
    chunk = chunk_id + struct.pack("<I", len(chunk_payload)) + chunk_payload
    updated = payload[:12] + chunk + payload[12:]
    return updated[:4] + struct.pack("<I", len(updated) - 8) + updated[8:]


def _wave_format_extensible() -> bytes:
    sample_rate, channels, bits = 16_000, 2, 16
    frames = np.arange(int(sample_rate * 0.3), dtype=np.float64) / sample_rate
    samples = np.column_stack((
        0.4 * np.sin(2 * np.pi * 190 * frames),
        0.3 * np.sin(2 * np.pi * 310 * frames + 0.2),
    ))
    data = np.int16(samples * 32767).tobytes()
    block_align = channels * bits // 8
    pcm_guid = bytes.fromhex("0100000000001000800000aa00389b71")
    fmt = struct.pack(
        "<HHIIHHH", 0xFFFE, channels, sample_rate,
        sample_rate * block_align, block_align, bits, 22,
    ) + struct.pack("<HI", bits, 0x3) + pcm_guid
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


@pytest.mark.parametrize(
    ("name", "payload", "expected_channels"),
    [
        ("mono_pcm16.wav", _pcm16_wav(8_000, 1), 1),
        ("stereo_pcm16.wav", _pcm16_wav(22_050, 2), 2),
        ("four_channel_pcm16.wav", _pcm16_wav(16_000, 4), 4),
        ("pcm_u8.wav", _soundfile_wav("PCM_U8"), 1),
        ("pcm_24.wav", _soundfile_wav("PCM_24"), 1),
        ("float32.wav", _soundfile_wav("FLOAT"), 1),
        ("extensible.wav", _wave_format_extensible(), 2),
        ("metadata.wav", _with_metadata_chunk(_pcm16_wav(), b"LIST"), 1),
    ],
    ids=["mono-pcm16", "stereo-pcm16", "four-channel-pcm16", "pcm-u8", "pcm-24", "float32", "extensible", "metadata-list"],
)
def test_valid_wav_variants_reach_fixed_model_shape(
    tmp_path: Path, name: str, payload: bytes, expected_channels: int
) -> None:
    path = tmp_path / name
    path.write_bytes(payload)
    spectrogram, waveform, sample_rate, metadata = extract_audio_features(path, include_metadata=True)
    assert spectrogram.shape == (64, 249)
    assert waveform.ndim == 1 and waveform.size > 0
    assert sample_rate == 16_000
    assert metadata["channels"] == expected_channels
    assert metadata["decoder"] in {"soundfile/libsndfile", "scipy.io.wavfile"}
    assert np.isfinite(spectrogram).all()


@pytest.mark.parametrize(
    ("filename", "mime_type", "payload"),
    [
        ("VOICE.WAV", "application/octet-stream", _pcm16_wav(16_000, 1)),
        ("stereo.wav", "audio/x-wav", _pcm16_wav(22_050, 2)),
        ("wave-mime.wav", "audio/wave", _pcm16_wav()),
        ("vendor-mime.wav", "audio/vnd.wave", _pcm16_wav()),
        ("metadata.wav", "audio/wav", _with_metadata_chunk(_pcm16_wav(), b"JUNK")),
    ],
    ids=["uppercase-octet-stream", "stereo-browser-mime", "audio-wave-mime", "vendor-wave-mime", "metadata-junk"],
)
def test_real_audio_endpoint_accepts_browser_and_valid_wav_variants(
    real_client, filename: str, mime_type: str, payload: bytes
) -> None:
    response = real_client.post(
        "/api/predict/audio",
        data={"audio": (io.BytesIO(payload), filename, mime_type)},
    )
    assert response.status_code == 200
    result = response.get_json()
    assert result["ok"] is True
    assert result["emotion"] in real_client.application.extensions["model_registry"].audio_classes
    assert result["metadata"]["model_input_shape"] == [64, 249]
    assert result["explanation"]["available"] is True
    assert result["audio_explanation"].startswith("data:image/png;base64,")
    probabilities = np.asarray(list(result["emotion_probabilities"].values()), dtype=float)
    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("organizer_index", range(3))
def test_real_organizer_wav_returns_http_200(real_client, organizer_index: int) -> None:
    candidates = sorted(Path("data/raw/Audios").rglob("*.wav"))
    assert len(candidates) >= 3, "At least three organizer WAV files are required for this regression test"
    candidate = candidates[organizer_index]
    with candidate.open("rb") as source:
        response = real_client.post(
            "/api/predict/audio",
            data={"audio": (source, candidate.name, "audio/x-wav")},
        )
    assert response.status_code == 200
    result = response.get_json()
    assert result["ok"] is True
    assert result["metadata"]["original_sample_rate"] > 0
    assert result["metadata"]["original_duration_seconds"] > 0


def test_explanation_failure_does_not_destroy_prediction(real_client, monkeypatch) -> None:
    def fail_visualization(*_args, **_kwargs):
        raise ValueError("simulated optional visualization failure")

    monkeypatch.setattr("mindfuse.inference.service.audio_explanation_image", fail_visualization)
    response = real_client.post(
        "/api/predict/audio",
        data={"audio": (io.BytesIO(_pcm16_wav()), "valid.wav", "audio/wav")},
    )
    assert response.status_code == 200
    result = response.get_json()
    assert result["ok"] is True
    assert result["audio_explanation"] is None
    assert result["explanation"]["available"] is False
    assert result["emotion"]
    assert result["stress_probabilities"]


@pytest.mark.parametrize(
    ("data", "expected_code"),
    [
        ({}, "MISSING_UPLOAD"),
        ({"audio": (io.BytesIO(b""), "empty.wav")}, "EMPTY_UPLOAD"),
        ({"audio": (io.BytesIO(b"not a wave"), "random.wav")}, "INVALID_AUDIO"),
        ({"audio": (io.BytesIO(_pcm16_wav()), "voice.mp3")}, "UNSUPPORTED_FILE_TYPE"),
    ],
)
def test_audio_endpoint_returns_structured_safe_errors(real_client, data, expected_code: str) -> None:
    response = real_client.post("/api/predict/audio", data=data)
    assert response.status_code == 400
    result = response.get_json()
    assert result["ok"] is False
    assert result["error"]["code"] == expected_code
    assert result["error"]["message"]
    serialized = response.get_data(as_text=True).lower()
    assert "traceback" not in serialized
    assert "instance\\uploads" not in serialized


def test_silent_valid_wav_has_specific_error(tmp_path: Path) -> None:
    path = tmp_path / "silent.wav"
    path.write_bytes(_pcm16_wav())
    payload = bytearray(path.read_bytes())
    data_offset = payload.index(b"data") + 8
    payload[data_offset:] = b"\0" * (len(payload) - data_offset)
    path.write_bytes(payload)
    with pytest.raises(AudioProcessingError) as captured:
        load_audio(path)
    assert captured.value.code == "SILENT_AUDIO"


def test_scipy_decoder_is_a_functional_fallback(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "fallback.wav"
    path.write_bytes(_with_metadata_chunk(_pcm16_wav(), b"JUNK"))
    monkeypatch.setattr(speech_module, "sf", None)
    loaded = load_audio(path)
    assert loaded.metadata.decoder == "scipy.io.wavfile"
    assert loaded.sample_rate == 16_000
    assert loaded.waveform.ndim == 1
    assert np.isfinite(loaded.waveform).all()
