"""RAVDESS-style filename parsing and dependency-light audio features."""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - exercised only in reduced deployments
    sf = None

from mindfuse.constants import SPEECH_EMOTION_IDS

LOGGER = logging.getLogger(__name__)


class AudioProcessingError(ValueError):
    """A browser-safe audio error with diagnostic context retained for logs."""

    def __init__(
        self,
        code: str,
        public_message: str,
        *,
        stage: str,
        decoder: str | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message
        self.stage = stage
        self.decoder = decoder
        self.detail = detail


@dataclass(frozen=True)
class AudioMetadata:
    decoder: str
    original_sample_rate: int
    channels: int
    original_duration_seconds: float
    source_format: str
    source_subtype: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoadedAudio:
    waveform: np.ndarray
    sample_rate: int
    metadata: AudioMetadata


def parse_speech_filename(path: str | Path) -> dict[str, object]:
    """Parse and validate the seven RAVDESS filename fields."""

    name = Path(path).stem
    fields = name.split("-")
    if len(fields) != 7 or any(len(field) != 2 or not field.isdigit() for field in fields):
        raise ValueError(f"Invalid speech filename: {Path(path).name}")
    modality, channel, emotion_id, intensity, statement, repetition, actor_text = fields
    actor = int(actor_text)
    if modality not in {"01", "02", "03"}:
        raise ValueError("Unknown modality identifier")
    if channel not in {"01", "02"}:
        raise ValueError("Unknown vocal-channel identifier")
    if emotion_id not in SPEECH_EMOTION_IDS:
        raise ValueError("Unknown emotion identifier")
    if intensity not in {"01", "02"} or statement not in {"01", "02"} or repetition not in {"01", "02"}:
        raise ValueError("Invalid stimulus identifier")
    if not 1 <= actor <= 24:
        raise ValueError("Actor identifier must be in [01, 24]")
    return {
        "modality": modality,
        "channel": channel,
        "emotion_id": emotion_id,
        "emotion": SPEECH_EMOTION_IDS[emotion_id],
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
        "actor": actor,
        "sex": "female" if actor % 2 == 0 else "male",
    }


def _validate_wave_signature(path: Path) -> None:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            header = handle.read(12)
    except OSError as exc:
        raise AudioProcessingError(
            "AUDIO_READ_FAILED",
            "The selected audio file could not be read. Please choose a valid WAV recording.",
            stage="container_validation",
            detail=type(exc).__name__,
        ) from exc
    if size == 0:
        raise AudioProcessingError(
            "EMPTY_AUDIO", "The selected audio file is empty.", stage="container_validation"
        )
    if len(header) < 12 or header[:4] not in {b"RIFF", b"RF64", b"RIFX"} or header[8:12] != b"WAVE":
        raise AudioProcessingError(
            "INVALID_AUDIO",
            "The selected file is not a valid WAV recording or is corrupt.",
            stage="container_validation",
        )


def _read_with_soundfile(path: Path) -> tuple[int, np.ndarray, AudioMetadata]:
    if sf is None:
        raise RuntimeError("soundfile is not installed")
    info = sf.info(str(path))
    waveform, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    metadata = AudioMetadata(
        decoder="soundfile/libsndfile",
        original_sample_rate=int(sample_rate),
        channels=int(info.channels),
        original_duration_seconds=float(info.frames / info.samplerate) if info.samplerate else 0.0,
        source_format=str(info.format or "WAV"),
        source_subtype=str(info.subtype or "unknown"),
    )
    return int(sample_rate), waveform, metadata


def _read_with_scipy(path: Path) -> tuple[int, np.ndarray, AudioMetadata]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", WavFileWarning)
        sample_rate, raw = wavfile.read(path)
    for warning in caught:
        LOGGER.warning("SciPy WAV decoder warning (%s): %s", type(warning.message).__name__, warning.message)
    source_dtype = str(raw.dtype)
    if raw.ndim == 1:
        channels = 1
        raw = raw[:, None]
    elif raw.ndim == 2:
        channels = int(raw.shape[1])
    else:
        raise ValueError("unsupported channel tensor")
    if np.issubdtype(raw.dtype, np.unsignedinteger):
        info = np.iinfo(raw.dtype)
        midpoint = float(info.max + 1) / 2.0
        waveform = (raw.astype(np.float32) - midpoint) / midpoint
    elif np.issubdtype(raw.dtype, np.signedinteger):
        info = np.iinfo(raw.dtype)
        scale = float(max(abs(info.min), info.max))
        waveform = raw.astype(np.float32) / scale
    else:
        waveform = raw.astype(np.float32)
    metadata = AudioMetadata(
        decoder="scipy.io.wavfile",
        original_sample_rate=int(sample_rate),
        channels=channels,
        original_duration_seconds=float(waveform.shape[0] / sample_rate) if sample_rate else 0.0,
        source_format="WAV",
        source_subtype=source_dtype,
    )
    return int(sample_rate), waveform, metadata


def load_audio(path: str | Path, target_sample_rate: int = 16_000) -> LoadedAudio:
    """Decode a WAV robustly, downmix deterministically, resample, and normalize."""

    audio_path = Path(path)
    _validate_wave_signature(audio_path)
    decoder_failures: list[str] = []
    try:
        sample_rate, channel_waveform, metadata = _read_with_soundfile(audio_path)
    except Exception as exc:
        decoder_failures.append(f"soundfile:{type(exc).__name__}:{exc}")
        LOGGER.warning("Primary WAV decoder failed; attempting SciPy fallback: %s", exc)
        try:
            sample_rate, channel_waveform, metadata = _read_with_scipy(audio_path)
        except Exception as fallback_exc:
            decoder_failures.append(f"scipy:{type(fallback_exc).__name__}:{fallback_exc}")
            raise AudioProcessingError(
                "AUDIO_DECODE_FAILED",
                "We could not decode this WAV recording. Try exporting it as PCM WAV and upload it again.",
                stage="decode",
                decoder="soundfile,scipy",
                detail=" | ".join(decoder_failures),
            ) from fallback_exc

    if sample_rate <= 0 or channel_waveform.size == 0 or channel_waveform.shape[0] == 0:
        raise AudioProcessingError(
            "EMPTY_AUDIO", "The WAV recording contains no audio samples.", stage="waveform_validation",
            decoder=metadata.decoder,
        )
    if channel_waveform.ndim != 2 or not 1 <= channel_waveform.shape[1] <= 32:
        raise AudioProcessingError(
            "UNSUPPORTED_CHANNEL_LAYOUT",
            "This WAV channel layout is not supported. Please use a mono or stereo recording.",
            stage="waveform_validation", decoder=metadata.decoder,
        )
    if not np.all(np.isfinite(channel_waveform)):
        raise AudioProcessingError(
            "INVALID_AUDIO_SAMPLES",
            "The WAV recording contains invalid samples and cannot be analyzed.",
            stage="waveform_validation", decoder=metadata.decoder,
        )

    waveform = channel_waveform.astype(np.float32, copy=False).mean(axis=1, dtype=np.float32)
    waveform = waveform - np.float32(waveform.mean(dtype=np.float64))
    peak = float(np.max(np.abs(waveform)))
    if peak <= 1e-8:
        raise AudioProcessingError(
            "SILENT_AUDIO",
            "The WAV recording is silent or too quiet to analyze.",
            stage="waveform_validation", decoder=metadata.decoder,
        )
    waveform = (waveform / peak).astype(np.float32, copy=False)
    if sample_rate != target_sample_rate:
        divisor = math.gcd(int(sample_rate), target_sample_rate)
        try:
            waveform = signal.resample_poly(
                waveform,
                target_sample_rate // divisor,
                int(sample_rate) // divisor,
            ).astype(np.float32)
        except Exception as exc:
            raise AudioProcessingError(
                "AUDIO_PREPROCESSING_FAILED",
                "The WAV recording could not be prepared for analysis.",
                stage="resample", decoder=metadata.decoder, detail=f"{type(exc).__name__}:{exc}",
            ) from exc
        sample_rate = target_sample_rate
    if waveform.size == 0 or not np.all(np.isfinite(waveform)):
        raise AudioProcessingError(
            "AUDIO_PREPROCESSING_FAILED",
            "The WAV recording could not be prepared for analysis.",
            stage="resample", decoder=metadata.decoder,
        )
    return LoadedAudio(waveform=waveform, sample_rate=sample_rate, metadata=metadata)


def load_waveform(path: str | Path, target_sample_rate: int = 16_000) -> tuple[np.ndarray, int]:
    """Backward-compatible waveform loader used by training and tests."""

    loaded = load_audio(path, target_sample_rate)
    return loaded.waveform, loaded.sample_rate


def _mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 20.0,
    f_max: float | None = None,
) -> np.ndarray:
    f_max = min(float(f_max or sample_rate / 2), sample_rate / 2)
    mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    hz = lambda value: 700.0 * (10.0 ** (value / 2595.0) - 1.0)
    mel_points = np.linspace(mel(f_min), mel(f_max), n_mels + 2)
    bins = np.floor((n_fft + 1) * hz(mel_points) / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = bins[index : index + 3]
        if center == left:
            center += 1
        if right == center:
            right += 1
        right = min(right, n_fft // 2 + 1)
        for frequency_bin in range(left, min(center, filters.shape[1])):
            filters[index, frequency_bin] = (frequency_bin - left) / max(center - left, 1)
        for frequency_bin in range(center, right):
            filters[index, frequency_bin] = (right - frequency_bin) / max(right - center, 1)
    return filters


def log_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = 16_000,
    duration_seconds: float = 4.0,
    n_fft: int = 512,
    hop_length: int = 256,
    n_mels: int = 64,
) -> np.ndarray:
    """Return a standardized fixed-size log-Mel spectrogram."""

    target_samples = int(sample_rate * duration_seconds)
    if waveform.size < target_samples:
        waveform = np.pad(waveform, (0, target_samples - waveform.size))
    else:
        waveform = waveform[:target_samples]
    _, _, stft = signal.stft(
        waveform,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    power = np.abs(stft).astype(np.float32) ** 2
    mel_power = _mel_filterbank(sample_rate, n_fft, n_mels) @ power
    log_mel = np.log10(np.maximum(mel_power, 1e-10))
    mean, std = float(log_mel.mean()), float(log_mel.std())
    return ((log_mel - mean) / max(std, 1e-6)).astype(np.float32)


def extract_audio_features(
    path: str | Path, *, include_metadata: bool = False
) -> tuple[np.ndarray, np.ndarray, int] | tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
    """Load a WAV and return model features plus waveform for visual explanation."""

    loaded = load_audio(path)
    try:
        spectrogram = log_mel_spectrogram(loaded.waveform, loaded.sample_rate)
    except Exception as exc:
        raise AudioProcessingError(
            "AUDIO_PREPROCESSING_FAILED",
            "The WAV recording could not be converted into model features.",
            stage="feature_extraction", decoder=loaded.metadata.decoder,
            detail=f"{type(exc).__name__}:{exc}",
        ) from exc
    if spectrogram.shape != (64, 249) or not np.all(np.isfinite(spectrogram)):
        raise AudioProcessingError(
            "AUDIO_PREPROCESSING_FAILED",
            "The WAV recording produced invalid model features.",
            stage="feature_validation", decoder=loaded.metadata.decoder,
            detail=f"shape={spectrogram.shape},finite={bool(np.all(np.isfinite(spectrogram)))}",
        )
    result = (spectrogram, loaded.waveform, loaded.sample_rate)
    if include_metadata:
        return (*result, loaded.metadata.as_dict())
    return result
