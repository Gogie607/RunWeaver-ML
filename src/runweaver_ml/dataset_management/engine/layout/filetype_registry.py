
import io
import math
import warnings
import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
from scipy.signal import resample_poly

from.npz_codex import decode_nested_npz

# -------------------------------------------------
# txt
# -------------------------------------------------


def load_txt(fileobj):

    if isinstance(fileobj, bytes):
        return fileobj.decode("utf-8")

    return fileobj.read().decode(
        "utf-8"
    )


# -------------------------------------------------
# npy
# -------------------------------------------------


def load_npy(fileobj):

    if isinstance(fileobj, bytes):
        fileobj = io.BytesIO(fileobj)
    else:
        fileobj = io.BytesIO(fileobj.read())

    return np.load(
        fileobj
    )

def load_npz(fileobj):
    return decode_nested_npz(load_npy(fileobj))

# -------------------------------------------------
# wav
# -------------------------------------------------


def load_wav(fileobj, target_sr=16000):

    data = _read_bytes(fileobj)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", WavFileWarning)
        sr, wav = wavfile.read(io.BytesIO(data))

    wav = _wav_to_float32(wav)

    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    elif wav.ndim > 2:
        raise RuntimeError(
            f"Unsupported wav shape: {wav.shape}"
        )

    if sr != target_sr:
        divisor = math.gcd(sr, target_sr)
        wav = resample_poly(
            wav,
            target_sr // divisor,
            sr // divisor,
        )

    return np.ascontiguousarray(
        wav,
        dtype=np.float32,
    )





def _read_bytes(fileobj):

    if isinstance(fileobj, bytes):
        return fileobj

    return fileobj.read()


def _wav_to_float32(wav):

    if np.issubdtype(wav.dtype, np.floating):
        return wav.astype(np.float32, copy=False)

    if wav.dtype == np.uint8:
        return (
            wav.astype(np.float32) - 128.0
        ) / 128.0

    if np.issubdtype(wav.dtype, np.integer):
        info = np.iinfo(wav.dtype)
        scale = max(abs(info.min), info.max)
        return wav.astype(np.float32) / float(scale)

    raise RuntimeError(
        f"Unsupported wav dtype: {wav.dtype}"
    )


# -------------------------------------------------
# registry
# -------------------------------------------------

FILETYPE_LOADERS = {

    "txt": load_txt,

    "npy": load_npy,

    "wav": load_wav,

    "npz": load_npz
}
