"""
Narrowband low-frequency analysis (LOFAR) used to identify vessel class from propeller tonals.

The 64-band log-mel spectrogram has only 2-3 bins below 100 Hz, so it cannot resolve the blade-rate harmonic comb
that distinguishes a 5-blade cargo ship (blade rate ~6-9.5 Hz) from a 4-blade tanker (~4.4-6.8 Hz). A 32768-point
Welch spectrum (0.67 Hz bins) resolves it; whitening by a running median turns tonal lines into positive peaks
independent of the broadband level, and the autocorrelation of the whitened spectrum gives the comb spacing
(= blade rate) without knowing the harmonic number.
"""

import numpy as np
from scipy import ndimage, signal

FS = 22050
NPERSEG = 32768
F_LO, F_HI = 8.0, 600.0
MEDIAN_BINS = 61                   # ~41 Hz running-median background
COMB_RANGE_HZ = (3.0, 200.0)       # candidate blade-rate / shaft spacing


def lofar_whitened(audio, fs=FS):
    """Whitened narrowband spectrum (dB above running-median background), shape (n_bins,), float32."""
    f, p = signal.welch(np.asarray(audio, dtype=np.float64), fs, nperseg=NPERSEG, noverlap=NPERSEG // 2, window="hann")
    m = (f >= F_LO) & (f < F_HI)
    db = 10.0 * np.log10(p[m] + 1e-20)
    bg = ndimage.median_filter(db, size=MEDIAN_BINS, mode="nearest")
    return (db - bg).astype(np.float32)


def bin_hz(fs=FS):
    return fs / NPERSEG


def comb_features(w, fs=FS):
    """Harmonic-comb descriptors of a whitened LOFAR vector: spacing f0 (Hz), strength 0..1, number of strong tones."""
    df = bin_hz(fs)
    x = np.clip(w.astype(np.float64), 0.0, None)
    n_tones = int((w > 6.0).sum())
    if x.sum() < 1e-6:
        return {"comb_f0_hz": 0.0, "comb_strength": 0.0, "lofar_tones": n_tones}
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-12)
    i0, i1 = int(COMB_RANGE_HZ[0] / df), int(COMB_RANGE_HZ[1] / df)
    seg = ac[i0:i1]
    pk, _ = signal.find_peaks(seg, height=0.05)
    if len(pk) == 0:
        return {"comb_f0_hz": 0.0, "comb_strength": 0.0, "lofar_tones": n_tones}
    h = seg[pk]
    first = pk[np.argmax(h >= 0.6 * h.max())]      # lowest-lag strong peak = fundamental spacing, not a multiple
    # parabolic refinement of the peak position
    k = i0 + first
    if 0 < k < len(ac) - 1:
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
    else:
        d = 0.0
    return {"comb_f0_hz": float((k + d) * df), "comb_strength": float(ac[k]), "lofar_tones": n_tones}
