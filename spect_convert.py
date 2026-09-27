import torch.nn.functional as F
import torch
import torchaudio.functional as AF

BT_FPS = 50
FM_FPS = 100
DB_OFFSET = 34.49

def upsample_time(x):
    """(B, 128, T) at 50 fps -> (B, 128, 2T - 2) at 100 fps, linear interpolation.

    input frame i sits at i / 50 s, output frame j at j / 100 s.
    the last frame is dropped, as MusicFM does for its own mel.
    """
    T = x.shape[-1]
    x = F.interpolate(x, size=2 * T - 1, mode="linear", align_corners=True)
    return x[..., :-1]


def bt_to_mag(spect):
    """Beat This log1p(1000 * mel magnitude) -> mel magnitude (exact inverse)"""
    return torch.expm1(spect.float()) / 1000.0

def band_centers_hz(fb, sr, n_fft):
    w = fb/fb.sum(0, keepdim=True)
    bins = torch.arange(fb.shape[0], dtype=fb.dtype)
    return (bins[:, None] * w).sum(0) * sr / n_fft

def build_freq_map():
    """Beat This bands -> MusicFM bands.

    returns
      width_bt : (128,) filter sum of each Beat This band
      freq_map : (128, 128) per-bin power at Beat This centers -> MusicFM band power
    assumes power is flat inside each band (the only choice once bins are summed).
    """
    fb_bt = AF.melscale_fbanks(513, 30.0, 11000.0, 128, 22050, norm=None, mel_scale="slaney")
    fb_fm = AF.melscale_fbanks(1025, 0.0, 12000.0, 128, 24000, norm=None, mel_scale="htk")
    hz_bt = band_centers_hz(fb_bt, 22050, 1024)
    hz_fm = band_centers_hz(fb_fm, 24000, 2048)

    f = hz_fm.clamp(hz_bt[0], hz_bt[-1])          # outside 30-11000 Hz: repeat the edge band
    k = torch.searchsorted(hz_bt, f).clamp(1, 127)
    a = (f - hz_bt[k - 1]) / (hz_bt[k] - hz_bt[k - 1])
    interp = torch.zeros(128, 128)                # linear interpolation over frequency
    j = torch.arange(128)
    interp[k - 1, j] = 1 - a
    interp[k, j] += a
    return fb_bt.sum(0), interp * fb_fm.sum(0)[None, :]

def mel_bt_to_fm(mag, width_bt, freq_map):
    """(B, 128, T) Beat This mel magnitude -> (B, 128, T) MusicFM mel power"""
    per_bin_power = (mag / width_bt[:, None]) ** 2          # divide by band width BEFORE squaring
    return torch.einsum("bkt,kj->bjt", per_bin_power, freq_map)

def power_to_musicfm(power, stats, db_offset=DB_OFFSET):
    """power -> MusicFM input: AmplitudeToDB (10 * log10, amin 1e-10), then msd_stats normalization"""
    db = 10.0 * torch.log10(power.clamp(min=1e-10)) + db_offset 
    return (db-stats["melspec_2048_mean"]) / stats["melspec_2048_std"]

def bt_to_musicfm(spect, stats, width_bt, freq_map, db_offset=DB_OFFSET, output_fps=100):
    mag = bt_to_mag(spect.transpose(1,2))
    power = mel_bt_to_fm(mag, width_bt, freq_map)
    mel = power_to_musicfm(power, stats, db_offset)
    if output_fps == 50:
        return mel
    if output_fps == 100:
        return upsample_time(mel)
    raise ValueError(f"unsupported MusicFM mel frame rate: {output_fps}")
