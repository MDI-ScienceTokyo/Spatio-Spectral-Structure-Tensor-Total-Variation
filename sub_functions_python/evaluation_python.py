"""Ports of ``sub_functions/evaluation/calc_*.m``."""
import numpy as np
from scipy.ndimage import correlate


def calc_MPSNR(restorated_HSI, org_HSI):
    """Mean PSNR over bands (peak value = 1)."""
    restorated_HSI = np.asarray(restorated_HSI, dtype=np.float64)
    org_HSI = np.asarray(org_HSI, dtype=np.float64)
    n1, n2, n3 = org_HSI.shape
    difference_HSI = org_HSI - restorated_HSI
    psnr_per_band = 20 * np.log10(np.sqrt(n1 * n2) / np.sqrt(np.sum(difference_HSI ** 2, axis=(0, 1))))
    return float(np.mean(psnr_per_band))


def _gaussian_kernel(sigma=1.5):
    radius = int(np.ceil(3 * sigma))
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return h / h.sum()


def ssim(A, ref, dynamic_range=1.0, sigma=1.5):
    """Same as MATLAB ssim(A, ref) with default options
    (11x11 Gaussian window with sigma = 1.5, replicate padding, K = [0.01, 0.03])."""
    A = np.asarray(A, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    h = _gaussian_kernel(sigma)
    filt = lambda x: correlate(x, h, mode="nearest")

    C1 = (0.01 * dynamic_range) ** 2
    C2 = (0.03 * dynamic_range) ** 2

    mux = filt(A)
    muy = filt(ref)
    mux2, muy2, muxy = mux * mux, muy * muy, mux * muy
    sigmax2 = filt(A * A) - mux2
    sigmay2 = filt(ref * ref) - muy2
    sigmaxy = filt(A * ref) - muxy

    num = (2 * muxy + C1) * (2 * sigmaxy + C2)
    den = (mux2 + muy2 + C1) * (sigmax2 + sigmay2 + C2)
    return float(np.mean(num / den))


def calc_MSSIM(restorated_HSI, org_HSI):
    """Mean SSIM over bands."""
    n3 = org_HSI.shape[2]
    return sum(ssim(org_HSI[:, :, k], restorated_HSI[:, :, k]) for k in range(n3)) / n3


def calc_SAM(tar, ref):
    """Spectral angle mapper (SAM) in degrees, and its 2-D angle map."""
    tar = np.asarray(tar, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    prod_scal = np.sum(ref * tar, axis=2)
    norm_orig = np.sum(ref * ref, axis=2)
    norm_fusa = np.sum(tar * tar, axis=2)
    prod_norm = np.sqrt(norm_orig * norm_fusa)
    prod_map = prod_norm.copy()
    prod_map[prod_map == 0] = np.finfo(np.float64).eps
    angle_map = np.arccos(np.clip(prod_scal / prod_map, -1, 1))
    mask = prod_norm.reshape(-1) != 0
    angolo = np.sum(np.arccos(np.clip(prod_scal.reshape(-1)[mask] / prod_norm.reshape(-1)[mask], -1, 1))) / mask.sum()
    angle_SAM = float(np.real(angolo) * 180 / np.pi)
    return angle_SAM, angle_map
