"""Port of ``sub_functions/Generate_obsv.m`` and ``sub_functions/normalize01.m``.

Note: the random number generators of MATLAB and NumPy differ, so the generated noise
is statistically equivalent but not bit-identical to the MATLAB version.
"""
import numpy as np


def _imnoise_salt_pepper(x, d, rng):
    """Same as MATLAB imnoise(x, 'salt & pepper', d)."""
    b = x.copy()
    r = rng.random(x.shape)
    b[r < d / 2] = 0
    b[(r >= d / 2) & (r < d)] = 1
    return b


def Generate_obsv(HSI_clean, deg, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    deg = dict(deg)
    n1, n2, n3 = HSI_clean.shape

    # Generating stripe noise
    stripe_rate = deg["stripe_rate"]
    stripe_intensity = deg["stripe_intensity"]

    sparse_stripe = (2 * (_imnoise_salt_pepper(0.5 * np.ones((1, n2, n3)), stripe_rate, rng) - 0.5)
                     * rng.random((1, n2, n3)) * np.ones((n1, n2, n3)))
    stripe_noise = stripe_intensity * sparse_stripe / np.max(np.abs(sparse_stripe))

    HSI_noisy = HSI_clean + stripe_noise
    deg["stripe_noise"] = stripe_noise

    # Generating Gaussian noise
    Gaussian_sigma = deg["Gaussian_sigma"]
    Gaussian_noise = Gaussian_sigma * rng.standard_normal((n1, n2, n3))

    HSI_noisy = HSI_noisy + Gaussian_noise
    deg["Gaussian_noise"] = Gaussian_noise

    # Generating sparse noise
    sparse_rate = deg["sparse_rate"]
    HSI_tmp = HSI_noisy.copy()

    Sp = _imnoise_salt_pepper(0.5 * np.ones((n1, n2, n3)), sparse_rate, rng)
    HSI_noisy[Sp == 0] = 0
    HSI_noisy[Sp == 1] = 1

    deg["sparse_noise"] = HSI_noisy - HSI_tmp
    return HSI_noisy, deg


def normalize01(x):
    x_max = np.max(x)
    x_min = np.min(x)
    return (x - x_min) / (x_max - x_min)
