"""Spatio-Spectral Structure Tensor Total Variation for Hyperspectral Image Denoising and Destriping
Python / PyTorch port of main_S3TTV.m

Article: S. Takemoto, K. Naganuma, S. Ono,
  ``Spatio-Spectral Structure Tensor Total Variation for Hyperspectral Image Denoising and Destriping''

Usage:
  uv run python main_S3TTV_Python.py                       # settings written below
  uv run python main_S3TTV_Python.py --image PaviaUniversity --cpu --standard
  uv run python main_S3TTV_Python.py --help
"""
import argparse
import os

import numpy as np
import scipy.io as sio
import torch

from solvers_python.S3TTV_python import S3TTV
from solvers_python.S3TTV_fast_python import S3TTV_fast
from sub_functions_python.evaluation_python import calc_MPSNR, calc_MSSIM
from sub_functions_python.generate_obsv_python import Generate_obsv

# %%%%%%%%%%%%%%%%%%%%% User settings of experiment %%%%%%%%%%%%%%%%%%%%%%%%%%%%
deg = {
    "Gaussian_sigma": 0.1,     # Standard derivation of Gaussian noise
    "sparse_rate": 0.05,       # Rate of sparse noise
    "stripe_rate": 0.05,       # Rate of stripe noise
    "stripe_intensity": 0.5,   # Range of intensity for stripe noise
}

image = "JasperRidge"
# image = "PaviaUniversity"
# image = "Beltsville"
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%% User Settings %%%%%%%%%%%%%%%%%%%%%%%%%%%%
params = {
    "rho": 0.95,             # Parameter for the radii of the noise terms
    "blocksize": [10, 10],   # Block size of S3TTV
    "stopcri": 1e-5,         # Stopping criterion
    "maxiter": 20000,        # Maximum number of iterations
    "disprate": 100,         # Period to display intermediate results
    "svt_method": "eigh",    # "eigh" (fast) or "svd" (same computation as MATLAB's svd)
}

use_GPU = 1   # 1, if you use GPU, 0, if you use CPU
use_fast = 1  # 1, if you use Fast algorithm, 0, if you use Standard algorithm
seed = 0      # Seed of the random number generator for noise generation
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def parse_args():
    parser = argparse.ArgumentParser(description="S3TTV for HSI denoising and destriping (Python/PyTorch)")
    parser.add_argument("--image", choices=["JasperRidge", "PaviaUniversity", "Beltsville"], default=image)
    parser.add_argument("--rho", type=float, default=params["rho"])
    parser.add_argument("--blocksize", type=int, nargs=2, default=params["blocksize"])
    parser.add_argument("--stopcri", type=float, default=params["stopcri"])
    parser.add_argument("--maxiter", type=int, default=params["maxiter"])
    parser.add_argument("--disprate", type=int, default=params["disprate"])
    parser.add_argument("--svt_method", choices=["eigh", "svd"], default=params["svt_method"])
    parser.add_argument("--cpu", action="store_true", default=not use_GPU, help="Run on CPU")
    parser.add_argument("--standard", action="store_true", default=not use_fast,
                        help="Use the standard algorithm instead of the fast one")
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--noisy_mat", type=str, default=None,
                        help="(optional) .mat file containing a variable 'HSI_noisy' to use instead of generating noise")
    parser.add_argument("--result_dir", type=str, default="result_python")
    parser.add_argument("--no_show", action="store_true", help="Do not open a figure window")
    return parser.parse_args()


def main():
    args = parse_args()
    print("******* initium *******")

    # Generating observation
    data = sio.loadmat(os.path.join("dataset", f"{args.image}.mat"), squeeze_me=True, struct_as_record=False)
    HSI_clean = data["HSI_clean"].astype(np.float64)
    n1, n2, n3 = HSI_clean.shape
    N = n1 * n2 * n3

    if args.noisy_mat is None:
        HSI_noisy, _ = Generate_obsv(HSI_clean, deg, np.random.default_rng(args.seed))
    else:
        HSI_noisy = sio.loadmat(args.noisy_mat)["HSI_noisy"].astype(np.float64)

    HSI_clean = HSI_clean.astype(np.float32)
    HSI_noisy = HSI_noisy.astype(np.float32)

    edge_width = 3

    # Setting parameters
    p = {
        "rho": args.rho,
        "blocksize": args.blocksize,
        "stopcri": args.stopcri,
        "maxiter": args.maxiter,
        "disprate": args.disprate,
        "svt_method": args.svt_method,
    }
    # Radius of v-centered l2-ball constraint serving data-fidelity
    p["epsilon"] = p["rho"] * deg["Gaussian_sigma"] * np.sqrt(N * (1 - deg["sparse_rate"]))
    # Radius of l1-ball constraint for sparse noise
    p["alpha"] = p["rho"] * (0.5 * N * deg["sparse_rate"])
    # Radius of l1-ball constraint for stripe noise
    p["beta"] = p["rho"] * N * (1 - deg["sparse_rate"]) * deg["stripe_rate"] * deg["stripe_intensity"] / 2

    if not args.cpu and not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        args.cpu = True
    device = "cpu" if args.cpu else "cuda"

    # Showing settings
    print("~~~ SETTINGS ~~~")
    print(f"Image: {args.image} Size: ({n1}, {n2}, {n3})")
    print(f"Gaussian sigma: {deg['Gaussian_sigma']:g}")
    print(f"Sparse rate: {deg['sparse_rate']:g}")
    print(f"Stripe rate: {deg['stripe_rate']:g}")
    print(f"Stripe intensity: {deg['stripe_intensity']:g}")
    print(f"Rho: {p['rho']:g}")
    print(f"Stopping criterion: {p['stopcri']:g}")
    print(f"Blocksize: ({p['blocksize'][0]}, {p['blocksize'][1]})")
    print(f"Device: {device}, Algorithm: {'Standard' if args.standard else 'Fast'}")

    # Denoising and destriping
    if args.standard:
        HSI_restored, removed_noise, iteration, converge_rate_U = S3TTV(HSI_noisy, p, device)
    else:
        HSI_restored, removed_noise, iteration, converge_rate_U = S3TTV_fast(HSI_noisy, p, device)

    # Plotting results
    sl = slice(edge_width, n3 - edge_width)
    mpsnr_S3TTV = calc_MPSNR(HSI_restored[:, :, sl], HSI_clean[:, :, sl])
    mssim_S3TTV = calc_MSSIM(HSI_restored[:, :, sl], HSI_clean[:, :, sl])

    print("~~~ RESULTS ~~~")
    print(f"MPSNR: {mpsnr_S3TTV:#.4g}")
    print(f"MSSIM: {mssim_S3TTV:#.4g}")

    # Saving results
    os.makedirs(args.result_dir, exist_ok=True)
    name = f"S3TTV_{args.image}"
    sio.savemat(os.path.join(args.result_dir, f"{name}.mat"), {
        "HSI_clean": HSI_clean, "HSI_noisy": HSI_noisy, "HSI_restored": HSI_restored,
        "sparse_noise": removed_noise["sparse_noise"], "stripe_noise": removed_noise["stripe_noise"],
        "gaussian_noise": removed_noise["gaussian_noise"], "iteration": iteration,
        "converge_rate_U": converge_rate_U, "mpsnr": mpsnr_S3TTV, "mssim": mssim_S3TTV,
        "params": p, "deg": deg,
    })

    # Showing result images
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    visband = min(59, n3) - 1  # band to visualize (59 in MATLAB 1-based indexing)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, img, title in zip(axes, [HSI_clean, HSI_noisy, HSI_restored],
                              ["Ground-truth", "Noisy HSI", "Restored HSI"]):
        ax.imshow(np.clip(img[:, :, visband], 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{args.image} (band {visband + 1}) MPSNR: {mpsnr_S3TTV:.2f}, MSSIM: {mssim_S3TTV:.4f}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.result_dir, f"{name}.png"), dpi=150)
    print(f"Results saved to {args.result_dir}/{name}.mat and .png")
    if not args.no_show:
        plt.show()

    print("******* finis *******")


if __name__ == "__main__":
    main()
