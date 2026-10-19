"""Spatio-Spectral Structure Tensor Total Variation for Hyperspectral Image Denoising and Destriping
Python / PyTorch port of S3TTV_CPU.m and S3TTV_GPU.m (standard P-PDS algorithm).

Article: S. Takemoto, K. Naganuma, S. Ono,
  ``Spatio-Spectral Structure Tensor Total Variation for Hyperspectral Image Denoising and Destriping''

INPUT:
  HSI_noisy: noisy hyperspectral image of size n1*n2*n3 normalized to [0,1] (numpy array or torch tensor)
  params: a dict whose fields are as follows:
      alpha: radius of l_1 ball for sparse noise
      beta: radius of l_1 ball for stripe noise
      epsilon: radius of l_2 ball serving data-fidelity
      blocksize: parameter of block size for spatio-spectral structure tensor
      maxiter: maximum number of iterations
      stopcri: stopping criterion of P-PDS
      disprate: (optional) period to display intermediate results (default: 100)
      svt_method: (optional) "eigh" (default, fast) or "svd" (same as MATLAB)
  device: "cuda" (GPU) or "cpu"
OUTPUT:
  HSI_restored: denoised hyperspectral image (numpy array)
  removed_noise: dict of removed noise (sparse_noise, stripe_noise, gaussian_noise)
  iteration: number of P-PDS iterations
  converge_rate_U: history of ||U_next - U|| / ||U||
"""
import time

import numpy as np
import torch

from sub_functions_python.operators_python import (
    D, Dt, Dv, Dvt, Ds, Dst, func_PeriodicExpansion, func_PeriodicExpansionTrans)
from sub_functions_python.prox_python import (
    ProjBox, ProjFastL1Ball, ProjL2ball, Prox_S3TTV)


@torch.no_grad()
def S3TTV(HSI_noisy, params, device="cuda"):
    print(f"** Running S3TTV ({device}) **")
    device = torch.device(device)
    dtype = torch.float32

    HSI_noisy = torch.as_tensor(np.asarray(HSI_noisy), dtype=dtype, device=device)
    n1, n2, n3 = HSI_noisy.shape

    alpha = float(params["alpha"])
    beta = float(params["beta"])
    epsilon = float(params["epsilon"])
    blocksize = tuple(int(b) for b in params["blocksize"])
    maxiter = int(params["maxiter"])
    stopcri = float(params["stopcri"])
    disprate = int(params.get("disprate", 100))
    svt_method = params.get("svt_method", "eigh")

    # Initializing primal and dual variables
    # primal variables
    # U: clean HSI
    # S: sparse noise (salt-and-pepper noise)
    # T: stripe noise
    U = torch.zeros((n1, n2, n3), dtype=dtype, device=device)
    S = torch.zeros_like(U)
    T = torch.zeros_like(U)

    # dual variables
    # Y1: term of S3TTV
    # Y2: term of l2ball
    # Y3: term of stripe noise
    Y1 = torch.zeros((n1, n2, n3, 2, blocksize[0], blocksize[1]), dtype=dtype, device=device)
    Y2 = torch.zeros_like(U)
    Y3 = torch.zeros_like(U)

    # Expansion operators
    P = lambda z: func_PeriodicExpansion(z, blocksize)
    Pt = func_PeriodicExpansionTrans

    # Setting stepsize parameters for P-PDS
    gamma1_U = 1 / (np.prod(blocksize) * 2 * 2 * 2 + 1)
    gamma1_S = 1.0
    gamma1_T = 1 / (2 + 1)
    gamma2_Y1 = 1 / (2 * 2)
    gamma2_Y2 = 1 / 3
    gamma2_Y3 = 1 / 2

    # main loop (P-PDS)
    converge_rate_U = np.zeros(maxiter, dtype=np.float32)
    print("~~~ P-PDS STARTS ~~~")
    start = time.perf_counter()

    for i in range(1, maxiter + 1):
        # Updating U
        U_tmp = U - gamma1_U * (Dst(Dt(Pt(Y1))) + Y2)
        U_next = ProjBox(U_tmp, 0, 1)

        # Updating S
        S_tmp = S - gamma1_S * Y2
        S_next = ProjFastL1Ball(S_tmp, alpha)

        # Updating T
        T_tmp = T - gamma1_T * (Y2 + Dvt(Y3))
        T_next = ProjFastL1Ball(T_tmp, beta)

        # Updating Y1
        Y1_tmp = Y1 + gamma2_Y1 * P(D(Ds(2 * U_next - U)))
        Y1_next = Y1_tmp - gamma2_Y1 * Prox_S3TTV(Y1_tmp / gamma2_Y1, 1 / gamma2_Y1, blocksize, svt_method)
        del Y1_tmp

        # Updating Y2
        Y2_tmp = Y2 + gamma2_Y2 * (2 * (U_next + S_next + T_next) - (U + S + T))
        Y2_next = Y2_tmp - gamma2_Y2 * ProjL2ball(Y2_tmp / gamma2_Y2, HSI_noisy, epsilon)

        # Updating Y3
        Y3_next = Y3 + gamma2_Y3 * Dv(2 * T_next - T)

        # Calculating error
        converge_rate_U[i - 1] = (torch.linalg.vector_norm(U_next - U) / torch.linalg.vector_norm(U)).item()

        # Updating all variables
        U, S, T = U_next, S_next, T_next
        Y1, Y2, Y3 = Y1_next, Y2_next, Y3_next

        # Convergence checking
        if i >= 2 and converge_rate_U[i - 1] < stopcri:
            print(f"Iter: {i}, Error: {converge_rate_U[i - 1]:0.6f}.")
            break
        if i % disprate == 0:  # Displaying intermediate results
            print(f"Iter: {i}, Error: {converge_rate_U[i - 1]:0.6f}, Time: {time.perf_counter() - start:.1f} s.")

    print("~~~ P-PDS ENDS ~~~")

    # Organizing results for output
    HSI_noisy = HSI_noisy.cpu().numpy()
    HSI_restored = U.cpu().numpy()
    iteration = i
    removed_noise = {
        "sparse_noise": S.cpu().numpy(),
        "stripe_noise": T.cpu().numpy(),
    }
    removed_noise["gaussian_noise"] = (HSI_noisy - HSI_restored
                                       - removed_noise["sparse_noise"] - removed_noise["stripe_noise"])
    converge_rate_U = converge_rate_U[:iteration]
    return HSI_restored, removed_noise, iteration, converge_rate_U
