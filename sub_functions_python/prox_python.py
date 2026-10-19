"""Proximal operators and projections (ports of ``sub_functions/prox/*.m``)."""
import torch


def ProjBox(u, a, b):
    """Projection onto the box [a, b]."""
    return torch.clamp(u, min=a, max=b)


def ProjL2ball(u, f, epsilon):
    """Projection onto the f-centered l2 ball with radius epsilon."""
    temp = u - f
    radius = torch.linalg.vector_norm(temp)
    if radius > epsilon:
        return f + (epsilon / radius) * temp
    return u


def ProjFastL1Ball(X, alpha):
    """Projection onto the l1 ball with radius alpha (whole array treated as one vector).

    If the computed threshold is negative, X is inside the ball and nothing is done.
    """
    x = X.reshape(-1)
    abs_x = x.abs()
    sorted_abs, _ = torch.sort(abs_x, descending=True)
    k = torch.arange(1, x.numel() + 1, dtype=x.dtype, device=x.device)
    thre = torch.clamp(torch.max((torch.cumsum(sorted_abs, dim=0) - alpha) / k), min=0)
    x = torch.clamp(abs_x - thre, min=0) * torch.sign(x)
    return x.reshape(X.shape)


def set_linalg_backend(device):
    """On CUDA, MAGMA's batched eigh is much faster than cuSOLVER for the ~100x100 Gram matrices used here."""
    if torch.device(device).type == "cuda":
        torch.backends.cuda.preferred_linalg_library("magma")


def _svt_batched(M, gamma, method):
    """Singular value thresholding U * max(S - gamma, 0) * V^T for a batch of matrices M (B, m, n)."""
    if method == "svd":
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        S = torch.clamp(S - gamma, min=0)
        return (U * S.unsqueeze(-2)) @ Vh

    if method == "eigh":
        # Same result computed via the eigen-decomposition of the (smaller) Gram matrix:
        #   U max(S-gamma,0) V^T = Q diag(max(1 - gamma/S, 0)) Q^T M,  M M^T = Q S^2 Q^T
        transpose = M.shape[-2] > M.shape[-1]
        if transpose:
            M = M.transpose(-2, -1)
        w, Q = torch.linalg.eigh(M @ M.transpose(-2, -1))
        s = torch.sqrt(torch.clamp(w, min=0))
        scale = torch.where(s > gamma, 1 - gamma / s.clamp(min=gamma), torch.zeros_like(s))
        out = (Q * scale.unsqueeze(-2)) @ (Q.transpose(-2, -1) @ M)
        return out.transpose(-2, -1) if transpose else out

    raise ValueError(f"Unknown SVT method: {method}")


def Prox_S3TTV(z, gamma, blocksize, method="svd", batch_size=None):
    """Proximal operator of the S3TTV (block-wise nuclear norm).

    z has the shape (n1, n2, n3, 2, b1, b2). For every non-overlapping spatial block
    of size b1 x b2 and every shift index (k, l), the block (b1, b2, n3, 2) is reshaped
    into a (b1*b2) x (2*n3) matrix and its singular values are soft-thresholded by gamma.
    As in the MATLAB code, pixels outside the last complete block (when n1 or n2 is not
    divisible by the block size) are left unchanged.

    method     : "svd" (torch.linalg.svd, same as MATLAB) or "eigh" (Gram-matrix based, faster on GPU)
    batch_size : number of matrices processed at once (None = all at once)
    """
    n1, n2, n3, d, s1, s2 = z.shape
    b1, b2 = int(blocksize[0]), int(blocksize[1])
    I, J = n1 // b1, n2 // b2

    z_out = z.clone()
    zb = z[:I * b1, :J * b2]
    # (I, b1, J, b2, n3, d, s1, s2) -> (I, J, s1, s2, b1, b2, n3, d)
    zb = zb.reshape(I, b1, J, b2, n3, d, s1, s2).permute(0, 2, 6, 7, 1, 3, 4, 5)
    M = zb.reshape(I * J * s1 * s2, b1 * b2, n3 * d)

    if batch_size is None or batch_size >= M.shape[0]:
        M = _svt_batched(M, gamma, method)
    else:
        M = torch.cat([_svt_batched(M[k:k + batch_size], gamma, method)
                       for k in range(0, M.shape[0], batch_size)], dim=0)

    zb = M.reshape(I, J, s1, s2, b1, b2, n3, d).permute(0, 4, 1, 5, 6, 7, 2, 3)
    z_out[:I * b1, :J * b2] = zb.reshape(I * b1, J * b2, n3, d, s1, s2)
    return z_out
