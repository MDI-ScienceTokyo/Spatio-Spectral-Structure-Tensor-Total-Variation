"""Linear operators used in S3TTV (ports of the anonymous functions in S3TTV_*.m
and of ``sub_functions/operator/func_PeriodicExpansion*.m``).

All tensors use the same axis order as the MATLAB code:
    HSI            : (n1, n2, n3)
    D(z)           : (n1, n2, n3, 2)
    P(D(z)) / Y1   : (n1, n2, n3, 2, blocksize[0], blocksize[1])
Every difference operator assumes periodic boundary conditions.
"""
import torch


# Spatial difference operator: vertical and horizontal differences, (n1,n2,n3) -> (n1,n2,n3,2)
def D(z):
    return torch.stack((torch.roll(z, -1, dims=0) - z, torch.roll(z, -1, dims=1) - z), dim=3)


# Adjoint of D: (n1,n2,n3,2) -> (n1,n2,n3)
def Dt(z):
    return (torch.roll(z[..., 0], 1, dims=0) - z[..., 0]
            + torch.roll(z[..., 1], 1, dims=1) - z[..., 1])


# Vertical difference operator and its adjoint
def Dv(z):
    return torch.roll(z, -1, dims=0) - z


def Dvt(z):
    return torch.roll(z, 1, dims=0) - z


# Spectral difference operator and its adjoint
def Ds(z):
    return torch.roll(z, -1, dims=2) - z


def Dst(z):
    return torch.roll(z, 1, dims=2) - z


def func_PeriodicExpansion(Du, blocksize):
    """Periodic expansion operator P: (n1,n2,n3,d) -> (n1,n2,n3,d,b1,b2).

    PDu[..., i, j] = circshift(Du, [-i, -j, 0])  (0-based i, j)
    """
    b1, b2 = int(blocksize[0]), int(blocksize[1])
    PDu = Du.new_empty(Du.shape + (b1, b2))
    for i in range(b1):
        for j in range(b2):
            PDu[..., i, j] = torch.roll(Du, shifts=(-i, -j), dims=(0, 1))
    return PDu


def func_PeriodicExpansionTrans(PDu):
    """Adjoint of the periodic expansion operator: (n1,n2,n3,d,b1,b2) -> (n1,n2,n3,d)."""
    b1, b2 = PDu.shape[-2], PDu.shape[-1]
    u = torch.zeros(PDu.shape[:-2], dtype=PDu.dtype, device=PDu.device)
    for i in range(b1):
        for j in range(b2):
            u += torch.roll(PDu[..., i, j], shifts=(i, j), dims=(0, 1))
    return u
