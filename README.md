# Spatio-Spectral Structure Tensor Total Variation for Hyperspectral Image Denoising and Destriping

This is a demo code of the proposed method in the following reference:

S. Takemoto, K. Naganuma, and S. Ono,
``Spatio-Spectral-Structure-Tensor-Total-Variation-for-Hyperspectral-Image-Denoising-and-Destriping,''
_IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing_, 2025.

Update history:
10, Aug. 2025: v1.0 

For more information, see the following

- Project website: https://www.mdi.c.titech.ac.jp/publications/s3ttv
- Preprint paper: https://arxiv.org/abs/2404.03313

## How to use
1. **Setting parameters**
 - Choose the image (JasperRidge, PaviaUniversity, or Beltsville)
 - Adjust the parameters
   - `params.rho`: parameter for the radii of the noise terms
   - `params.blocksize`: Block size of spatio-spectral structure tensor
   - `params.stopcri`: Stopping criterion
   - `params.maxiter`: Maximum number of iterations
   - `params.disprate`: Period to display intermediate results
 - Set as `use_GPU` = 1 if you use GPU.
 - Set as `use_fast` = 1 if you use fast convergence version.

2. Run ```main_S3TTV.m```


## Python (PyTorch) version
A Python/PyTorch port of the MATLAB code is also provided. It runs on both GPU (CUDA) and CPU.

| MATLAB | Python |
| --- | --- |
| `main_S3TTV.m` | `main_S3TTV_Python.py` |
| `S3TTV_CPU.m`, `S3TTV_GPU.m` | `S3TTV_python.py` (`S3TTV(HSI_noisy, params, device)`) |
| `S3TTV_CPU_fast.m`, `S3TTV_GPU_fast.m` | `S3TTV_fast_python.py` (`S3TTV_fast(HSI_noisy, params, device)`) |
| `sub_functions/operator/*.m` | `sub_functions_python/operators_python.py` |
| `sub_functions/prox/*.m` | `sub_functions_python/prox_python.py` |
| `sub_functions/Generate_obsv.m`, `normalize01.m` | `sub_functions_python/generate_obsv_python.py` |
| `sub_functions/evaluation/*.m` | `sub_functions_python/evaluation_python.py` |

### Setting up the environment
The environment is managed with [uv](https://docs.astral.sh/uv/).

1. Install uv (if not installed)
   ```bash
   # Linux / macOS
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
2. Create the virtual environment (`.venv/`) and install the dependencies (Python 3.12, PyTorch, NumPy, SciPy, Matplotlib)
   ```bash
   uv sync
   ```
3. Check that PyTorch can use the GPU
   ```bash
   uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

By default, PyTorch is installed from the CUDA 12.4 wheel index (`cu124`, see `[tool.uv.sources]` / `[[tool.uv.index]]` in `pyproject.toml`), which works with NVIDIA drivers supporting CUDA 12.4 or later.
If your driver supports a different CUDA version (check with `nvidia-smi`), replace `cu124` in `pyproject.toml` with the matching one (e.g., `cu118`, `cu126`, `cu128`) and run `uv sync` again.
On macOS, the default (CPU) PyTorch is installed automatically.

### Running
1. **Setting parameters**: edit the "User settings" block at the top of `main_S3TTV_Python.py` (same parameters as `main_S3TTV.m`)
   - `image`: `JasperRidge`, `PaviaUniversity`, or `Beltsville`
   - `deg`: noise settings (Gaussian sigma, sparse rate, stripe rate, stripe intensity)
   - `params["rho"]`, `params["blocksize"]`, `params["stopcri"]`, `params["maxiter"]`, `params["disprate"]`
   - `params["svt_method"]`: `"eigh"` (default) computes the singular value thresholding in the prox of S3TTV through the eigendecomposition of the Gram matrix, which is much faster on GPU. `"svd"` uses `torch.linalg.svd` as in the MATLAB code (same result, slower).
   - `use_GPU = 1` to use GPU, `use_fast = 1` to use the fast convergence version
2. Run
   ```bash
   uv run python main_S3TTV_Python.py
   ```
   The settings can also be overridden from the command line, e.g.,
   ```bash
   uv run python main_S3TTV_Python.py --image PaviaUniversity --maxiter 5000
   uv run python main_S3TTV_Python.py --cpu --standard   # CPU, standard algorithm
   uv run python main_S3TTV_Python.py --help             # list of all options
   ```
3. MPSNR and MSSIM are displayed, and the restored HSI, the removed noise components, the convergence history, and a figure are saved in `result_python/S3TTV_<image>.mat` and `.png`.

Notes:
- MATLAB and NumPy use different random number generators, so the generated noise is statistically the same but not identical to that of `main_S3TTV.m`. To restore exactly the same observation as in MATLAB, save `HSI_noisy` to a `.mat` file in MATLAB and pass it with `--noisy_mat <file>.mat`.
- The S3TTV solver can also be called directly from Python:
  ```python
  from S3TTV_fast_python import S3TTV_fast
  HSI_restored, removed_noise, iteration, converge_rate_U = S3TTV_fast(HSI_noisy, params, device="cuda")
  ```
  where `params` is a dict with `alpha`, `beta`, `epsilon`, `blocksize`, `maxiter`, `stopcri` (see `main_S3TTV_Python.py`).


## Our Reference
If you use this code, please cite the following paper:

```
@ARTICLE{takemoto2025spatiospectral,
  author={Takemoto, Shingo and Naganuma, Kazuki and Ono, Shunsuke},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Spatio-Spectral Structure Tensor Total Variation for Hyperspectral Image Denoising and Destriping}, 
  year={2025},
  volume={18},
  number={},
  pages={19157-19175},
  doi={10.1109/JSTARS.2025.3586779}
}
```


## Comparison with existing methods
This repository also supports comparison with several existing denoising and destriping methods.

### QRNN3D

To evaluate **QRNN3D** [1], follow the steps below:

1. Download the official code from [https://github.com/Vandermode/QRNN3D](https://github.com/Vandermode/QRNN3D?tab=readme-ov-file)

2. Download the fine-tuned checkpoint (Pavia Centre) from [Google Drive](https://drive.google.com/file/d/1o1R3PVZhsJzbpJjRHlhMGp8fH6Ua4XV-/view?usp=drive_link)

3. Run `hsi_test.py`.


### Other conventional methods

To compare with other methods (**SSTV** [2], **HSSTV** [3], **l0-l1HTV** [4], **STV** [5], **SSST** [6], **LRTDTV** [7], **FGSLR** [8], **TPTV** [9], and **FastHyMix** [10]):

1. Download and extract the following repositories, and place each extracted folder into the `compared_methods/` directory:

    - **LRTDTV**  
      [https://github.com/zhaoxile/Hyperspectral-Image-Restoration-via-Total-Variation-Regularized-Low-rank-Tensor-Decomposition](https://github.com/zhaoxile/Hyperspectral-Image-Restoration-via-Total-Variation-Regularized-Low-rank-Tensor-Decomposition)

    - **FGSLR**  
      [https://chenyong1993.github.io/yongchen.github.io/](https://chenyong1993.github.io/yongchen.github.io/)

    - **TPTV**  
      [https://github.com/chuchulyf/ETPTV](https://github.com/chuchulyf/ETPTV)

    - **FastHyMix**  
      [https://github.com/LinaZhuang/HSI-MixedNoiseRemoval-FastHyMix](https://github.com/LinaZhuang/HSI-MixedNoiseRemoval-FastHyMix)

2. Run `main_with_comparisons.m`
 - Choose the target image (`JasperRidge`, `PaviaUniversity`, or `Beltsville`)
 - Enable each method by setting `"enable"` to `true` in the corresponding section
 - Adjust parameters for each method

3. Run the result visualization script: `plot_result.m`
- Use `show_band` to select the band index for visualization
- The script compares results in `result/<condition>/<name_method>/` and selects the best result for each method



---

## References

```bibtex
[1] K. Wei, Y. Fu, and H. Huang, ``3-D quasi-recurrent neural network for hyperspectral image denoising,'' IEEE Trans. Neural Netw. Learn. Syst., vol. 32, no. 1, pp. 363--375, 2021.

[2] H. K. Aggarwal and A. Majumdar, ``Hyperspectral image denoising using spatio-spectral total variation,'' IEEE Geosci. Remote Sens. Lett., vol. 13, no. 3, pp. 442--446, 2016.

[3] S. Takeyama, S. Ono, and I. Kumazawa, ``A constrained convex optimization approach to hyperspectral image restoration with hybrid spatio-spectral regularization,'' Remote Sens., vol. 12, no. 21, 2020.

[4] M. Wang, Q. Wang, J. Chanussot, and D. Hong, ``$l_0$-$l_1$ hybrid total variation regularization and its applications on hyperspectral image mixed noise removal and compressed sensing,'' IEEE Trans. Geosci. Remote Sens., vol. 59, no. 9, pp. 7695--7710, 2021.

[5] S. Lefkimmiatis, A. Roussos, P. Maragos, and M. Unser, ``Structure tensor total variation,'' SIAM J. Imag. Sci., vol. 8, no. 2, pp. 1090--1122, 2015.

[6] R. Kurihara, S. Ono, K. Shirai, and M. Okuda, ``Hyperspectral image restoration based on spatio-spectral structure tensor regularization,'' in Proc. Eur. Signal Process. Conf. (EUSIPCO), 2017, pp. 488--492.

[7] Y. Wang, J. Peng, Q. Zhao, Y. Leung, X. Zhao, and D. Meng, ``Hyperspectral image restoration via total variation regularized low-rank tensor decomposition,'' IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., vol. 11, no. 4, pp. 1227--1243, 2018.

[8] Y. Chen, T. Huang, W. He, X. Zhao, H. Zhang, and J. Zeng, ``Hyperspectral image denoising using factor group sparsity-regularized nonconvex low-rank approximation,'' IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 1--16, 2022.

[9] Y. Chen, W. Cao, L. Pang, J. Peng, and X. Cao, ``Hyperspectral image denoising via texture-preserved total variation regularizer,'' IEEE Trans. Geosci. Remote Sens., vol. 61, pp. 1--14, 2023.

[10] L. Zhuang and M. K. Ng, ``FastHyMix: Fast and parameter-free hyperspectral image mixed noise removal,'' IEEE Trans. Neural Netw. Learn. Syst., vol. 34, no. 8, pp. 4702--4716, 2023.
