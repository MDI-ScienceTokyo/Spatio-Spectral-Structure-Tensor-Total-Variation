%% Spatio-Spectral Structure Tensor Total Variation for Hyperspectral Image Denoising and Destriping
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Shingo Takemoto (takemoto.s.e908@m.isct.ac.jp)
% Last version: June 15, 2025
% Article: S. Takemoto, K. Naganuma, S. Ono, 
%   ``Spatio-Spectral Structure Tensor Total Variation for Hyperspectral Image Denoising and Destriping''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all
addpath(genpath('sub_functions'))
addpath('solvers')
fprintf('******* initium *******\n');
rng('default')

%% Generating observation
%%%%%%%%%%%%%%%%%%%%% User settings of experiment %%%%%%%%%%%%%%%%%%%%%%%%%%%%
deg.Gaussian_sigma      = 0.1; % Standard derivation of Gaussian noise
deg.sparse_rate         = 0.05; % Rate of sparse noise
deg.stripe_rate         = 0.05; % Rate of stripe noise
deg.stripe_intensity    = 0.5; % Range of intensity for stripe noise

image = 'JasperRidge';
% image = 'PaviaUniversity';
% image = 'Beltsville';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch image
    case 'JasperRidge'
        load('./dataset/JasperRidge.mat');

    case 'PaviaUniversity'
        load('./dataset/PaviaUniversity.mat');

    case 'Beltsville'
        load('./dataset/Beltsville.mat');
end

[HSI_noisy, deg] = Generate_obsv(HSI_clean, deg);

HSI_clean = single(HSI_clean);
HSI_noisy = single(HSI_noisy);

edge_width = 3;


%% Setting parameters
%%%%%%%%%%%%%%%%%%%%% User Settings %%%%%%%%%%%%%%%%%%%%%%%%%%%%
params.rho = 0.95; % Parameter for the radii of the noise terms

params.blocksize = [10,10]; % Block size of S3TTV

params.stopcri = 1e-5; % Stopping criterion
params.maxiter = 20000; % Maximum number of iterations

use_GPU = 1; % 1, if you use GPU, 0, if you use CPU
use_fast = 1; % 1, if you use Fast algorithm, 0, if you use Standard algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Radius of v-centered l2-ball constraint serving data-fidelity
params.epsilon = params.rho * deg.Gaussian_sigma * sqrt(hsi.N * (1 - deg.sparse_rate));

% Radius of l1-ball constraint for sparse noise
params.alpha = params.rho * (0.5 * hsi.N * deg.sparse_rate);

% Radius of l1-ball constraint for stripe noise
params.beta = params.rho * hsi.N * (1 - deg.sparse_rate) ...
    * deg.stripe_rate * deg.stripe_intensity / 2; 


%% Showing settings
fprintf('~~~ SETTINGS ~~~\n');
fprintf('Image: %s Size: (%d, %d, %d)\n', image, hsi.n1, hsi.n2, hsi.n3);
fprintf('Gaussian sigma: %g\n', deg.Gaussian_sigma);
fprintf('Sparse rate: %g\n', deg.sparse_rate);
fprintf('Stripe rate: %g\n', deg.stripe_rate);
fprintf('Stripe intensity: %g\n', deg.stripe_intensity);
fprintf('Rho: %g\n', params.rho);
fprintf('Stopping criterion: %g\n', params.stopcri);
fprintf('Blocksize: (%d, %d)\n', params.blocksize(1), params.blocksize(2))


%% Denoising and destriping
if use_fast == 1
    if use_GPU == 1
        [HSI_restored, ~, ~, ~] = S3TTV_GPU_fast(HSI_noisy, params); % for GPU user
    elseif use_GPU == 0
        [HSI_restored, ~, ~, ~] = S3TTV_CPU_fast(HSI_noisy, params); % for CPU user
    else
    end
elseif use_fast == 0
    if use_GPU == 1
        [HSI_restored, ~, ~, ~] = S3TTV_GPU(HSI_noisy, params); % for GPU user
    elseif use_GPU == 0
        [HSI_restored, ~, ~, ~] = S3TTV_CPU(HSI_noisy, params); % for CPU user
    else
    end
end


%% Plotting results
mpsnr_S3TTV  = calc_MPSNR(HSI_restored(:, :, edge_width+1:end-edge_width), HSI_clean(:, :, edge_width+1:end-edge_width));
mssim_S3TTV  = calc_MSSIM(HSI_restored(:, :, edge_width+1:end-edge_width), HSI_clean(:, :, edge_width+1:end-edge_width));

fprintf('~~~ RESULTS ~~~\n');
fprintf('MPSNR: %#.4g\n', mpsnr_S3TTV);
fprintf('MSSIM: %#.4g\n', mssim_S3TTV);


%% Showing result images
visband = 59; % band to visualize

figure(1);
subplot(1,3,1), imshow(HSI_clean(:,:,visband)), title('Ground-truth');
subplot(1,3,2), imshow(HSI_noisy(:,:,visband)), title('Noisy HSI');
subplot(1,3,3), imshow(HSI_restored(:,:,visband)), title('Restored HSI');

fprintf('******* finis *******\n');
