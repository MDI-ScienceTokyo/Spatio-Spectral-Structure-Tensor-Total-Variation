%% Spatio-Spectral Structure Tensor Total Variation for Hyperspectral Image Denoising and Destriping
%% =========================== First part notes===========================
% Author: Shingo Takemoto (takemoto.s.e908@m.isct.ac.jp)
% Last version: June 15, 2025
% Article: S. Takemoto, K. Naganuma, S. Ono, 
%   ``Spatio-Spectral Structure Tensor Total Variation for Hyperspectral Image Denoising and Destriping''
% -------------------------------------------------------------------------
%% =========================== Second part notes =========================== 
% INPUT:
%   HSI_noisy: noisy hyperspectral image of size n1*n2*n3 normalized to [0,1]
%   params: an option structure whose fields are as follows:           
%       alpha: radius of l_1 ball for sparse noise
%       beta: radius of l_1 ball for stripe noise
%       epsilon: radius of l_2 ball serving data-fidelity
%       blocksize: parameter of block size for spatio-spectral structure tensor
%       max_iter: maximum number of iterations
%       stop_cri: stopping criterion of P-PDS
%       disprate: Period to display intermediate results
% OUTPUT:
%   restored_HSI: denoised hyperspectral image
%   removed_noise: removed noise
%   iteration: number of P-PDS iteration
%  ========================================================================

function [HSI_restored, removed_noise, iteration, converge_rate_U] ...
     = S3TTV_CPU_fast(HSI_noisy, params)
fprintf('** Running S3TTV_CPU_fast **\n');
HSI_noisy = single(HSI_noisy);
[n1, n2, n3] = size(HSI_noisy);

alpha       = single(params.alpha);
beta        = single(params.beta);
epsilon     = single(params.epsilon);
blocksize   = single(params.blocksize);
maxiter     = single(params.maxiter);
stopcri     = single(params.stopcri);

%% Setting params
disprate    = gpuArray(single(100));


%% Initializing primal and dual variables

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% primal variables
% U: clean HSI
% S: sparse noise(salt-and-pepper noise)
% T: stripe noise
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

U = zeros([n1, n2, n3], 'single');
S = zeros([n1, n2, n3], 'single');
T = zeros([n1, n2, n3], 'single');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dual variables
% Y1: term of S3TTV
% Y2: term of l2ball
% Y3: term of stripe noise
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Y1 = zeros([n1, n2, n3, 2, blocksize(1), blocksize(2)], 'single');
Y2 = zeros([n1, n2, n3], 'single');
Y3 = zeros([n1, n2, n3], 'single');
Y4 = zeros([n1, n2, n3], 'single');
Y5 = zeros([n1, n2, n3], 'single');


%% Setting operators
% Difference operators
D       = @(z) cat(4, z([2:end, 1],:,:) - z, z(:,[2:end, 1],:) - z);
Dt      = @(z) z([end,1:end-1],:,:,1) - z(:,:,:,1) + z(:,[end,1:end-1],:,2) - z(:,:,:,2);
Dv      = @(z) z([2:end, 1],:,:) - z;
Dvt     = @(z) z([end,1:end-1],:,:) - z(:,:,:);
Ds      = @(z) z(:, :, [2:end, 1], :) - z;
Dst     = @(z) z(:,:,[end,1:end-1],:) - z(:,:,:,:);

% Expansion operators
P = @(z) func_PeriodicExpansion(z, blocksize);
Pt = @(z) func_PeriodicExpansionTrans(z);


%% Setting stepsize parameters for P-PDS
gamma1_U    = single(1./(prod(blocksize) * 2*2 * 2 + 1)); % P*D*Ds + I
gamma1_S    = single(1);
gamma1_T    = single(1/(2 + 1));
gamma2_Y1   = single(1/(2*2));
gamma2_Y2   = single(1);
gamma2_Y3   = single(1);
gamma2_Y4   = single(1);
gamma2_Y5   = single(1/2);


%% main loop (P-PDS)
converge_rate_U = zeros([1, maxiter], 'single');
fprintf('~~~ P-PDS STARTS ~~~\n');

for i = 1:maxiter   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating U, S, T
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    U_tmp   = U - gamma1_U.*(Dst(Dt(Pt(Y1))) + Y2);
    S_tmp   = S - gamma1_S.*Y3;
    T_tmp   = T - gamma1_T.*(Y4 + Dvt(Y5));

    Primal_sum = U_tmp + S_tmp + T_tmp;
    Primal_sum = ProjL2ball(Primal_sum, HSI_noisy, epsilon) - Primal_sum;

    U_next = U_tmp + Primal_sum/3;
    S_next = S_tmp + Primal_sum/3;
    T_next = T_tmp + Primal_sum/3;

    U_res = 2*U_next - U;
    S_res = 2*S_next - S;
    T_res = 2*T_next - T;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating Y1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Y1_tmp  = Y1 + gamma2_Y1.*(P(D(Ds(2*U_next - U))));
    Y1_next = Y1_tmp - gamma2_Y1.*Prox_S3TTV(Y1_tmp./gamma2_Y1, 1./gamma2_Y1, blocksize);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating Y2
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Y2_tmp  = Y2 + gamma2_Y2.*U_res;
    Y2_next = Y2_tmp - gamma2_Y2*ProjBox(Y2_tmp/gamma2_Y2, 0, 1);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating Y3
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Y3_tmp  = Y3 + gamma2_Y3.*S_res;
    Y3_next = Y3_tmp - gamma2_Y3.*ProjFastL1Ball(Y3_tmp./gamma2_Y3, alpha);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating Y4
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Y4_tmp  = Y4 + gamma2_Y4.*T_res;
    Y4_next = Y4_tmp - gamma2_Y4*ProjFastL1Ball(Y4_tmp/gamma2_Y4, beta);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating Y5
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Y5_next = Y5 + gamma2_Y5.*Dv(T_res);

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculating error
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    converge_rate_U(i) = norm(U_next(:) - U(:),2)/norm(U(:),2);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updating all variables
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    U   = U_next;
    S   = S_next;
    T   = T_next;
    
    Y1  = Y1_next;
    Y2  = Y2_next;
    Y3  = Y3_next;
    Y4  = Y4_next;
    Y5  = Y5_next;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Convergence checking
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if i>=2 && converge_rate_U(i) < stopcri
        fprintf('Iter: %d, Error: %0.6f.\n', i, converge_rate_U(i));
        break
    end
    if (mod(i, disprate) == 0) % Displaying intermediate results
        fprintf('Iter: %d, Error: %0.6f.\n', i, converge_rate_U(i));
    end
end

fprintf('~~~ P-PDS ENDS ~~~\n');

%% Organizing results for output
HSI_noisy                       = gather(HSI_noisy);
HSI_restored                    = gather(U);
iteration                       = gather(i);
removed_noise.sparse_noise      = gather(S);
removed_noise.stripe_noise      = gather(T);
removed_noise.gaussian_noise    = HSI_noisy - HSI_restored - ...
                                    removed_noise.sparse_noise - removed_noise.stripe_noise;
converge_rate_U                 = gather(converge_rate_U(1:iteration));
