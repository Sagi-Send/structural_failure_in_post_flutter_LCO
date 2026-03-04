clc;
clear;
close all;

define_parallel_processing();

%% Build structural model
% Geometry and material parameters are defined inside this script.
create_AiryStressPlateModel

%% Analysis setup
params = build_analysis_params(NModes_w, xMesh, yMesh, a, b, D);

% add what VM needs (keep minimal: just store constants already in workspace)
params.h  = h;
params.nu = nu;
params.D  = D;

%% Pressure sweep
[w_center, w_i, lambda_F, lco_amp, flutter_onset_idx, ...
    natural_frequencies_hz_array, damping_array, unstable, max_real_eig, ...
    vm_upper, vm_lower] = pressure_sweep( ...
    params, psi_w, psi_w_xx, psi_w_yy, psi_w_xy, struct_mat_B2, ...
    struct_mat_K, struct_mat_Aw_not_scaled, struct_mat_Minv, ...
    NModes_w, struct_mat_L2, struct_mat_Awdot_not_scaled);

if ~isnan(lambda_F)
    fprintf('Flutter onset at lambda = %.3g\n', lambda_F);
else
    fprintf('No flutter detected in the scanned range.\n');
end

reduced_freq_array = nondimentionalize(params, natural_frequencies_hz_array);

plot_output(params, w_center, lco_amp, lambda_F, ...
    flutter_onset_idx, h, reduced_freq_array, damping_array);

% vm_upper / vm_lower are available for post-processing
% e.g., max over time and surfaces:
% vm_max = squeeze(max(cat(4, vm_upper, vm_lower), [], 4)); % [n_pressures x nPts x Nt]
% vm_max_over_time = squeeze(max(vm_max, [], 3));           % [n_pressures x nPts]


function params = build_analysis_params(NModes_w, xMesh, yMesh, a, b, D)
    params.a = a;   params.b = b;
    params.T_max_nonlinear_solution = 9;
    params.Nt = 900;                    % A Nt/T=100 ratio looks best.
    params.t_eval = linspace(0, params.T_max_nonlinear_solution, params.Nt);

    q0 = zeros(NModes_w,1);
    q0(1) = 1e-6;                       % tiny displacement perturbation
    params.q_qdot_ics = [q0; zeros(NModes_w,1)];

    params.x_points = reshape(xMesh, 1, []);
    params.y_points = reshape(yMesh, 1, []);

    params.disc_stress      = 10;
    params.disc_pressure    = 50;
    params.pinf_sweep = linspace(0, 75e3, params.disc_pressure); % [Pa]
    params.gamma = 1.4;
    params.Minf = 4.0;
    params.T0 = 400; % [K], for aerodynamic damping nondimensionalization

    params.lambda = params.gamma * params.pinf_sweep * params.Minf * (a^3 / D);
    params.tol = 1e-15; % dimensionless safety factor
end


function [w_center, w_i, lambda_F, lco_amps, first_unstable_idx, ...
    natural_frequencies_hz_array, damping_array, unstable, max_real_eig, ...
    vm_upper, vm_lower] = ...
    pressure_sweep(params, psi_w, psi_w_xx, psi_w_yy, psi_w_xy, struct_mat_B2, ...
    struct_mat_K, struct_mat_Aw_not_scaled, struct_mat_Minv, NModes_w, ...
    struct_mat_L2, struct_mat_Awdot_not_scaled)

    a          = params.a;  b = params.b;
    h          = params.h;
    nu         = params.nu;
    D          = params.D;

    pinf_sweep = params.pinf_sweep;
    lambda     = params.lambda;
    gamma      = params.gamma;
    Minf       = params.Minf;
    tol        = params.tol;
    t_eval     = params.t_eval;
    q_qdot_ics = params.q_qdot_ics;
    T0         = params.T0;

    % Stress evaluation grid (disc_stress x disc_stress)
    x_lin = linspace(0, a, params.disc_stress);
    y_lin = linspace(-b/2, b/2, params.disc_stress);
    [Xg, Yg] = meshgrid(x_lin, y_lin);
    x_points = reshape(Xg, 1, []);
    y_points = reshape(Yg, 1, []);

    n_pressures = numel(pinf_sweep);
    Nt          = numel(t_eval);
    nPts        = numel(x_points);

    % Preallocation
    w_center                     = zeros(n_pressures, Nt);
    w_i                          = zeros(n_pressures, nPts, Nt);
    vm_upper                     = zeros(n_pressures, nPts, Nt);
    vm_lower                     = zeros(n_pressures, nPts, Nt);

    natural_frequencies_hz_array = zeros(NModes_w, n_pressures);
    damping_array                = zeros(NModes_w, n_pressures);
    max_real_eig                 = zeros(1, n_pressures);
    unstable                     = false(1, n_pressures);
    lco_amps                     = zeros(1, n_pressures);

    % Find center point index (nearest)
    [~, i_center] = min((x_points - a/2).^2 + (y_points - 0).^2);

    parfor idx = 1:n_pressures
        pinf_i = pinf_sweep(idx);

        [struct_mat_Aw, struct_mat_Awdot] = aerodynamic_stiffness_damping( ...
            struct_mat_Aw_not_scaled, struct_mat_Awdot_not_scaled, ...
            pinf_i, gamma, Minf, T0);

        rhs_local = @(t, y) rhs_func_aero( ...
            t, y, NModes_w, struct_mat_Minv, struct_mat_K, struct_mat_L2, ...
            struct_mat_Aw, struct_mat_Awdot);

        [~, w_modal] = ode45(rhs_local, t_eval, q_qdot_ics);
        Q = w_modal(:, 1:NModes_w); % [Nt x NModes_w]

        % deflection at all points for this pressure
        w_i_local = zeros(nPts, Nt);
        for it = 1:Nt
            w_i_local(:, it) = modal2physical( ...
                Q(it, :), x_points, y_points, psi_w).';
        end

        % store full-field + center trace
        w_i(idx,:,:)    = reshape(w_i_local, [1, nPts, Nt]);
        w_center_local  = w_i_local(i_center, :);
        w_center(idx,:) = w_center_local;

        lco_amps(idx) = estimate_lco_amplitude(t_eval, w_center_local, 0.8);

        % VM stresses on upper/lower surfaces at all points and all times
        [vmU_local, vmL_local] = compute_vm_surfaces( ...
            Q, x_points, y_points, psi_w_xx, psi_w_yy, psi_w_xy, ...
            struct_mat_B2, h, nu, D);

        vm_upper(idx,:,:) = reshape(vmU_local, [1, nPts, Nt]);
        vm_lower(idx,:,:) = reshape(vmL_local, [1, nPts, Nt]);

        struct_mat_K_total = struct_mat_K + struct_mat_Aw;
        struct_mat_C       = struct_mat_Awdot;

        [natural_frequencies_hz_array(:, idx), damping_array(:, idx), ...
            max_real_eig(idx), omega_scale] = ...
            solve_coupled_eigensystem(struct_mat_Minv, struct_mat_K_total, ...
                                      struct_mat_C, NModes_w);

        unstable(idx) = max_real_eig(idx) > (omega_scale * tol);
    end

    first_unstable_idx = find(unstable, 1, 'first');
    if isempty(first_unstable_idx)
        lambda_F = nan;
    else
        lambda_F = lambda(first_unstable_idx);
    end
end


function [vm_upper, vm_lower] = compute_vm_surfaces( ...
    Q, x_points, y_points, psi_w_xx, psi_w_yy, psi_w_xy, ...
    struct_mat_B2, h, nu, D)

    % Q: [Nt x N]
    [Nt, N] = size(Q);
    nPts = numel(x_points);

    % basis second-derivative matrices at points (N x nPts)
    Psi_xx = zeros(N, nPts);
    Psi_yy = zeros(N, nPts);
    Psi_xy = zeros(N, nPts);
    for n = 1:N
        Psi_xx(n,:) = psi_w_xx{n}(x_points, y_points);
        Psi_yy(n,:) = psi_w_yy{n}(x_points, y_points);
        Psi_xy(n,:) = psi_w_xy{n}(x_points, y_points);
    end

    % curvatures at points over time (nPts x Nt)
    w_xx = (Q * Psi_xx).';
    w_yy = (Q * Psi_yy).';
    w_xy = (Q * Psi_xy).';

    % bending moments (nPts x Nt)
    Mxx = -D * (w_xx + nu * w_yy);
    Myy = -D * (w_yy + nu * w_xx);
    Mxy = -D * (1 - nu) * w_xy;

    % Airy coefficients c(t): F(x,y,t) = sum c_n(t) psi_n(x,y)
    % c = (A^{-1}B) : (q ⊗ q) = struct_mat_B2(q,q)
    Ccoef = zeros(Nt, N);
    for it = 1:Nt
        q = Q(it,:).';                              % [N x 1]
        tmp = tensorprod(struct_mat_B2, q, 3, 1);    % -> [N x N]
        c   = tensorprod(tmp, q, 2, 1);              % -> [N x 1]
        Ccoef(it,:) = c.';
    end

    % Airy second derivatives at points over time (nPts x Nt)
    F_xx = (Ccoef * Psi_xx).';
    F_yy = (Ccoef * Psi_yy).';
    F_xy = (Ccoef * Psi_xy).';

    % stress resultants from Airy
    Nxx = F_yy;
    Nyy = F_xx;
    Nxy = -F_xy;

    % membrane stresses
    sxx_m = Nxx / h;
    syy_m = Nyy / h;
    sxy_m = Nxy / h;

    % bending stresses at z = ±h/2
    coef = 6 / h^2;   % (12z/h^3) with z=±h/2
    sxx_b = coef * Mxx;
    syy_b = coef * Myy;
    sxy_b = coef * Mxy;

    % upper (+h/2) and lower (-h/2)
    sxxU = sxx_m + sxx_b;   syyU = syy_m + syy_b;   sxyU = sxy_m + sxy_b;
    sxxL = sxx_m - sxx_b;   syyL = syy_m - syy_b;   sxyL = sxy_m - sxy_b;

    % von Mises (plane stress)
    vm_upper = sqrt(sxxU.^2 - sxxU.*syyU + syyU.^2 + 3*sxyU.^2);
    vm_lower = sqrt(sxxL.^2 - sxxL.*syyL + syyL.^2 + 3*sxyL.^2);
end


function [struct_mat_Aw, struct_mat_Awdot] = aerodynamic_stiffness_damping ...
    (struct_mat_Aw_not_scaled, struct_mat_Awdot_not_scaled, pinf_i, gamma, Minf, T0)
    coeff_Aw = gamma * pinf_i * Minf;
    struct_mat_Aw = coeff_Aw * struct_mat_Aw_not_scaled;

    Rgas = 287;                 % [J/(kg*K)]
    a_inf = sqrt(gamma * Rgas * T0);

    coeff_Awdot = gamma * pinf_i / a_inf;
    struct_mat_Awdot = coeff_Awdot * struct_mat_Awdot_not_scaled;
end

function [natural_frequencies_hz, damping, max_real_eig, omega_scale] = ...
    solve_coupled_eigensystem(struct_mat_Minv, struct_mat_K_total, struct_mat_C, NModes_w)

    A = [zeros(NModes_w), eye(NModes_w); ...
         -struct_mat_Minv * struct_mat_K_total, -struct_mat_Minv * struct_mat_C];

    eigvals_all = eig(A);
    max_real_eig = max(real(eigvals_all));
    omega_scale = max(1, max(abs(imag(eigvals_all))));

    positive_frequency_mask = imag(eigvals_all) > 0;
    eigvals = eigvals_all(positive_frequency_mask);

    [omega_rad_s, sort_idx] = sort(imag(eigvals));
    sigma = real(eigvals(sort_idx));

    natural_frequencies_hz = omega_rad_s / (2 * pi);
    damping = sigma ./ omega_rad_s;
end

% Estimate LCO amplitude from a scalar time series w(t).
function lco_amp = estimate_lco_amplitude(t, w, transientFrac)
    if nargin < 3 || isempty(transientFrac), transientFrac = 0.33; end

    Nt = numel(t);
    i0 = max(1, floor(transientFrac*Nt) + 1); % index where steady window starts
    w_ss = w(i0:end);

    lco_amp = 0.5*(max(w_ss) - min(w_ss));
end


function plot_output ...
    (params, w_center, A_LCO, lambda_F, flutter_onset_idx, h, ...
    reduced_freq_array, damping_array)
    lambda = params.lambda;
    t_eval = params.t_eval;

    figure;
    tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

    % ---------------- w_center(t)/h for selected lambdas ----------------
    nexttile; hold on; grid off;

    n_show   = min(400, numel(lambda));
    idx_show = unique(round(linspace(1, numel(lambda), n_show)));

    for k = 1:numel(idx_show)
        i = idx_show(k);
        if abs(lambda(i) - 1147.26) < 1
            plot(t_eval, w_center(i,:)/h, 'LineWidth', 1.5, ...
                'DisplayName', sprintf('$\\lambda = %.1f$', lambda(i)));
        end
    end

    set(gca,'FontSize',18);
    xlabel('$t$','Interpreter','latex','FontSize',24);
    ylabel('$w_{center}/h$','Interpreter','latex','FontSize',24);
    legend('show','Interpreter','latex','Location','best');
    xlim([0, t_eval(end)]);

    % ---------------- damping vs lambda ----------------
    nexttile; hold on; grid off;

    scatter(lambda, damping_array, 300, '.', 'MarkerEdgeAlpha', 1);
    set(gca,'FontSize',18);
    xlim([0, 1300]);
    xlabel('$\lambda$','Interpreter','latex','FontSize',24);
    ylabel('$\zeta$','Interpreter','latex','FontSize',24);

    % ---------------- lambda vs LCO amp ----------------
    nexttile; hold on; grid off;

    plot(lambda, A_LCO/h, '-o');
    set(gca,'FontSize',18);
    xlim([0, 1300]);
    xlabel('$\lambda$','Interpreter','latex','FontSize',24);
    ylabel('$(w_{center}/h)_{amp.}$','Interpreter','latex','FontSize',24);
end


function reduced_freq_array = nondimentionalize(params, natural_frequencies_hz_array)
    a     = params.a;
    gamma = params.gamma;
    T0    = params.T0;
    Minf  = params.Minf;

    Rgas = 287;                 % [J/(kg*K)]
    a_inf = sqrt(gamma * Rgas * T0);
    Uinf = Minf * a_inf;
    Lref = a;

    reduced_freq_array = (2 * pi * natural_frequencies_hz_array) * (Lref / Uinf);
end


function define_parallel_processing()
    p = gcp('nocreate');
    if isempty(p)
        parpool('IdleTimeout', Inf);
    end
end