clc;
clear;
close all;

define_parallel_processing();

%% Caching controls
results_mat_file = 'flutter_analysis_cache.mat';
force_resolve = false;

%% Build structural model
% Geometry and material parameters are defined inside this script.
create_AiryStressPlateModel

%% Analysis setup
params = build_analysis_params(NModes_w, xMesh, yMesh, a, b, D);

% add what VM needs (keep minimal: just store constants already in workspace)
params.h  = h;
params.nu = nu;
params.D  = D;

%% Pressure sweep (or load cached results)
[cache_loaded, lambda_F, plot_data] = FlutterAnalysisCache.try_load( ...
    results_mat_file, params, force_resolve);

if ~cache_loaded
    [w_center, w_i, lambda_F, amp_transient, amp_steady, flutter_onset_idx, ...
        natural_frequencies_hz_array, damping_array, unstable, max_real_eig, ...
        vm_upper, vm_lower] = pressure_sweep( ...
        params, psi_w, psi_w_xx, psi_w_yy, psi_w_xy, struct_mat_B2, ...
        struct_mat_K, struct_mat_Aw_not_scaled, struct_mat_Minv, ...
        NModes_w, struct_mat_L2, struct_mat_Awdot_not_scaled);

    solve_data = struct( ...
        'w_center', w_center, ...
        'w_i', w_i, ...
        'lambda_F', lambda_F, ...
        'amp_transient', amp_transient, ...
        'amp_steady', amp_steady, ...
        'flutter_onset_idx', flutter_onset_idx, ...
        'natural_frequencies_hz_array', natural_frequencies_hz_array, ...
        'damping_array', damping_array, ...
        'unstable', unstable, ...
        'max_real_eig', max_real_eig, ...
        'vm_upper', vm_upper, ...
        'vm_lower', vm_lower);

    plot_data = FlutterAnalysisCache.save_with_plot_data( ...
        results_mat_file, params, solve_data);
end

if ~isnan(lambda_F)
    fprintf('Flutter onset at lambda = %.3g\n', lambda_F);
else
    fprintf('No flutter detected in the scanned range.\n');
end

plot_output(params, plot_data);

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
    params.disc_pressure    = 30;
    params.pinf_sweep = linspace(0, 108e3, params.disc_pressure); % [Pa]
    params.gamma = 1.4;
    params.Minf = 4.0;
    params.T0 = 400; % [K], for aerodynamic damping nondimensionalization

    params.lambda   = params.gamma * params.pinf_sweep * params.Minf *...
        (a^3 / D);
    params.tol      = 1e-15; % dimensionless safety factor
    
    sf = 2; sigma_y = 450*10^6;
    params.sf_rel = sigma_y/sf;

    params.trans_frac   = 0.015;
    params.steady_frac  = 0.2;
end


function [w_center, w_i, lambda_F, amp_transient, amp_steady, first_unstable_idx, ...
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
    steady_frac = params.steady_frac;
    trans_frac = params.trans_frac;

    % Stress evaluation grid (disc_stress x disc_stress)
    x_lin = linspace(0, a, params.disc_stress);
    y_lin = linspace(-b/2, b/2, params.disc_stress);
    [Xg, Yg] = meshgrid(x_lin, y_lin);
    x_points = reshape(Xg, 1, []);
    y_points = reshape(Yg, 1, []);

    % Precompute modal shape values once on the stress grid
    Psi_w  = build_shape_matrix(psi_w,    x_points, y_points);
    Psi_xx = build_shape_matrix(psi_w_xx, x_points, y_points);
    Psi_yy = build_shape_matrix(psi_w_yy, x_points, y_points);
    Psi_xy = build_shape_matrix(psi_w_xy, x_points, y_points);

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
    amp_transient                = zeros(1, n_pressures);
    amp_steady                   = zeros(1, n_pressures);

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

        % deflection at all points for this pressure (vectorized)
        w_i_local = modal2physical(Q, Psi_w).';

        % store full-field + center trace
        w_i(idx,:,:)    = reshape(w_i_local, [1, nPts, Nt]);
        w_center_local  = w_i_local(i_center, :);
        w_center(idx,:) = w_center_local;

        amp_transient(idx) = estimate_window_amplitude(t_eval, w_center_local, [0, trans_frac]);
        amp_steady(idx)    = estimate_window_amplitude(t_eval, w_center_local, [1-steady_frac, 1.0]);

        % VM stresses on upper/lower surfaces at all points and all times
        [vmU_local, vmL_local] = von_mises( ...
            Q, Psi_xx, Psi_yy, Psi_xy, struct_mat_B2, h, nu, D);

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

    opts = struct('tol', 1e-10, 'maxit', 500, 'issym', false, 'isreal', false);
    [~, D_right, flag_right] = eigs(A, 1, 'lr', opts);

    if flag_right == 0 && all(isfinite(diag(D_right)))
        max_real_eig = real(D_right(1,1));
    else
        max_real_eig = max(real(eigvals_all));
    end
    omega_scale = max(1, max(abs(imag(eigvals_all))));

    positive_frequency_mask = imag(eigvals_all) > 0;
    eigvals = eigvals_all(positive_frequency_mask);

    [omega_rad_s, sort_idx] = sort(imag(eigvals));
    sigma = real(eigvals(sort_idx));

    natural_frequencies_hz = omega_rad_s / (2 * pi);
    damping = sigma ./ omega_rad_s;
end

function Psi = build_shape_matrix(psi_cell, x_points, y_points)
    N = numel(psi_cell);
    nPts = numel(x_points);
    Psi = zeros(N, nPts);
    for n = 1:N
        Psi(n,:) = psi_cell{n}(x_points, y_points);
    end
end

% Estimate displacement amplitude from a selected [startFrac, endFrac] window.
function amp = estimate_window_amplitude(t, w, frac_window)
    idx_window = select_time_window_indices(t, frac_window(1), frac_window(2));
    w_window = w(idx_window);
    amp = 0.5*(max(w_window) - min(w_window));
end

function idx_window = select_time_window_indices(t, startFrac, endFrac)
    Nt = numel(t);
    i_start = max(1, floor(startFrac*Nt) + 1);
    i_end   = min(Nt, max(i_start, floor(endFrac*Nt)));
    idx_window = i_start:i_end;
end


function plot_output(params, plot_data)

    style = apply_paper_plot_settings();

    lambda = plot_data.lambda;
    t_eval = plot_data.t_eval;
    pinf   = plot_data.pinf;
    a      = plot_data.a;
    b      = plot_data.b;
    h      = plot_data.h;
    w_center      = plot_data.w_center;
    A_transient   = plot_data.A_transient;
    A_steady      = plot_data.A_steady;
    damping_array = plot_data.damping_array;
    vm_max_transient = plot_data.vm_max_transient;
    vm_max_steady    = plot_data.vm_max_steady;
    stress_cr_transient = vm_max_transient / params.sf_rel;
    stress_cr_steady    = vm_max_steady / params.sf_rel;

    % ---------------- w_center(t)/h for selected lambdas in a separate window ----------------
    figure('Color', style.figureColor, 'Position', style.figurePosition);
    tiledlayout(2,2,'TileSpacing',style.tileSpacing,'Padding',style.tilePadding);

    selected_idx = [round(numel(lambda)/4), round(numel(lambda)/2), round(0.85*numel(lambda)),numel(lambda)];
    for k = 1:numel(selected_idx)
        idx = selected_idx(k);
        nexttile; hold on; grid off;
        plot(t_eval, w_center(idx,:)/h, 'LineWidth', style.lineWidth);
        set(gca,'FontSize',style.axesFontSize);
        xlabel('$t [sec]$','Interpreter','latex','FontSize',style.labelFontSize);
        ylabel('$w_{center}/h$','Interpreter','latex','FontSize',style.labelFontSize);
        title(sprintf('$\\lambda = %.1f$', lambda(idx)), ...
            'Interpreter','latex', 'FontSize', style.titleFontSize);
        xlim([0, 0.115*t_eval(end)]);
        axis square
    end

    figure('Color', style.figureColor, 'Position', style.figurePosition);
    tiledlayout(1,2,'TileSpacing',style.tileSpacing,'Padding',style.tilePadding);
    
    % ---------------- lambda vs transient amp ----------------
    nexttile; hold on; grid off;

    plot(lambda, A_transient/h, '-o', 'LineWidth', style.lineWidth, 'MarkerSize', style.markerSize);
    set(gca,'FontSize',style.axesFontSize);
    xlim([0, max(lambda)]);
    xlabel('$\lambda$','Interpreter','latex','FontSize',style.labelFontSize);
    ylabel('$(w_{center}/h)_{amp}^{trans.}$','Interpreter','latex','FontSize',style.labelFontSize);
    axis square

    % ---------------- max transient VM vs lambda ----------------
    nexttile; hold on; grid off;
    plot(lambda, stress_cr_transient, '-o', 'LineWidth', style.lineWidth, 'MarkerSize', style.markerSize);
    set(gca,'FontSize',style.axesFontSize);
    xlabel('$\lambda$','Interpreter','latex','FontSize',style.labelFontSize);
    ylabel('$\sigma_{cr}^{trans,}$','Interpreter','latex','FontSize',style.labelFontSize);
    axis square

    sgtitle('Transient window (first 20% of time marching)', ...
        'FontSize', style.titleFontSize, 'FontWeight', 'normal');

    figure('Color', style.figureColor, 'Position', style.figurePosition);
    tiledlayout(1,2,'TileSpacing',style.tileSpacing,'Padding',style.tilePadding);

    % ---------------- lambda vs steady amp ----------------
    nexttile; hold on; grid off;

    plot(lambda, A_steady/h, '-o', 'LineWidth', style.lineWidth, 'MarkerSize', style.markerSize);
    set(gca,'FontSize',style.axesFontSize);
    xlim([0, max(lambda)]);
    xlabel('$\lambda$','Interpreter','latex','FontSize',style.labelFontSize);
    ylabel('$(w_{center}/h)_{amp}^{steady}$','Interpreter','latex','FontSize',style.labelFontSize);
    axis square

    % ---------------- max steady VM vs lambda ----------------
    nexttile; hold on; grid off;
    plot(lambda, stress_cr_steady, '-o', 'LineWidth', style.lineWidth, 'MarkerSize', style.markerSize);
    set(gca,'FontSize',style.axesFontSize);
    xlabel('$\lambda$','Interpreter','latex','FontSize',style.labelFontSize);
    ylabel('$\sigma_{cr}^{steady}$','Interpreter','latex','FontSize',style.labelFontSize);
    axis square

    sgtitle('Steady window (last 20% of time marching)', ...
        'FontSize', style.titleFontSize, 'FontWeight', 'normal');

    figure('Color', style.figureColor, 'Position', style.figurePosition);
    tiledlayout(1,3,'TileSpacing',style.tileSpacing,'Padding',style.tilePadding);

    % ---------------- location of transient max VM on the panel ----------------
    nexttile; hold on; grid on;
    
    xN = plot_data.x_max_vm_transient./a;          % x normalized by panel length a
    yN = plot_data.y_max_vm_transient./b;          % y normalized by panel width  b
    
    scatter(xN, yN, style.scatterSizeMedium, lambda, 'filled');  % color by pressure
    
    cb = colorbar; cb.Label.String = '$\lambda$';
    cb.Label.Interpreter = 'latex';
    cb.TickLabelInterpreter = 'latex';
    cb.Label.FontSize = style.labelFontSize;
    cb.FontSize = style.axesFontSize;
    
    set(gca,'FontSize',style.axesFontSize);
    xlabel('$x/a$','Interpreter','latex','FontSize',style.labelFontSize);
    ylabel('$y/b$','Interpreter','latex','FontSize',style.labelFontSize);
    title('Transient stress hotspot','FontSize',style.titleFontSize);

    xlim([0, 1]);
    ylim([-0.5, 0.5]);
    axis square

    % ---------------- location of steady max VM on the panel ----------------
    nexttile; hold on; grid on;

    xN = plot_data.x_max_vm_steady./a;
    yN = plot_data.y_max_vm_steady./b;

    scatter(xN, yN, style.scatterSizeMedium, lambda, 'filled');

    cb = colorbar; cb.Label.String = '$\lambda$';
    cb.Label.Interpreter = 'latex';
    cb.TickLabelInterpreter = 'latex';
    cb.Label.FontSize = style.labelFontSize;
    cb.FontSize = style.axesFontSize;

    set(gca,'FontSize',style.axesFontSize);
    xlabel('$x/a$','Interpreter','latex','FontSize',style.labelFontSize);
    ylabel('$y/b$','Interpreter','latex','FontSize',style.labelFontSize);
    title('Steady stress hotspot','FontSize',style.titleFontSize);
    
    xlim([0, 1]);
    ylim([-0.5, 0.5]);
    axis square

    % ---------------- damping vs lambda ----------------
    nexttile; hold on; grid off;

    scatter(lambda, damping_array, style.scatterSizeLarge, '.', 'MarkerEdgeAlpha', 1);
    set(gca,'FontSize',style.axesFontSize);
    xlabel('$\lambda$','Interpreter','latex','FontSize',style.labelFontSize);
    ylabel('$\zeta$','Interpreter','latex','FontSize',style.labelFontSize);
    axis square
end


function define_parallel_processing()
    p = gcp('nocreate');
    if isempty(p)
        parpool('IdleTimeout', Inf);
    end
end
