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
    [w_center, w_i, lambda_F, lco_amp, flutter_onset_idx, ...
        natural_frequencies_hz_array, damping_array, unstable, max_real_eig, ...
        vm_upper, vm_lower] = pressure_sweep( ...
        params, psi_w, psi_w_xx, psi_w_yy, psi_w_xy, struct_mat_B2, ...
        struct_mat_K, struct_mat_Aw_not_scaled, struct_mat_Minv, ...
        NModes_w, struct_mat_L2, struct_mat_Awdot_not_scaled);

    solve_data = struct( ...
        'w_center', w_center, ...
        'w_i', w_i, ...
        'lambda_F', lambda_F, ...
        'lco_amp', lco_amp, ...
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
    params.disc_pressure    = 50;
    params.pinf_sweep = linspace(0, 108e3, params.disc_pressure); % [Pa]
    params.gamma = 1.4;
    params.Minf = 4.0;
    params.T0 = 400; % [K], for aerodynamic damping nondimensionalization

    params.lambda   = params.gamma * params.pinf_sweep * params.Minf *...
        (a^3 / D);
    params.tol      = 1e-15; % dimensionless safety factor
    
    sf = 2; sigma_y = 450*10^6;
    params.sf_rel = sigma_y/sf;
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
        [vmU_local, vmL_local] = von_mises( ...
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


function plot_output(params, plot_data)

    style = apply_paper_plot_settings();

    lambda = plot_data.lambda;
    t_eval = plot_data.t_eval;
    pinf   = plot_data.pinf;
    a      = plot_data.a;
    b      = plot_data.b;
    h      = plot_data.h;
    w_center      = plot_data.w_center;
    A_LCO         = plot_data.A_LCO;
    lambda_F      = plot_data.lambda_F;
    flutter_onset_idx = plot_data.flutter_idx;
    reduced_freq_array = plot_data.reduced_freq_array;
    damping_array = plot_data.damping_array;
    vm_max_p      = plot_data.vm_max_p;
    x_max         = plot_data.x_max_vm;
    y_max         = plot_data.y_max_vm;

    stress_cr = vm_max_p / params.sf_rel;

    % ---------------- w_center(t)/h for selected lambdas in a separate window ----------------
    figure('Color', style.figureColor, 'Position', style.figurePosition);
    tiledlayout(1,3,'TileSpacing',style.tileSpacing,'Padding',style.tilePadding);

    selected_idx = pick_lambda_indices(numel(lambda), flutter_onset_idx);
    for k = 1:numel(selected_idx)
        idx = selected_idx(k);
        nexttile; hold on; grid off;
        plot(t_eval, w_center(idx,:)/h, 'LineWidth', style.lineWidth);
        set(gca,'FontSize',style.axesFontSize);
        xlabel('$t [sec]$','Interpreter','latex','FontSize',style.labelFontSize);
        ylabel('$w_{center}/h$','Interpreter','latex','FontSize',style.labelFontSize);
        title(sprintf('$\lambda = %.1f$', lambda(idx)), 'Interpreter', 'latex', ...
            'FontSize', style.titleFontSize);
        xlim([0, 0.115*t_eval(end)]);
    end

    figure('Color', style.figureColor, 'Position', style.figurePosition);
    tiledlayout(2,2,'TileSpacing',style.tileSpacing,'Padding',style.tilePadding);

    % ---------------- damping vs lambda ----------------
    nexttile; hold on; grid off;

    scatter(lambda, damping_array, style.scatterSizeLarge, '.', 'MarkerEdgeAlpha', 1);
    set(gca,'FontSize',style.axesFontSize);
    xlim([0, 1300]);
    xlabel('$\lambda$','Interpreter','latex','FontSize',style.labelFontSize);
    ylabel('$\zeta$','Interpreter','latex','FontSize',style.labelFontSize);

    % ---------------- lambda vs LCO amp ----------------
    nexttile; hold on; grid off;

    plot(lambda, A_LCO/h, '-o', 'LineWidth', style.lineWidth, 'MarkerSize', style.markerSize);
    set(gca,'FontSize',style.axesFontSize);
    xlim([0, max(lambda)]);
    xlabel('$\lambda$','Interpreter','latex','FontSize',style.labelFontSize);
    ylabel('$(w_{center}/h)_{amp.}$','Interpreter','latex','FontSize',style.labelFontSize);

    % ---------------- max VM vs p_inf ----------------
    nexttile; hold on; grid off;
    plot(lambda, stress_cr, '-o', 'LineWidth', style.lineWidth, 'MarkerSize', style.markerSize);
    set(gca,'FontSize',style.axesFontSize);
    xlabel('$\lambda$','Interpreter','latex','FontSize',style.labelFontSize);
    ylabel('$\sigma_{cr}$','Interpreter','latex','FontSize',style.labelFontSize);

    % ---------------- location of max VM on the panel ----------------
    nexttile; hold on; grid on;
    scatter(x_max, y_max, style.scatterSizeMedium, lambda, 'filled');  % color by pressure
    cb = colorbar; cb.Label.String = '$\lambda$';
    cb.Label.Interpreter = 'latex';
    cb.TickLabelInterpreter = 'latex';
    cb.Label.FontSize = style.labelFontSize;
    cb.FontSize = style.axesFontSize;
    set(gca,'FontSize',style.axesFontSize);
    xlabel('$x$ [m]','Interpreter','latex','FontSize',style.labelFontSize);
    ylabel('$y$ [m]','Interpreter','latex','FontSize',style.labelFontSize);
    title('Location of $\max\sigma_{\mathrm{VM}}$','Interpreter','latex', 'FontSize', style.titleFontSize);
    xlim([0, a]); ylim([-b/2, b/2]);

    % highlight flutter-onset location if available
    plot(x_max(flutter_onset_idx), y_max(flutter_onset_idx), 'kp', ...
        'MarkerSize', style.flutterMarkerSize, 'LineWidth', style.highlightLineWidth);
end




function selected_idx = pick_lambda_indices(n_lambda, flutter_onset_idx)
    if n_lambda <= 3
        selected_idx = 1:n_lambda;
        return;
    end

    if ~isempty(flutter_onset_idx) && ~isnan(flutter_onset_idx)
        selected_idx = [max(1, flutter_onset_idx-1), flutter_onset_idx, ...
            min(n_lambda, flutter_onset_idx+1)];
    else
        selected_idx = round(linspace(1, n_lambda, 3));
    end

    selected_idx = unique(selected_idx, 'stable');

    if numel(selected_idx) < 3
        fallback_idx = round(linspace(1, n_lambda, 3));
        selected_idx = unique([selected_idx, fallback_idx], 'stable');
        selected_idx = selected_idx(1:3);
    end
end


function define_parallel_processing()
    p = gcp('nocreate');
    if isempty(p)
        parpool('IdleTimeout', Inf);
    end
end
