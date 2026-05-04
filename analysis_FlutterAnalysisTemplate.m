clc;
clear;
close all;

%% Caching and animation controls
results_mat_file = 'flutter_analysis_cache.mat';
force_resolve = false;
video_options = build_video_options();

%% Build structural model
% Geometry and material parameters are defined inside this script.
create_AiryStressPlateModel

%% Analysis setup
params = build_analysis_params(NModes_w, xMesh, yMesh, a, b, D);

params.h  = h;
params.nu = nu;
params.D  = D;

%% Pressure sweep (or load cached results)
[cache_loaded, lambda_F, plot_data, q_history] = FlutterAnalysisCache.try_load( ...
    results_mat_file, params, force_resolve);

if ~cache_loaded
    [w_center, w_i, q_history, lambda_F, amp_steady, flutter_onset_idx, ...
        natural_frequencies_hz_array, damping_array, unstable, max_real_eig, ...
        vm_upper, vm_lower] = pressure_sweep( ...
        params, psi_w, psi_w_xx, psi_w_yy, psi_w_xy, struct_mat_B2, ...
        struct_mat_K, struct_mat_Aw_not_scaled, struct_mat_Minv, ...
        NModes_w, struct_mat_L2, struct_mat_Awdot_not_scaled);

    solve_data = struct( ...
        'w_center', w_center, ...
        'w_i', w_i, ...
        'q_history', q_history, ...
        'lambda_F', lambda_F, ...
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

[mode_1_lowest, mode_2_lowest] = compute_lowest_frequency_modes( ...
    struct_mat_Minv, struct_mat_K, psi_w, xMesh, yMesh);
plot_data.mode_shape_1 = mode_1_lowest;
plot_data.mode_shape_2 = mode_2_lowest;

plot_output(params, plot_data, psi_w, xMesh, yMesh);

if video_options.export_mp4
    export_plate_response_video( ...
        params, plot_data, q_history, psi_w, psi_w_xx, psi_w_yy, psi_w_xy, ...
        struct_mat_B2, xMesh, yMesh, video_options);
end

function video_options = build_video_options()
    video_options.export_mp4 = true;
    video_options.mp4_file = 'plate_response_until_yield.mp4';
    video_options.fps = 20;
    video_options.frames_per_lambda = 12;
    video_options.final_hold_frames = 20;
    video_options.close_animation_figure = false;
end

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

    params.disc_stress      = 20;
    params.disc_pressure    = 60;
    params.pinf_sweep = linspace(0, 120e3, params.disc_pressure); % [Pa]
    params.gamma = 1.4;
    params.Minf = 4.0;
    params.T0 = 400; % [K], for aerodynamic damping nondimensionalization

    params.lambda   = params.gamma * params.pinf_sweep * params.Minf *...
        (a^3 / D);
    params.tol      = 1e-15; % dimensionless safety factor
    
    sf = 2; sigma_y = 450*10^6;
    params.sf_rel = sigma_y/sf;

    params.steady_frac  = 0.2;
end


function [w_center, w_i, q_history, lambda_F, amp_steady, first_unstable_idx, ...
    natural_frequencies_hz_array, damping_array, unstable, max_real_eig, ...
    vm_upper, vm_lower] = ...
    pressure_sweep(params, psi_w, psi_w_xx, psi_w_yy, psi_w_xy, struct_mat_B2, ...
    struct_mat_K, struct_mat_Aw_not_scaled, struct_mat_Minv, NModes_w, ...
    struct_mat_L2, struct_mat_Awdot_not_scaled)

    define_parallel_processing();

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

    progress_step = max(1, ceil(0.10 * n_pressures));
    completed = 0;
    dq = parallel.pool.DataQueue;
    afterEach(dq, @update_progress);
    fprintf('Pressure sweep progress: 0/%d (0%%)\n', n_pressures);

    % Preallocation
    w_center                     = zeros(n_pressures, Nt);
    w_i                          = zeros(n_pressures, nPts, Nt);
    q_history                    = zeros(n_pressures, Nt, NModes_w);
    vm_upper                     = zeros(n_pressures, nPts, Nt);
    vm_lower                     = zeros(n_pressures, nPts, Nt);

    natural_frequencies_hz_array = zeros(NModes_w, n_pressures);
    damping_array                = zeros(NModes_w, n_pressures);
    max_real_eig                 = zeros(1, n_pressures);
    unstable                     = false(1, n_pressures);
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

        % vectorized deflection at all points for this pressure
        w_i_local = modal2physical(Q, Psi_w).';

        % store full-field + center trace
        w_i(idx,:,:)    = reshape(w_i_local, [1, nPts, Nt]);
        q_history(idx,:,:) = reshape(Q, [1, Nt, NModes_w]);
        w_center_local  = w_i_local(i_center, :);
        w_center(idx,:) = w_center_local;

        amp_steady(idx) = estimate_window_amplitude...
            (t_eval, w_center_local, [1-steady_frac, 1.0]);

        % VM stresses on upper/lower surfaces at all points and all times
        [vmU_local, vmL_local] = von_mises( ...
            Q, Psi_xx, Psi_yy, Psi_xy, struct_mat_B2, h, nu, D);

        vm_upper(idx,:,:) = reshape(vmU_local, [1, nPts, Nt]);
        vm_lower(idx,:,:) = reshape(vmL_local, [1, nPts, Nt]);

        struct_mat_K_total = struct_mat_K + struct_mat_Aw;
        struct_mat_C       = struct_mat_Awdot;

        [natural_frequencies_hz_array(:, idx), damping_array(:, idx), ...
            max_real_eig(idx), omega_scale] = solve_coupled_eigensystem...
            (struct_mat_Minv, struct_mat_K_total, struct_mat_C, NModes_w);

        unstable(idx) = max_real_eig(idx) > (omega_scale * tol);
        send(dq, 1);
    end

    fprintf('Pressure sweep progress: %d/%d (100%%)\n', ...
        completed, n_pressures);

    first_unstable_idx = find(unstable, 1, 'first');
    if isempty(first_unstable_idx)
        lambda_F = nan;
    else
        lambda_F = lambda(first_unstable_idx);
    end


    function update_progress(~)
        completed = completed + 1;
        if mod(completed, progress_step) == 0 || completed == n_pressures
            pct = 100 * completed / n_pressures;
            fprintf('Pressure sweep progress: %d/%d (%.0f%%)\n', ...
                completed, n_pressures, pct);
        end
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

    opts = struct('tol', 1e-10, 'maxit', 500);
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

% estimate displacement amplitude from a selected [startFrac, endFrac] window.
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


function plot_output(params, plot_data, ~, xMesh, yMesh)

    style = apply_paper_plot_settings();

    lambda = plot_data.lambda;
    t_eval = plot_data.t_eval;
    a      = plot_data.a;
    b      = plot_data.b;
    h      = plot_data.h;
    w_center      = plot_data.w_center;
    A_steady      = plot_data.A_steady;
    damping_array = plot_data.damping_array;
    vm_max_steady    = plot_data.vm_max_steady;
    stress_cr_steady    = vm_max_steady / params.sf_rel;
    reduced_freq_array = plot_data.reduced_freq_array;

    %% Amp. w_center(t)/h for selected lambdas
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

    %% Lambda vs steady amp
    nexttile; hold on; grid off;

    plot(lambda, A_steady/h, '--o', ...
        'LineWidth', style.lineWidth, ...
        'MarkerSize', style.markerSize);

    set(gca,'FontSize',style.axesFontSize);
    xlim([0, max(lambda)]);
    xlabel('$\lambda$','Interpreter','latex','FontSize',style.labelFontSize);
    ylabel('$(w_{center}/h)_{amp}^{steady}$','Interpreter','latex','FontSize',style.labelFontSize);
    axis square

    %% Max steady VM vs lambda
    nexttile; hold on; grid off;

    is_upper = logical(plot_data.max_vm_is_upper_steady);
    is_lower = ~is_upper;

    h_base = plot(lambda, stress_cr_steady, '--', ...
        'LineWidth', style.lineWidth, ...
        'HandleVisibility', 'off');
    base_color = h_base.Color;

    if any(is_upper)
        plot(lambda(is_upper), stress_cr_steady(is_upper), '--o', ...
            'LineStyle', 'none', ...
            'MarkerSize', style.markerSize, ...
            'LineWidth', style.lineWidth, ...
            'MarkerEdgeColor', base_color, ...
            'DisplayName', 'Upper surface');
    end

    if any(is_lower)
        plot(lambda(is_lower), stress_cr_steady(is_lower), '--^', ...
            'LineStyle', 'none', ...
            'MarkerSize', style.markerSize, ...
            'LineWidth', style.lineWidth, ...
            'MarkerEdgeColor', base_color, ...
            'DisplayName', 'Lower surface');
    end

    y_target = 1;
    lambda_at_target = nan;
    exact_idx = find(stress_cr_steady == y_target, 1, 'first');
    if ~isempty(exact_idx)
        lambda_at_target = lambda(exact_idx);
    else
        y_shift = stress_cr_steady - y_target;
        cross_idx = find(y_shift(1:end-1).*y_shift(2:end) <= 0, 1, 'first');
        if ~isempty(cross_idx)
            lambda_at_target = interp1( ...
                stress_cr_steady(cross_idx:cross_idx+1), ...
                lambda(cross_idx:cross_idx+1), y_target, 'linear');
        end
    end

    if ~isnan(lambda_at_target)
        plot(lambda_at_target, y_target, 'o', ...
            'MarkerSize', style.markerSize*1.8, ...
            'MarkerFaceColor', 'none', ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', style.lineWidth, ...
            'DisplayName', 'Critical point');
    end

    set(gca,'FontSize',style.axesFontSize);
    xlim([0, max(lambda)]);
    ylim([0, max(stress_cr_steady)*1.05])
    xlabel('$\lambda$','Interpreter','latex','FontSize',style.labelFontSize);
    ylabel('$\eta_{f}$','Interpreter','latex','FontSize',style.labelFontSize);
    axis square
    legend('Location','best','FontSize',style.axesFontSize);

    sgtitle('Steady window (last 20% of time marching)', ...
        'FontSize', style.titleFontSize, 'FontWeight', 'normal');

    figure('Color', style.figureColor, 'Position', style.figurePosition);
    tiledlayout(1,3,'TileSpacing',style.tileSpacing,'Padding',style.tilePadding);
    
    %% Location of steady max VM on the panel
    nexttile; hold on; grid off;
    
    xN = plot_data.x_max_vm_steady./a;
    yN = plot_data.y_max_vm_steady./b;
    lambda_F = plot_data.lambda_F;
    
    is_pre_flutter = lambda < lambda_F;
    is_post_flutter = ~is_pre_flutter;
    
    legend_handles = gobjects(0);
    
    if any(is_pre_flutter)
        scatter(xN(is_pre_flutter), yN(is_pre_flutter), style.scatterSizeMedium, ...
            lambda(is_pre_flutter), 'o', 'filled', ...
            'HandleVisibility','off');
        legend_handles(end+1) = scatter(nan, nan, style.scatterSizeMedium, ...
            'o', 'filled', ...
            'MarkerFaceColor', [0.47 0.69 0.83], ...
            'MarkerEdgeColor', [0.47 0.69 0.83], ...
            'DisplayName', '$\lambda < \lambda_F$');
    end
    
    if any(is_post_flutter)
        scatter(xN(is_post_flutter), yN(is_post_flutter), style.scatterSizeMedium, ...
            lambda(is_post_flutter), '^', 'filled', ...
            'HandleVisibility','off');
        legend_handles(end+1) = scatter(nan, nan, style.scatterSizeMedium, ...
            '^', 'filled', ...
            'MarkerFaceColor', [0.47 0.69 0.83], ...
            'MarkerEdgeColor', [0.47 0.69 0.83], ...
            'DisplayName', '$\lambda \geq \lambda_F$');
    end
    
    finite_lambda = isfinite(lambda);
    if any(finite_lambda)
        lambda_for_max = lambda;
        lambda_for_max(~finite_lambda) = -inf;
        [~, idx_max_lambda] = max(lambda_for_max);
        h_critical = scatter(xN(idx_max_lambda), yN(idx_max_lambda), style.scatterSizeMedium*2.5, ...
            'o', 'MarkerFaceColor', 'none', 'MarkerEdgeColor', [0.47 0.69 0.83], ...
            'LineWidth', style.lineWidth, 'DisplayName', 'Critical location');
        legend_handles(end+1) = h_critical;
    end
    
    legend(legend_handles, 'Location','best', ...
        'Interpreter','latex','FontSize',style.axesFontSize*0.6);
    
    cb = colorbar; cb.Label.String = '$\lambda$';
    cb.Label.Interpreter = 'latex';
    cb.TickLabelInterpreter = 'latex';
    cb.Label.FontSize = style.labelFontSize*0.6;
    cb.FontSize = style.axesFontSize*0.6;
    
    set(gca,'FontSize',style.axesFontSize*0.6);
    xlabel('$x/a$','Interpreter','latex','FontSize',style.labelFontSize*0.6);
    ylabel('$y/b$','Interpreter','latex','FontSize',style.labelFontSize*0.6);
    
    xlim([0, 1]);
    ylim([-0.5, 0.5]);
    axis square
    
    %% Damping vs lambda
    nexttile; hold on; grid off;
    
    for j = 1:size(damping_array,1)
        scatter(lambda, damping_array(j,:), style.scatterSizeLarge, '.', ...
            'MarkerEdgeAlpha', 1);
    end
    
    set(gca,'FontSize',style.axesFontSize);
    xlabel('$\lambda$','Interpreter','latex','FontSize',style.labelFontSize);
    ylabel('$\zeta$','Interpreter','latex','FontSize',style.labelFontSize);
    axis square
    
    %% Reduced frequency vs lambda
    nexttile; hold on; grid off;
    
    for j = 1:size(reduced_freq_array,1)
        plot(lambda, reduced_freq_array(j,:), '-', ...
            'LineWidth', style.lineWidth);
    end
    
    set(gca,'FontSize',style.axesFontSize);
    xlabel('$\lambda$','Interpreter','latex','FontSize',style.labelFontSize*1.5);
    ylabel('$k$','Interpreter','latex','FontSize',style.labelFontSize*1.5);
    xlim([1000,1150]);  ylim([0.15,0.43]);
    axis square

    %% Transverse displacements at yield
    figure('Color', style.figureColor, 'Position', style.figurePosition);
    hold on; grid off;

    x_yield = linspace(0, a, params.disc_stress);
    y_yield = linspace(-b/2, b/2, params.disc_stress);
    [xYieldMesh, yYieldMesh] = meshgrid(x_yield, y_yield);
    yield_wh = plot_data.yield_deformation_wh;
    yield_min = min(yield_wh(:));
    yield_max = max(yield_wh(:));
    if yield_max <= yield_min
        delta = max(1, abs(yield_min)) * 1e-6;
        yield_min = yield_min - delta;
        yield_max = yield_max + delta;
    end
    contour_levels = linspace(yield_min, yield_max, 200);
    contourf(xYieldMesh./a, yYieldMesh./b, yield_wh, ...
        contour_levels, ...
        'LineStyle', 'none');

    set(gca,'FontSize',style.axesFontSize);
    xlabel('$x/a$','Interpreter','latex','FontSize',style.labelFontSize);
    ylabel('$y/b$','Interpreter','latex','FontSize',style.labelFontSize);
    clim([yield_min, yield_max]);

    cb_failure = colorbar;
    cb_failure.Label.String = '$w/h$';
    cb_failure.Label.Interpreter = 'latex';
    cb_failure.TickLabelInterpreter = 'latex';
    cb_failure.Label.FontSize = style.labelFontSize*2;
    cb_failure.FontSize = style.axesFontSize;
    
    %% First two mode shapes
    figure('Color', style.figureColor, 'Position', style.figurePosition);
    tiledlayout(1,2,'TileSpacing',style.tileSpacing,'Padding',style.tilePadding);

    mode_1 = plot_data.mode_shape_1;
    mode_2 = plot_data.mode_shape_2;

    nexttile; hold on; grid off;
    contourf(xMesh./a, yMesh./b, mode_1, 20, 'LineStyle', 'none');
    set(gca,'FontSize',style.axesFontSize);
    xlabel('$x/a$','Interpreter','latex','FontSize',style.labelFontSize);
    ylabel('$y/b$','Interpreter','latex','FontSize',style.labelFontSize);
    axis square
    cb_mode1 = colorbar;
    cb_mode1.Label.String = 'Mode shape';
    cb_mode1.Label.Interpreter = 'none';
    cb_mode1.TickLabelInterpreter = 'latex';
    cb_mode1.Label.FontSize = style.labelFontSize*2;
    cb_mode1.FontSize = style.axesFontSize;
    clim([-1,1]);
    
    nexttile; hold on; grid off;
    contourf(xMesh./a, yMesh./b, mode_2, 20, 'LineStyle', 'none');
    set(gca,'FontSize',style.axesFontSize);
    xlabel('$x/a$','Interpreter','latex','FontSize',style.labelFontSize);
    ylabel('$y/b$','Interpreter','latex','FontSize',style.labelFontSize);
    axis square
    cb_mode2 = colorbar;
    cb_mode2.Label.String = 'Mode shape';
    cb_mode2.Label.Interpreter = 'none';
    cb_mode2.TickLabelInterpreter = 'latex';
    cb_mode2.Label.FontSize = style.labelFontSize*2;
    cb_mode2.FontSize = style.axesFontSize;
    clim([-1,1]);
end


function export_plate_response_video( ...
    params, plot_data, q_history, psi_w, psi_w_xx, psi_w_yy, psi_w_xy, ...
    struct_mat_B2, xMesh, yMesh, video_options)

    if isempty(q_history)
        error(['Cannot export the plate animation without q_history. ' ...
            'Regenerate the cache so the modal histories are saved.']);
    end

    style = apply_paper_plot_settings();

    lambda = plot_data.lambda(:);
    lambda_F = plot_data.lambda_F;
    if ~isscalar(lambda_F)
        lambda_F = lambda_F(1);
    end

    t_eval = params.t_eval(:).';
    amp_norm = plot_data.A_steady(:) ./ params.h;
    eta_f_steady = plot_data.vm_max_steady(:) ./ params.sf_rel;
    damping_array = real(plot_data.damping_array);
    w_center_norm = plot_data.w_center ./ params.h;
    time_response_idx = select_time_response_indices(t_eval, 1.0);

    [eta_panel_time, yield_lambda_idx, yield_time_idx, failure_reached] = ...
        compute_panel_failure_history( ...
        params, q_history, psi_w_xx, psi_w_yy, psi_w_xy, struct_mat_B2);

    [frame_lambda_idx, frame_time_idx] = build_animation_frame_schedule(eta_panel_time);
    if isempty(frame_lambda_idx)
        warning('Skipping MP4 export because no animation frames were selected.');
        return;
    end

    x_points_plot = reshape(xMesh, 1, []);
    y_points_plot = reshape(yMesh, 1, []);
    Psi_plot = build_shape_matrix(psi_w, x_points_plot, y_points_plot);

    [contour_levels, plate_color_limits] = frame_contour_levels(zeros(size(xMesh)));

    amp_ylim = [0, max(amp_norm) * 1.08];
    if amp_ylim(2) <= amp_ylim(1)
        amp_ylim(2) = amp_ylim(1) + 1e-6;
    end

    eta_ylim = [0, max(max(eta_f_steady) * 1.08, 1.05)];

    damping_finite = damping_array(isfinite(damping_array));
    if isempty(damping_finite)
        damping_limits = [-1e-3, 1e-3];
    else
        damping_span = max(max(damping_finite) - min(damping_finite), 1e-3);
        damping_limits = [ ...
            min(min(damping_finite), 0) - 0.08*damping_span, ...
            max(max(damping_finite), 0) + 0.08*damping_span];
    end

    fig = figure( ...
        'Color', style.figureColor, ...
        'Position', [80, 60, 1900, 1350]);
    fig.CloseRequestFcn = @(~, ~) fprintf( ...
        'Animation export is running; the figure can be closed after export completes.\n');
    tl = tiledlayout(fig, 3, 2, ...
        'TileSpacing', style.tileSpacing, ...
        'Padding', style.tilePadding);

    ax_amp = nexttile(tl, 1);
    hold(ax_amp, 'on');
    grid(ax_amp, 'off');
    h_amp_line = plot(ax_amp, nan, nan, '-o', ...
        'LineWidth', style.highlightLineWidth, ...
        'MarkerSize', style.markerSize, ...
        'Color', [0.00, 0.45, 0.74]);
    h_amp_marker = plot(ax_amp, nan, nan, 'o', ...
        'MarkerSize', style.markerSize*1.6, ...
        'LineWidth', style.highlightLineWidth, ...
        'MarkerEdgeColor', 'k', ...
        'MarkerFaceColor', [0.00, 0.45, 0.74]);
    if isfinite(lambda_F)
        h_amp_flutter = xline(ax_amp, lambda_F, '--', '$\lambda_F$', ...
            'LineWidth', style.lineWidth, ...
            'Color', [0.85, 0.33, 0.10], ...
            'Interpreter', 'latex');
        h_amp_flutter.Visible = 'off';
    else
        h_amp_flutter = gobjects(0);
    end
    xlim(ax_amp, [0, max(lambda)]);
    ylim(ax_amp, amp_ylim);
    xlabel(ax_amp, '$\lambda$', 'Interpreter', 'latex', 'FontSize', style.labelFontSize);
    ylabel(ax_amp, '$(w_{center}/h)_{amp}^{steady}$', ...
        'Interpreter', 'latex', 'FontSize', style.labelFontSize);
    title(ax_amp, 'Normalized steady amplitude', ...
        'FontSize', style.titleFontSize, 'Interpreter', 'latex');

    ax_fail = nexttile(tl, 3);
    hold(ax_fail, 'on');
    grid(ax_fail, 'off');
    h_fail_line = plot(ax_fail, nan, nan, '-o', ...
        'LineWidth', style.highlightLineWidth, ...
        'MarkerSize', style.markerSize, ...
        'Color', [0.47, 0.67, 0.19]);
    h_fail_marker = plot(ax_fail, nan, nan, 'o', ...
        'MarkerSize', style.markerSize*1.6, ...
        'LineWidth', style.highlightLineWidth, ...
        'MarkerEdgeColor', 'k', ...
        'MarkerFaceColor', [0.47, 0.67, 0.19]);
    yline(ax_fail, 1, '--k', '$\eta_f = 1$', ...
        'LineWidth', style.lineWidth, 'FontSize', style.axesFontSize, ...
        'Interpreter', 'latex');
    if isfinite(lambda_F)
        h_fail_flutter = xline(ax_fail, lambda_F, '--', '$\lambda_F$', ...
            'LineWidth', style.lineWidth, ...
            'Color', [0.85, 0.33, 0.10], ...
            'Interpreter', 'latex');
        h_fail_flutter.Visible = 'off';
    else
        h_fail_flutter = gobjects(0);
    end
    xlim(ax_fail, [0, max(lambda)]);
    ylim(ax_fail, eta_ylim);
    xlabel(ax_fail, '$\lambda$', 'Interpreter', 'latex', 'FontSize', style.labelFontSize);
    ylabel(ax_fail, '$\eta_f$', 'Interpreter', 'latex', 'FontSize', style.labelFontSize);
    title(ax_fail, 'Failure criterion', ...
        'FontSize', style.titleFontSize, 'Interpreter', 'latex');

    ax_stability = nexttile(tl, 2);
    hold(ax_stability, 'on');
    grid(ax_stability, 'off');
    n_modes = size(damping_array, 1);
    h_damping = gobjects(n_modes, 1);
    for mode_idx = 1:n_modes
        h_damping(mode_idx) = scatter(ax_stability, nan, nan, ...
            style.scatterSizeMedium*0.45, ...
            'o', 'filled', ...
            'MarkerFaceColor', [0.12, 0.47, 0.71], ...
            'MarkerEdgeColor', 'none', ...
            'MarkerFaceAlpha', 0.28, ...
            'MarkerEdgeAlpha', 0.28);
    end
    yline(ax_stability, 0, '--k', '$\zeta = 0$', ...
        'LineWidth', style.lineWidth, 'FontSize', style.axesFontSize, ...
        'Interpreter', 'latex');
    h_stability_current = scatter(ax_stability, nan, nan, style.scatterSizeMedium*0.65, ...
        'filled', 'MarkerFaceColor', [0.85, 0.33, 0.10], ...
        'MarkerEdgeColor', 'k');
    h_stability_lambda = xline(ax_stability, lambda(1), ':', 'Current $\lambda$', ...
        'LineWidth', style.lineWidth, 'Color', [0.25, 0.25, 0.25], ...
        'Interpreter', 'latex');
    if isfinite(lambda_F)
        h_stability_flutter = xline(ax_stability, lambda_F, '--', '$\lambda_F$', ...
            'LineWidth', style.lineWidth, ...
            'Color', [0.85, 0.33, 0.10], ...
            'Interpreter', 'latex');
        h_stability_flutter.Visible = 'off';
    else
        h_stability_flutter = gobjects(0);
    end
    xlim(ax_stability, [0, max(lambda)]);
    ylim(ax_stability, damping_limits);
    xlabel(ax_stability, '$\lambda$', 'Interpreter', 'latex', 'FontSize', style.labelFontSize);
    ylabel(ax_stability, '$\zeta$', 'Interpreter', 'latex', 'FontSize', style.labelFontSize);
    title(ax_stability, 'Linear stability analysis', ...
        'FontSize', style.titleFontSize, 'Interpreter', 'latex');

    ax_response = nexttile(tl, 4);
    [h_response, h_response_marker] = initialize_time_response_panel( ...
        ax_response, t_eval(time_response_idx), style, [0.30, 0.45, 0.80]);

    ax_plate = nexttile(tl, 6);
    hold(ax_plate, 'on');
    grid(ax_plate, 'off');
    h_contour = contourf(ax_plate, xMesh./params.a, yMesh./params.b, ...
        field_for_contour(zeros(size(xMesh)), contour_levels), ...
        contour_levels, 'LineStyle', 'none');
    colormap(ax_plate, parula(256));
    clim(ax_plate, plate_color_limits);
    axis(ax_plate, 'square');
    xlim(ax_plate, [0, 1]);
    ylim(ax_plate, [-0.5, 0.5]);
    xlabel(ax_plate, '$x/a$', 'Interpreter', 'latex', 'FontSize', style.labelFontSize);
    ylabel(ax_plate, '$y/b$', 'Interpreter', 'latex', 'FontSize', style.labelFontSize);
    cb_plate = colorbar(ax_plate);
    cb_plate.Label.String = '$w/h$';
    cb_plate.Label.Interpreter = 'latex';
    cb_plate.TickLabelInterpreter = 'latex';
    cb_plate.Label.FontSize = style.labelFontSize;
    cb_plate.FontSize = style.axesFontSize;

    if failure_reached
        video_title = sprintf( ...
            'Animation stops at first yield: $\\lambda = %.1f$, $t = %.2f$ s', ...
            lambda(yield_lambda_idx), t_eval(yield_time_idx));
    else
        video_title = sprintf( ...
            'No yield reached in the scanned range ($\\lambda_{max} = %.1f$)', ...
            lambda(end));
    end
    sgtitle(tl, video_title, 'FontSize', style.titleFontSize, 'Interpreter', 'latex');

    writer = VideoWriter(video_options.mp4_file, 'MPEG-4');
    writer.FrameRate = video_options.fps;
    open(writer);

    n_frames = numel(frame_lambda_idx);
    progress_step = max(1, ceil(0.10 * n_frames));
    last_frame = [];
    fprintf('MP4 export progress: 0/%d (0%%)\n', n_frames);

    for frame_idx = 1:n_frames
        lambda_idx = frame_lambda_idx(frame_idx);
        time_idx = frame_time_idx(frame_idx);
        lambda_now = lambda(lambda_idx);

        set(h_amp_line, 'XData', lambda(1:lambda_idx), 'YData', amp_norm(1:lambda_idx));
        set(h_amp_marker, 'XData', lambda_now, 'YData', amp_norm(lambda_idx));

        set(h_fail_line, 'XData', lambda(1:lambda_idx), 'YData', eta_f_steady(1:lambda_idx));
        set(h_fail_marker, 'XData', lambda_now, 'YData', eta_f_steady(lambda_idx));

        for mode_idx = 1:n_modes
            set(h_damping(mode_idx), ...
                'XData', lambda(1:lambda_idx), ...
                'YData', damping_array(mode_idx, 1:lambda_idx));
        end
        set(h_stability_current, ...
            'XData', lambda_now * ones(n_modes, 1), ...
            'YData', damping_array(:, lambda_idx));
        h_stability_lambda.Value = lambda_now;

        set_flutter_marker_visibility(h_amp_flutter, lambda_now, lambda_F);
        set_flutter_marker_visibility(h_fail_flutter, lambda_now, lambda_F);
        set_flutter_marker_visibility(h_stability_flutter, lambda_now, lambda_F);

        w_response_now = w_center_norm(lambda_idx, time_response_idx);
        update_time_response_panel( ...
            ax_response, h_response, h_response_marker, ...
            t_eval(time_response_idx), w_response_now, lambda_now, style);

        q_frame = reshape(q_history(lambda_idx, time_idx, :), 1, []);
        wh_frame = reshape(modal2physical(q_frame, Psi_plot), size(xMesh));
        wh_frame = real(wh_frame) ./ params.h;
        [contour_levels, plate_color_limits] = frame_contour_levels(wh_frame);

        if isgraphics(h_contour)
            delete(h_contour);
        end
        h_contour = contourf(ax_plate, xMesh./params.a, yMesh./params.b, ...
            field_for_contour(wh_frame, contour_levels), ...
            contour_levels, 'LineStyle', 'none');
        clim(ax_plate, plate_color_limits);
        title(ax_plate,...
            "Physical Response", 'FontSize', style.titleFontSize);

        drawnow;
        last_frame = capture_figure_frame(fig);
        writeVideo(writer, last_frame);

        if mod(frame_idx, progress_step) == 0 || frame_idx == n_frames
            fprintf('MP4 export progress: %d/%d (%.0f%%)\n', ...
                frame_idx, n_frames, 100*frame_idx/n_frames);
        end
    end

    for hold_idx = 1:video_options.final_hold_frames
        writeVideo(writer, last_frame);
    end

    close(writer);
    if isgraphics(fig)
        fig.CloseRequestFcn = 'closereq';
    end
    if video_options.close_animation_figure
        close(fig);
    end

    if failure_reached
        fprintf(['Saved MP4 animation to %s (stopped at yield: lambda = %.3f, ' ...
            't = %.3f s)\n'], video_options.mp4_file, ...
            lambda(yield_lambda_idx), t_eval(yield_time_idx));
    else
        fprintf('Saved MP4 animation to %s (no yield reached).\n', ...
            video_options.mp4_file);
    end
end


function idx_time = select_time_response_indices(t_eval, max_time_seconds)
    t0 = t_eval(1);
    idx_time = find(t_eval <= t0 + max_time_seconds);
    if isempty(idx_time)
        idx_time = 1;
    end
end


function response_ylim = compute_time_response_ylim(w_response)
    response_min = min(w_response(:));
    response_max = max(w_response(:));

    if response_max <= response_min
        padding = max(1e-6, abs(response_min) * 0.08);
    else
        padding = 0.08 * (response_max - response_min);
    end

    response_ylim = [response_min - padding, response_max + padding];
end


function [h_response, h_marker] = initialize_time_response_panel(ax, t_eval, style, line_color)
    hold(ax, 'on');
    grid(ax, 'off');
    h_response = plot(ax, nan, nan, '-', ...
        'LineWidth', style.highlightLineWidth, ...
        'Color', line_color);
    h_marker = plot(ax, nan, nan, 'o', ...
        'MarkerSize', style.markerSize*1.4, ...
        'MarkerFaceColor', line_color, ...
        'MarkerEdgeColor', 'k', ...
        'LineWidth', style.lineWidth);
    yline(ax, 0, ':', 'Color', [0.40, 0.40, 0.40], ...
        'LineWidth', max(1.0, 0.5*style.lineWidth), ...
        'HandleVisibility', 'off', ...
        'Interpreter', 'latex');
    xlim(ax, [t_eval(1), t_eval(end)]);
    xlabel(ax, '$t$ [sec]', 'Interpreter', 'latex', 'FontSize', style.labelFontSize);
    ylabel(ax, '$w_{center}/h$', 'Interpreter', 'latex', 'FontSize', style.labelFontSize);
    title(ax, 'Time response', 'Interpreter', 'latex', 'FontSize', style.titleFontSize);
end


function update_time_response_panel( ...
    ax, h_response, h_marker, t_eval, w_center_norm, lambda_value, style)

    response_ylim = compute_time_response_ylim(w_center_norm);
    set(h_response, 'XData', t_eval, 'YData', w_center_norm);
    set(h_marker, 'XData', t_eval(end), 'YData', w_center_norm(end));
    ylim(ax, response_ylim);
    title(ax, sprintf('Time response ($\\lambda = %.1f$)', lambda_value), ...
        'Interpreter', 'latex', 'FontSize', style.titleFontSize);
end


function Z = field_for_contour(Z, contour_levels)
    if max(Z(:)) > min(Z(:))
        return;
    end

    contour_span = max(contour_levels) - min(contour_levels);
    perturbation = max(contour_span * 1e-9, 1e-12);
    Z(1) = Z(1) - perturbation;
    Z(end) = Z(end) + perturbation;
end


function [eta_panel_time, yield_lambda_idx, yield_time_idx, failure_reached] = ...
    compute_panel_failure_history(params, q_history, psi_w_xx, psi_w_yy, psi_w_xy, struct_mat_B2)

    x_lin = linspace(0, params.a, params.disc_stress);
    y_lin = linspace(-params.b/2, params.b/2, params.disc_stress);
    [Xg, Yg] = meshgrid(x_lin, y_lin);
    x_points = reshape(Xg, 1, []);
    y_points = reshape(Yg, 1, []);

    Psi_xx = build_shape_matrix(psi_w_xx, x_points, y_points);
    Psi_yy = build_shape_matrix(psi_w_yy, x_points, y_points);
    Psi_xy = build_shape_matrix(psi_w_xy, x_points, y_points);

    n_pressures = size(q_history, 1);
    Nt = size(q_history, 2);
    n_modes = size(q_history, 3);

    eta_panel_time = zeros(n_pressures, Nt);
    for idx = 1:n_pressures
        Q = reshape(q_history(idx, :, :), [Nt, n_modes]);
        [vm_upper, vm_lower] = von_mises( ...
            Q, Psi_xx, Psi_yy, Psi_xy, struct_mat_B2, ...
            params.h, params.nu, params.D);

        vm_panel_time = max(max(vm_upper, vm_lower), [], 1);
        eta_panel_time(idx, :) = reshape(vm_panel_time ./ params.sf_rel, 1, []);
    end

    yield_lambda_idx = find(any(eta_panel_time >= 1, 2), 1, 'first');
    failure_reached = ~isempty(yield_lambda_idx);

    if failure_reached
        yield_time_idx = find(eta_panel_time(yield_lambda_idx, :) >= 1, 1, 'first');
    else
        [~, idx_linear] = max(eta_panel_time(:));
        [yield_lambda_idx, yield_time_idx] = ind2sub(size(eta_panel_time), idx_linear);
    end
end


function [frame_lambda_idx, frame_time_idx] = build_animation_frame_schedule(eta_panel_time)

    n_pressures = size(eta_panel_time, 1);
    Nt = size(eta_panel_time, 2);
    frame_lambda_idx = zeros(n_pressures, 1);
    frame_time_idx = zeros(n_pressures, 1);

    for idx = 1:n_pressures
        yield_time_idx = find(eta_panel_time(idx, :) >= 1, 1, 'first');
        frame_lambda_idx(idx) = idx;
        if isempty(yield_time_idx)
            frame_time_idx(idx) = Nt;
        else
            frame_time_idx(idx) = yield_time_idx;
            break;
        end
    end

    last_frame_idx = find(frame_lambda_idx > 0, 1, 'last');
    frame_lambda_idx = frame_lambda_idx(1:last_frame_idx);
    frame_time_idx = frame_time_idx(1:last_frame_idx);
end


function [contour_levels, color_limits] = frame_contour_levels(wh_frame)
    frame_abs_max = max(abs(wh_frame(:)));
    frame_abs_max = max(frame_abs_max, 1e-6);
    color_limits = [-frame_abs_max, frame_abs_max];
    contour_levels = linspace(color_limits(1), color_limits(2), 61);
end


function set_flutter_marker_visibility(h_marker, lambda_now, lambda_F)
    if isempty(h_marker) || ~isgraphics(h_marker)
        return;
    end

    if isfinite(lambda_F) && lambda_now >= lambda_F
        h_marker.Visible = 'on';
    else
        h_marker.Visible = 'off';
    end
end


function frame = capture_figure_frame(fig)
    if ~isgraphics(fig)
        error('Animation figure was closed before MP4 export completed.');
    end

    drawnow;
    try
        frame = getframe(fig);
    catch first_error
        pause(0.05);
        drawnow;
        try
            frame = getframe(fig);
        catch
            rethrow(first_error);
        end
    end
end


function [mode_1, mode_2] = compute_lowest_frequency_modes( ...
    struct_mat_Minv, struct_mat_K, psi_w, xMesh, yMesh)

    A_dry = struct_mat_Minv * struct_mat_K;
    [V_dry, D_dry] = eig(A_dry);
    omega_sq = real(diag(D_dry));

    valid_idx = find(isfinite(omega_sq) & (omega_sq > 0));

    [~, rel_order] = sort(omega_sq(valid_idx), 'ascend');
    idx_mode1 = valid_idx(rel_order(1));
    idx_mode2 = valid_idx(rel_order(2));

    q1 = V_dry(:, idx_mode1);
    [~, i_ref_1] = max(abs(q1));
    q1 = q1 * exp(-1i*angle(q1(i_ref_1)));

    q2 = V_dry(:, idx_mode2);
    [~, i_ref_2] = max(abs(q2));
    q2 = q2 * exp(-1i*angle(q2(i_ref_2)));

    mode_1 = zeros(size(xMesh));
    mode_2 = zeros(size(xMesh));
    for n = 1:numel(psi_w)
        mode_1 = mode_1 + q1(n) * psi_w{n}(xMesh, yMesh);
        mode_2 = mode_2 + q2(n) * psi_w{n}(xMesh, yMesh);
    end

    mode_1 = real(mode_1);
    mode_2 = real(mode_2);

    max_abs_1 = max(abs(mode_1(:)));
    max_abs_2 = max(abs(mode_2(:)));

    mode_1 = mode_1 / max_abs_1;
    mode_2 = mode_2 / max_abs_2;
end

function define_parallel_processing()
    p = gcp('nocreate');
    if isempty(p)
        parpool('IdleTimeout', Inf);
    end
end
