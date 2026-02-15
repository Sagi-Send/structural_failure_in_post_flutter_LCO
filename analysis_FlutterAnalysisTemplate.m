clc;
clear;
close all;

define_parallel_processing();

%% Build structural model
% Geometry and material parameters are defined inside this script.
create_AiryStressPlateModel

%% Analysis setup
params = build_analysis_params(NModes_w, xMesh, yMesh, a, D);

%% Pressure sweep
[w_center, lambda_F, flutter_onset_idx, natural_frequencies_hz_array, damping_array, unstable, max_real_eig] = pressure_sweep( ...
    psi_w, ...
    params.pinf_sweep, ...
    params.lambda, ...
    params.gamma, ...
    params.Minf, ...
    struct_mat_K, ...
    struct_mat_Aw_not_scaled, ...
    struct_mat_Minv, ...
    NModes_w, ...
    params.tol, ...
    params.t_eval, ...
    params.q_qdot_ics, ...
    struct_mat_L2, a, struct_mat_Q_not_scaled);

if ~isnan(lambda_F)
    fprintf('Flutter onset at lambda = %.3g\n', lambda_F);
else
    fprintf('No flutter detected in the scanned range.\n');
end

reduced_freq_array = nondimentionalize( ...
    params.gamma, params.T0, params.Minf, a, natural_frequencies_hz_array);

plot_output(w_center, params.lambda, lambda_F, flutter_onset_idx, h, reduced_freq_array, damping_array, params.t_eval);


function params = build_analysis_params(NModes_w, xMesh, yMesh, a, D)
    params.T_max_nonlinear_solution = 0.5;
    params.Nt = 500;
    params.t_eval = linspace(0, params.T_max_nonlinear_solution, params.Nt);
    q0 = zeros(NModes_w,1);
    q0(1) = 1e-6;                       % tiny displacement perturbation
    params.q_qdot_ics = zeros(2 * NModes_w,1);
    
    params.x_points = reshape(xMesh, 1, []);
    params.y_points = reshape(yMesh, 1, []);

    params.disc = 200;
    params.pinf_sweep = linspace(0, 75e3, params.disc); % [Pa]
    params.gamma = 1.4;
    params.Minf = 4.0;
    params.T0 = 400; % [K], for aerodynamic damping nondimensionalization

    params.lambda = params.gamma * params.pinf_sweep * params.Minf * (a^3 / D);
    params.tol = 1e-15; % dimensionless safety factor
end


function [w_center, lambda_F, first_unstable_idx, natural_frequencies_hz_array, damping_array, unstable, max_real_eig] = pressure_sweep( ...
    psi_w, pinf_sweep, lambda, gamma, Minf, struct_mat_K, struct_mat_Aw_not_scaled, ...
    struct_mat_Minv, NModes_w, tol, t_eval, q_qdot_ics, struct_mat_L2, a, struct_mat_Q_not_scaled)

    n_pressures = numel(pinf_sweep);

    % Preallocation
    Nt = numel(t_eval);
    w_center = zeros(n_pressures, Nt);
    natural_frequencies_hz_array = zeros(NModes_w, n_pressures);
    damping_array = zeros(NModes_w, n_pressures);
    max_real_eig = zeros(1, n_pressures);
    unstable = false(1, n_pressures);
    parfor idx = 1:n_pressures
        pinf_i = pinf_sweep(idx);

        struct_mat_Aw = aerodynamic_stiffness(struct_mat_Aw_not_scaled, pinf_i, gamma, Minf);

        deltaP_single_case = 1 ; % [kPa]
        struct_mat_Q = deltaP_single_case * struct_mat_Q_not_scaled ;
        rhs_local = @(t, y) rhs_func_aero( ...
            t, y, NModes_w, struct_mat_Minv, struct_mat_K, struct_mat_L2, struct_mat_Aw, struct_mat_Q);

        [~, w_modal] = ode45(rhs_local, t_eval, q_qdot_ics);

        % maximum deflection in time for each pressue value
        x_c = a/2;
        y_c = 0;
        
        w_center_local = zeros(1, Nt);
        
        for it = 1:Nt
            w_center_local(it) = modal2physical( ...
                w_modal(it, 1:NModes_w), x_c, y_c, psi_w);
        end
        w_center(idx,:) = w_center_local;

        struct_mat_K_total = struct_mat_K + struct_mat_Aw;

        [natural_frequencies_hz_array(:, idx), damping_array(:, idx), max_real_eig(idx), omega_scale] = ...
            solve_coupled_eigensystem(struct_mat_Minv, struct_mat_K_total, NModes_w);

        unstable(idx) = max_real_eig(idx) > (omega_scale * tol);
    end

    first_unstable_idx = find(unstable, 1, 'first');
    if isempty(first_unstable_idx)
        lambda_F = nan;
    else
        lambda_F = lambda(first_unstable_idx);
    end
end


function struct_mat_Aw = aerodynamic_stiffness(struct_mat_Aw_not_scaled, pinf_i, gamma, Minf)
    coeff_Aw = gamma * pinf_i * Minf;
    struct_mat_Aw = coeff_Aw * struct_mat_Aw_not_scaled;
end

function [natural_frequencies_hz, damping, max_real_eig, omega_scale] =...
    solve_coupled_eigensystem(struct_mat_Minv, struct_mat_K_total, NModes_w)

    A = [zeros(NModes_w), eye(NModes_w); ...
         -struct_mat_Minv * struct_mat_K_total, zeros(NModes_w)];

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


function plot_output(w_center, lambda, lambda_F, flutter_onset_idx, h, reduced_freq_array, damping_array, t_eval)
    figure;
    tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
    
    % ---------------- w_center(t)/h for selected lambdas ----------------
    nexttile; hold on; grid off;
    
    n_show   = min(4, numel(lambda));                 % show up to 4 curves
    idx_show = unique(round(linspace(1, numel(lambda), n_show)));
    
    for k = 1:numel(idx_show)
        i = idx_show(k);
        % if lambda(i) < lambda_F
            plot(t_eval, w_center(i,:)/h, 'LineWidth', 1.5, ...
                'DisplayName', sprintf('$\\lambda = %.1f$', lambda(i)));
        % else
        %     plot(t_eval, w_center(flutter_onset_idx,:)/h, 'LineWidth', 1.5, ...
        %         'DisplayName', sprintf('$\\lambda_F = %.1f$', lambda(flutter_onset_idx)));
        %     break;
        % end
    end
    
    set(gca,'FontSize',18);
    xlabel('$t$','Interpreter','latex','FontSize',24);
    ylabel('$w_{center}(t)/h$','Interpreter','latex','FontSize',24);
    legend('show','Interpreter','latex','Location','best');
    xlim([0, t_eval(end)]);
    
    % ---------------- damping vs lambda ----------------
    nexttile; hold on; grid off;
    
    scatter(lambda, damping_array, 300, '.', 'MarkerEdgeAlpha', 1);
    set(gca,'FontSize',18);
    xlim([0, 1300]);
    xlabel('$\lambda$','Interpreter','latex','FontSize',24);
    ylabel('$\zeta$','Interpreter','latex','FontSize',24);

    % ---------------- deflection vs lambda ----------------
    % nexttile; hold on;  grid off;
    % plot(lambda, abs(w_center/h), '-x', 'LineWidth', 1.5);  
    % set(gca, 'FontSize', 18);
    % % xlim([0, max(p_sweep)]);
    % % ylim([w_max(2), max(w_max)]);
    % ylabel('$w_{center}/h$', 'Interpreter', 'latex', 'FontSize', 50);
    % xlabel('$\lambda$', 'Interpreter', 'latex', 'FontSize', 50);
end


function reduced_freq_array = nondimentionalize(gamma, T0, Minf, a, natural_frequencies_hz_array)
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
