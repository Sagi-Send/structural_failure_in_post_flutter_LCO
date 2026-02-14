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
[w_max, lambda_F, natural_frequencies_hz_array, damping_array, unstable, max_real_eig] = pressure_sweep( ...
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
    params.T_max_nonlinear_solution, ...
    params.q_qdot_ics, ...
    struct_mat_L2, ...
    params.x_points, ...
    params.y_points);

if ~isnan(lambda_F)
    fprintf('Flutter onset at lambda = %.3g\n', lambda_F);
else
    fprintf('No flutter detected in the scanned range.\n');
end

reduced_freq_array = nondimentionalize( ...
    params.gamma, params.T0, params.Minf, a, natural_frequencies_hz_array);

plot_output(w_max, params.pinf_sweep, params.lambda, reduced_freq_array, damping_array);


function params = build_analysis_params(NModes_w, xMesh, yMesh, a, D)
    params.T_max_nonlinear_solution = 10;
    q0 = zeros(NModes_w,1);
    q0(1) = 1e-6;                       % tiny displacement perturbation
    params.q_qdot_ics = [q0; zeros(NModes_w,1)];
    
    params.x_points = reshape(xMesh, 1, []);
    params.y_points = reshape(yMesh, 1, []);

    params.disc = 10;
    params.pinf_sweep = linspace(0, 75e3, params.disc); % [Pa]
    params.gamma = 1.4;
    params.Minf = 4.0;
    params.T0 = 400; % [K], for aerodynamic damping nondimensionalization

    params.lambda = params.gamma * params.pinf_sweep * params.Minf * (a^3 / D);
    params.tol = 1e-15; % dimensionless safety factor
end


function [w_max, lambda_F, natural_frequencies_hz_array, damping_array, unstable, max_real_eig] = pressure_sweep( ...
    psi_w, pinf_sweep, lambda, gamma, Minf, struct_mat_K, struct_mat_Aw_not_scaled, ...
    struct_mat_Minv, NModes_w, tol, T_max_nonlinear_solution, q_qdot_ics, ...
    struct_mat_L2, x_point, y_point)

    n_pressures = numel(pinf_sweep);

    % Preallocation
    w_max = zeros(1, n_pressures);
    natural_frequencies_hz_array = zeros(NModes_w, n_pressures);
    damping_array = zeros(NModes_w, n_pressures);
    max_real_eig = zeros(1, n_pressures);
    unstable = false(1, n_pressures);

    parfor idx = 1:n_pressures
        pinf_i = pinf_sweep(idx);

        struct_mat_Aw = aerodynamic_stiffness(struct_mat_Aw_not_scaled, pinf_i, gamma, Minf);

        rhs_local = @(t, y) rhs_func_aero( ...
            t, y, NModes_w, struct_mat_Minv, struct_mat_K, struct_mat_L2, struct_mat_Aw);

        [~, w_modal] = ode45(rhs_local, [0, T_max_nonlinear_solution], q_qdot_ics);

        w_phys_all = modal2physical(w_modal(end, 1:NModes_w), x_point, y_point, psi_w);
        w_max(idx) = max(abs(w_phys_all), [], 'all');

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


function plot_output(w_max, p_sweep, lambda, reduced_freq_array, damping_array)
    figure();
    hold on;
    grid off;
    plot(p_sweep, w_max, '-x', 'LineWidth', 1.5);  
    set(gca, 'FontSize', 18);
    % xlim([0, max(p_sweep)]);
    % ylim([w_max(2), max(w_max)]);
    ylabel('$w_{max}$', 'Interpreter', 'latex', 'FontSize', 50);
    xlabel('$p_{\infty}$', 'Interpreter', 'latex', 'FontSize', 50);
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
