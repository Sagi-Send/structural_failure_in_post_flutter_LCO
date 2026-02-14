clc;clear all;close all;

% Parallel pool
define_parallel_processing();

%% Calculate the structural tensors
% we first create the structural model - calculate the structural tensors
% geometry and material parameters are hardcoded inside of that script
create_LinearPlateModel

% analysis parameters 
% - aerodynamics
disc = 200;
pinf_sweep  = linspace(0,75e3,disc); % [Pa], freestream static pressure values
gamma       = 1.4;
Minf        = 4.0 ; % freestream Mach number
T0          = 300 ; % [K] - flow temperature is only important for the aerodynamic damping term

% get the nondimensional parameter
lambda = gamma * pinf_sweep * Minf * (a^3 / D);  % this is equal to rho_inf * U_inf^2 * (a^3/D)

% physical solution visualization
x_point = reshape(xMesh,1,[]);  y_point = reshape(yMesh,1,[]);

tol = 1e-15; % dimensionless safety factor

%% Sweep pressure
[xF, natural_frequencies_hz_array, damping_array, unstable, max_real_eig] = flutter_response_for_location( ...
    psi_w, pinf_sweep, lambda, Minf, ...
    struct_mat_K, struct_mat_Aw_not_scaled, struct_mat_Awdot_not_scaled, struct_mat_Minv, NModes_w, tol);

if ~isnan(xF)
    fprintf('Flutter onset at lambda = %.3g\n', xF);
else
    fprintf('No flutter detected in the scanned range.\n');
end


reduced_freq_array = nondimentionalize...
    (gamma, T0, Minf, a, natural_frequencies_hz_array);
plot_output(lambda,...
    reduced_freq_array, damping_array);


function [lambda_F, natural_frequencies_hz_array,...
    damping_array, unstable, max_real_eig] = ...
    flutter_response_for_location...
    (psi_w, pinf_sweep, lambda, Minf, ...
    struct_mat_K, struct_mat_Aw_not_scaled, struct_mat_Awdot_not_scaled,...
    struct_mat_Minv, NModes_w, tol)

    disc = length(pinf_sweep);

    % Preallocations
    natural_frequencies_hz_array    = zeros(NModes_w, disc);
    damping_array                   = zeros(NModes_w, disc);

    % Stability indicator (max real part of eigenvalues)
    max_real_eig = zeros(1, disc);
    unstable     = false(1, disc);

    %% Loop over Pinf values
    for idx = 1:disc
        pinf_i = pinf_sweep(idx);

        % update aerodynamic stiffness and damping
        coeff_Aw    = 1.4 * pinf_i * Minf ;
        % aerodynamic damping - only if you choose to consider
        coeff_Awdow = 0 ; 
        struct_mat_Awdot = coeff_Awdow * struct_mat_Awdot_not_scaled;

         % aerodynamic stiffness scaling 
        struct_mat_Aw    = coeff_Aw * struct_mat_Aw_not_scaled ;


        % calculate the total stiffness term
        %    K_tot = K + K_s + Aw
        struct_mat_K_total = ...
            struct_mat_K + struct_mat_Aw;

        % calculate the total damping term if you choose to consider
        struct_mat_C_total = struct_mat_Awdot;

        % solve for the eigenvalues and eigenmodes of the
        % fluid-structure coupled system
        A = [ zeros(NModes_w), eye(NModes_w) ; ...
             -struct_mat_Minv * struct_mat_K_total,...
             -struct_mat_Minv * struct_mat_C_total];

        eigvals = eig(A);

        % collect the natural frequencies for post-processing
        natural_frequencies_hz = imag(eigvals);
        decay_rates            = real(eigvals);

        % keep positive frequencies only (can also be the negative).
        index_of_positive_frequencies = natural_frequencies_hz > 0;

        natural_frequencies_hz = natural_frequencies_hz...
            (index_of_positive_frequencies);
        decay_rates            = decay_rates...
            (index_of_positive_frequencies);

        % sort the natural frequencies from small to large, keep indices
        [natural_frequencies_hz, sort_idx] = sort(natural_frequencies_hz);
        decay_rates = decay_rates(sort_idx);

        % units from rads to Hz
        natural_frequencies_hz_array(:,idx) = ...
            natural_frequencies_hz / 2 / pi;
        damping_array(:,idx) = ...
            decay_rates ./ natural_frequencies_hz;

        % determine stability
        max_real_eig(idx) = max(real(eigvals));
        w_scale   = max(1, max(abs(imag(eigvals))));  % tolerance scale
        unstable(idx) = max_real_eig(idx) > (w_scale * tol);
    end

    flutter_idx = find(unstable, 1, 'first');
    if ~isempty(flutter_idx)
        lambda_F = lambda(flutter_idx);
    else    lambda_F = nan;
    end
end

function plot_output...
                (lambda, reduced_freq_array,...
                damping_array)

            % Plot flutter onset vs spring stiffness (pressure)
            figure();hold on;grid off;
            scatter(lambda,reduced_freq_array, 300, '.', 'MarkerEdgeAlpha', 1)
            set(gca,'FontSize',18)
            xlim([0, 1300]);
            ylabel('$k=\omega L/U_\infty$','Interpreter','latex','FontSize',50)
            xlabel('$\lambda$','Interpreter','latex','FontSize',50)
            
            figure();hold on;grid off;
            scatter(lambda, damping_array, 300, '.', 'MarkerEdgeAlpha', 1)
            set(gca,'FontSize',18)
            xlim([0, 1300]);
            ylabel('$\zeta$','Interpreter','latex','FontSize',50)
            xlabel('$\lambda$','Interpreter','latex','FontSize',50)
end

function reduced_freq_array = nondimentionalize...
    (gamma, T0, Minf, a, natural_frequencies_hz_array)
    % scalling parameters
    Rgas  = 287;                       % air [J/kg/K]
    a_inf = sqrt(gamma*Rgas*T0);       % speed of sound
    Uinf  = Minf * a_inf;              % freestream velocity
    Lref  = a;                         % choose L = a

    reduced_freq_array =...
        (2*pi*natural_frequencies_hz_array) * (Lref/Uinf);  % k = omega*L/U
end

function define_parallel_processing()
    p = gcp('nocreate');
    if isempty(p)
        parpool('IdleTimeout', Inf);
    end
end
