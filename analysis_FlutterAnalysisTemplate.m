clc;clear all;close all;

% Parallel pool
define_parallel_processing();

%% Calculate the structural tensors
% we first create the structural model - calculate the structural tensors
% geometry and material parameters are hardcoded inside of that script
% TODO: finish the implementations in create_LinearPlateModel
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

% - point spring constraint
x_spring = 0.5 * a; y_spring = 0.0 * b;
K_spring = 20000    ; % it will be helpful to figure out the scaling of this in terms of a, h, D, rho_m


% physical solution visualization
x_point = reshape(xMesh,1,[]);  y_point = reshape(yMesh,1,[]);

tol = 1e-15; % dimensionless safety factor

%% Sweep stiffness
K_sweep = logspace(0, 6, 40);   % 1 to 1e6  [N/m]

flutter_lambda_vs_K = nan(size(K_sweep));

parfor iK = 1:length(K_sweep)
    [flutter_lambda_vs_K(iK)] =...
        flutter_response_for_location(...
        x_spring, y_spring, psi_w, K_sweep(iK), pinf_sweep, lambda, Minf,...
        struct_mat_K, struct_mat_Aw_not_scaled,...
        struct_mat_Awdot_not_scaled, struct_mat_Minv, NModes_w, tol);
end

%% Sweep spring location
N = [101, 101];                    % spring locations
x_sweep = linspace(0.05*a, 0.95*a, N(1));
y_sweep = linspace(-0.45*b, 0.45*b, N(2));

Npts = N(1)*N(2);

% preallocate
flutter_lambda_vec = nan(Npts,1);   flutter_pinf_vec   = nan(Npts,1);
coupling_ratio_vec = nan(Npts,1);   participation_ratio_vec = nan(Npts,1);

parfor pidx = 1:Npts
    [iy, ix] = ind2sub(N, pidx);

    % spring-induced mode coupling metrics at this location
    psi_c = zeros(NModes_w,1);
    for n = 1:NModes_w
        psi_c(n) = psi_w{n}(x_sweep(ix), y_sweep(iy));
    end

    [flutter_lambda_vec(pidx)] = flutter_response_for_location( ...
        x_sweep(ix), y_sweep(iy), psi_w, K_spring, pinf_sweep, lambda, Minf, ...
        struct_mat_K, struct_mat_Aw_not_scaled,...
        struct_mat_Awdot_not_scaled, struct_mat_Minv, NModes_w, tol);
end

% reshape back to maps
% location-flutter
flutter_lambda_map      = reshape(flutter_lambda_vec, N);
flutter_pinf_map        = reshape(flutter_pinf_vec,   N);

%% Sweep pressure
[xF, natural_frequencies_hz_array, damping_array, unstable, max_real_eig] = flutter_response_for_location( ...
    x_spring, y_spring, psi_w, K_spring, pinf_sweep, lambda, Minf, ...
    struct_mat_K, struct_mat_Aw_not_scaled, struct_mat_Awdot_not_scaled, struct_mat_Minv, NModes_w, tol);

if ~isnan(xF)
    fprintf('Flutter onset at lambda = %.3g\n', xF);
else
    fprintf('No flutter detected in the scanned range.\n');
end


reduced_freq_array = nondimentionalize...
    (gamma, T0, Minf, a, natural_frequencies_hz_array);
plot_output(x_sweep/a, y_sweep/b, flutter_lambda_map,...
    K_sweep*a^2/D, flutter_lambda_vs_K, lambda,...
    reduced_freq_array, damping_array);


function [lambda_F, natural_frequencies_hz_array,...
    damping_array, unstable, max_real_eig] = ...
    flutter_response_for_location...
    (x_spring, y_spring, psi_w, K_spring, pinf_sweep, lambda, Minf, ...
    struct_mat_K, struct_mat_Aw_not_scaled, struct_mat_Awdot_not_scaled,...
    struct_mat_Minv, NModes_w, tol)

    % TODO: finish the implementations in create_K_spring_at_point(...)
    struct_mat_K_spring_not_scaled = create_K_spring_at_point...
        (psi_w, x_spring, y_spring);
    struct_mat_K_spring = K_spring * struct_mat_K_spring_not_scaled;

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
            struct_mat_K + struct_mat_K_spring + struct_mat_Aw;

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
                (x_sweep_scaled, y_sweep_scaled, flutter_lambda_map,...
                K_sweep, flutter_lambda_vs_K, lambda, reduced_freq_array,...
                damping_array)
            % Plot flutter onset map
            figure(); hold on; grid off;
            imagesc(x_sweep_scaled, y_sweep_scaled, flutter_lambda_map);
            set(gca,'YDir','normal');
            axis equal;
            cb = colorbar;
            set(gca,'FontSize',18)
            ylabel(cb,'$\lambda_F$','Interpreter','latex','FontSize',50);
            xlabel('$x_c/a$','Interpreter','latex','FontSize',50);
            ylabel('$y_c/b$','Interpreter','latex','FontSize',50);

            % Plot flutter onset vs spring stiffness (nondimensional)
            figure(); hold on; grid off;box on;
            semilogx(K_sweep, flutter_lambda_vs_K, 'o-', 'LineWidth', 2.5, 'MarkerSize', 8);
            xlim([3, 12170]);
            set(gca,'FontSize',18)
            xlabel('$K_{\mathrm{spring}}$','Interpreter','latex','FontSize',50);
            ylabel('$\lambda$','Interpreter','latex','FontSize',50);

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
