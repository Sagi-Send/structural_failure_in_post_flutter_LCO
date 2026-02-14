clc;clear all;close all;

% Parallel pool
define_parallel_processing();

%% Calculate the structural tensors
% we first create the structural model - calculate the structural tensors
% geometry and material parameters are hardcoded inside of that script
create_AiryStressPlateModel

% analysis parameters
T_max_nonlinear_solution = 1;
q_qdot_ics                   = zeros(2 * NModes_w,1); % initial conditions - zeros - the plate starts flat

% physical solution visualization
x_point = reshape(xMesh,1,[]);
y_point = reshape(yMesh,1,[]);

% - aerodynamics
disc = 200;
pinf_sweep  = linspace(0,75e3,disc); % [Pa], freestream static pressure values
gamma       = 1.4;
Minf        = 4.0 ; % freestream Mach number
T0          = 400 ; % [K] - flow temperature is only important for the aerodynamic damping term

% get the nondimensional parameter
lambda = gamma * pinf_sweep * Minf * (a^3 / D);  % this is equal to rho_inf * U_inf^2 * (a^3/D)

% physical solution visualization
x_point = reshape(xMesh,1,[]);  y_point = reshape(yMesh,1,[]);

tol = 1e-15; % dimensionless safety factor

%% Sweep pressure
[w_max, xF, natural_frequencies_hz_array, damping_array, unstable, max_real_eig] = pressure_sweep( ...
    psi_w, pinf_sweep, lambda, Minf, ...
    struct_mat_K, struct_mat_Aw_not_scaled, struct_mat_Minv, NModes_w, tol, T_max_nonlinear_solution, q_qdot_ics,struct_mat_L2, x_point, y_point, a);

if ~isnan(xF)
    fprintf('Flutter onset at lambda = %.3g\n', xF);
else
    fprintf('No flutter detected in the scanned range.\n');
end


reduced_freq_array = nondimentionalize...
    (gamma, T0, Minf, a, natural_frequencies_hz_array);
plot_output(w_max, pinf_sweep, lambda, reduced_freq_array, damping_array);


function [w_max, lambda_F, natural_frequencies_hz_array, damping_array,...
    unstable, max_real_eig] = pressure_sweep(psi_w,...
    pinf_sweep, lambda, Minf, struct_mat_K, struct_mat_Aw_not_scaled,...
    struct_mat_Minv, NModes_w, tol, T_max_nonlinear_solution, q_qdot_ics,struct_mat_L2, x_point, y_point, a)

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

         % aerodynamic stiffness scaling 
        struct_mat_Aw    = coeff_Aw * struct_mat_Aw_not_scaled ;

    % define the RHS of the ODE for this iteration
    rhs_local = @(t,y) rhs_func_aero( ...
                                    t, ...
                                    y, ...
                                    NModes_w, ...
                                    struct_mat_Minv, ...
                                    struct_mat_K, ...
                                    struct_mat_L2, struct_mat_Aw);

    % solve ODE by time marching
    [t,w_modal] = ode45(rhs_local, [0 T_max_nonlinear_solution], q_qdot_ics);

    % calculate the physical displacement
    %   use the last time step of the solution in w_modal, and use only
    %   the displacement components (ignore velocity components)
    w_phys_all = modal2physical(w_modal(end,1:NModes_w), x_point, y_point, psi_w);
    w_phys_center = modal2physical(w_modal(end,1:NModes_w), a/2, 0.0, psi_w);

    % extract the maximum displacement and the displacement at the plate
    % center (they can generally be different)
    w_max(idx) = max(abs(w_phys_all),[],"all"); % the maximum displacement
    w_center(idx) = abs(w_phys_center); % displacement at the center of the plate is at x = a/2, y = 0
    % obtain and SAVE w_static for plotting the deformed shape

    % we extract the static equilibrium solution
    w_modal_nonlinear_static(:,idx) = w_modal(end,1:NModes_w);

    % 2. calculate the added stiffness term
    %    use w_modal_nonlinear_static and the structural nonlinear
    %    stiffness
    w_s = w_modal_nonlinear_static(:,idx); % for easier notation

    % TODO: complete added stiffness implementation
    % ---------- YOUR CODE HERE - START ----------
    struct_mat_K_deformed = tensorprod(tensorprod(struct_mat_L2, w_s, 2, 1), w_s, 3, 1) ...
                      + tensorprod(tensorprod(struct_mat_L2, w_s, 3, 1), w_s, 3, 1) ...
                      + tensorprod(tensorprod(struct_mat_L2, w_s, 2, 1), w_s, 2, 1);
    % ---------- YOUR CODE HERE - END ----------

        % calculate the total stiffness term
        %    K_tot = K + K_s + Aw
        struct_mat_K_total = struct_mat_K + struct_mat_Aw - struct_mat_K_deformed;

        % solve for the eigenvalues and eigenmodes of the
        % fluid-structure coupled system
        A = [ zeros(NModes_w), eye(NModes_w) ; ...
             -struct_mat_Minv * struct_mat_K_total,...
             zeros(NModes_w)];

        eigvals = eig(A);

        % collect the natural frequencies for post-processing
        natural_frequencies_hz = imag(eigvals); decay_rates = real(eigvals);

        % keep positive frequencies only (can also be the negative).
        index_of_positive_frequencies = natural_frequencies_hz > 0;

        natural_frequencies_hz = natural_frequencies_hz...
            (index_of_positive_frequencies);
        decay_rates = decay_rates(index_of_positive_frequencies);

        % sort the natural frequencies from small to large, keep indices
        [natural_frequencies_hz, sort_idx] = sort(natural_frequencies_hz);
        decay_rates = decay_rates(sort_idx);

        % units from rads to Hz
        natural_frequencies_hz_array(:,idx) = ...
            natural_frequencies_hz / 2 / pi;
        damping_array(:,idx) = decay_rates ./ natural_frequencies_hz;

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

function plot_output(w_max, p_sweep, lambda, reduced_freq_array, damping_array)

            % Plot flutter onset vs spring stiffness (pressure)
            % figure();hold on;grid off;
            % scatter(lambda,reduced_freq_array, 300, '.', 'MarkerEdgeAlpha', 1)
            % set(gca,'FontSize',18)
            % xlim([0, 1300]);
            % ylabel('$k=\omega L/U_\infty$','Interpreter','latex','FontSize',50)
            % xlabel('$\lambda$','Interpreter','latex','FontSize',50)
            % 
            % figure();hold on;grid off;
            % scatter(lambda, damping_array, 300, '.', 'MarkerEdgeAlpha', 1)
            % set(gca,'FontSize',18)
            % xlim([0, 1300]);
            % ylabel('$\zeta$','Interpreter','latex','FontSize',50)
            % xlabel('$\lambda$','Interpreter','latex','FontSize',50)

            figure();hold on;grid off;
            scatter(p_sweep, w_max , '.', 'MarkerEdgeAlpha', 1)
            set(gca,'FontSize',18)
            xlim([0, max(p_sweep)]);
            ylabel('$w_{max}$','Interpreter','latex','FontSize',50)
            xlabel('$p_{\infty}$','Interpreter','latex','FontSize',50)
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
