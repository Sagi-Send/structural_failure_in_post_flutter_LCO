clc;clear all;close all;

%% Calculate the structural tensors
% we first create the strcutural model - calculate the structural tensors
% geometry and material parameters are hardcoded inside of that script

% at this step the following script is complete
create_AiryStressPlateModel

% analysis parameters - nonlinear vibration 
static_pressure_sweep        = linspace(0,10e3,20); % [Pa], static pressure differential values
q_qdot_ics                   = zeros(2 * NModes_w,1); % initial conditions - zeros - the plate starts flat

% nonlinear static equilibrium solver parameters
T_max_nonlinear_solution = 0.1;   % time marching duration

% physical solution visualization
x_point = reshape(xMesh,1,[]);
y_point = reshape(yMesh,1,[]);


%% PART 1 - solve a single case and visualize

deltaP_single_case = 1e3 ; % [kPa]
struct_mat_Q = deltaP_single_case * struct_mat_Q_not_scaled ; % this is the static load generalized force

% define the RHS of the ODE - this is the dynamic equation in modal
% coordinates
% TODO: step inside the rhs_func and implement the RHS operations
rhs_local = @(t,y) rhs_func(t, y, NModes_w, struct_mat_Minv, ...
                                struct_mat_K, struct_mat_L2, struct_mat_Q);
% solve ODE by time marching
[t,w_modal] = ode45(rhs_local, [0 T_max_nonlinear_solution], q_qdot_ics);

% PLOT solution pseudo-transient
% use this figure to test different solver parameters
figure();hold on;grid on;grid minor;
plot(t,w_modal(:,1:5) / h)
xlabel('$t, sec$','Interpreter','latex','FontSize',20)
ylabel('$w_i / h$','Interpreter','latex','FontSize',20)
title('Static equilibrium solution convergence')


% TODO: extract the nonlinear static solution in modal coordinates
%       you have the w_modal solution from pseudo time-marching, but
%       which part of it do we need?
% ---------- YOUR CODE HERE - START ----------
w_modal_nonlinear_static_single_case = w_modal(end, 1:NModes_w);
% ---------- YOUR CODE HERE - END ----------

% transform the nonlinear static solution from modal to physical
% coordinates and PLOT
w_phys_nonlinear_static_single_case = modal2physical(w_modal_nonlinear_static_single_case, x_point, y_point, psi_w);

figure();hold on;grid on;grid minor;
surf(xMesh/a, yMesh/b,reshape(w_phys_nonlinear_static_single_case / h,size(xMesh)))

xlabel('$x / a$','Interpreter','latex','FontSize',20)
ylabel('$y/b$','Interpreter','latex','FontSize',20)
zlabel('$w/h$','Interpreter','latex','FontSize',20)
view([-30 20])
title(['Static equilibrium solution in physical coordinates'],'Interpreter','latex','FontSize',20)
colormap copper
shading interp


%% PART 2 - loop over static pressure differenetial values
%           Now we combine everything together: solving the static
%           nonlinear solution, eigenvalue analysis, solution processing,
%           and visualization.

for idx = 1:length(static_pressure_sweep)
    dp_i = static_pressure_sweep(idx);
    % 1. solve nonlinear static equilibrium
    struct_mat_Q = dp_i * struct_mat_Q_not_scaled ; % this is the static load we use in this iteration 

    % define the RHS of the ODE for this iteration
    rhs_local = @(t,y) rhs_func( ...
                                    t, ...
                                    y, ...
                                    NModes_w, ...
                                    struct_mat_Minv, ...
                                    struct_mat_K, ...
                                    struct_mat_L2, ...
                                    struct_mat_Q);

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


    % 3. find the eigenvalues and eigenmodes of the deformed structure
    %    K_tot = K + K_NL(w_static)
    struct_mat_K_total = struct_mat_K - struct_mat_K_deformed;

    A = [ zeros(NModes_w), eye(NModes_w) ; ...
         -1 * inv(struct_mat_M) * struct_mat_K_total, zeros(NModes_w)];
    [eigen_vectors, eigen_values] = eig(A);
    

    % 4. collect the natural frequencies for post-processing
    % take the imaginary part - oscillation frequency
    % take only the diagonal of eigen_values (the rest are zeros).
    natural_frequencies_hz = imag(diag(eigen_values)) ;
    
    % now we rememeber that we have REPEATING (conjugate pairs) eigenvalues because of the nature
    % of our problem. We need to keep only one of each. Keep the positive
    % frequencies only (can also be the nagative).
    index_of_positive_frequencies = natural_frequencies_hz>0;
    
    % get the respective natural mode shapes and frequencies
    natural_mode_shapes = eigen_vectors(NModes_w+1:2*NModes_w,index_of_positive_frequencies);
    natural_frequencies_hz = natural_frequencies_hz(index_of_positive_frequencies);
    
    % sort the natural frequencies from small to large, keep the indices
    [natural_frequencies_hz, sort_idx] = sort(natural_frequencies_hz);
    natural_mode_shapes = natural_mode_shapes(:,sort_idx); % sort mode shapes accordingly
    
    % units from rads to Hz
    natural_frequencies_hz_array(:,idx) = natural_frequencies_hz / 2 / pi;
end


figure();hold on;grid on;grid minor;
plot(static_pressure_sweep,w_max / h)
plot(static_pressure_sweep,static_pressure_sweep * w_max(2) / h / static_pressure_sweep(2),'--')
ylabel('$w_{center} / h$','Interpreter','latex','FontSize',20)
xlabel('$\Delta p, \, kPa$','Interpreter','latex','FontSize',20)
legend('Nonlinear','Linear')

figure();hold on;grid on;grid minor;
plot(static_pressure_sweep,natural_frequencies_hz_array)
ylabel('$w_{center} / h$','Interpreter','latex','FontSize',20)
xlabel('$\Delta p, \, kPa$','Interpreter','latex','FontSize',20)
ylim([0 1000])
