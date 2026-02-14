%% DEFINE PARAMETERS
%   plate chord length, and width are used in the difinition of the basis function
%   so they must be defined every time we modify geometry.
a = 0.25  ;
b = 0.125  ;

% plate thickness
h = 0.65e-3;

% material properties
E = 200e9;
nu = 0.33;
rho_m = 7800;
m_s = h * rho_m;
D = E * h^3 / (12 * (1-nu^2));


% Numerical integration settings -----------------------------------------
integrationPoints = 200;
xInterval         = linspace(0   ,a  , integrationPoints);
yInterval         = linspace(-b/2,b/2, integrationPoints);
[xMesh,yMesh]     = meshgrid(xInterval, yInterval);
integrand         = zeros(integrationPoints, integrationPoints);


% Mode ordering --------------------------------------------------------
modeOrder_w_x  = 1:4 ;  % we use 4 chord-wise modes
modeOrder_w_y  = 1:4 ;  % and 4 span-wise modes

% now collect them in pairs
count = 1;
for i = modeOrder_w_x
    for j = modeOrder_w_y
        modeOrder_w(count,1:2) = [i,j];
        count = count + 1;
    end
end
NModes_w  = length(modeOrder_w(:,1));


%% Assemble the basis functions 
% ALL PINNED (simply-supported) BCS
syms X Y x y
for n = 1:1:NModes_w
    mx = modeOrder_w(n,1);
    my = modeOrder_w(n,2);

    X = sin( pi* mx * x / a )             ;
    Y = sin( pi* my * ( y + b / 2 ) / b ) ;  % shifted to center

    psi_w_sym(n) = X*Y;
end

%% Transform the SYMBOLIC variables to functions that we can EVALUATE
%   at specific values. We no longer need symbolic expressions from
%   that point.

% initialize
psi_w = cell(NModes_w,1);
psi_w_x = cell(NModes_w,1);     psi_w_y = cell(NModes_w,1);
psi_w_xx = cell(NModes_w,1);    psi_w_yy = cell(NModes_w,1);
psi_w_xy = cell(NModes_w,1);    psi_w_bihar = cell(NModes_w,1);

for n = 1:NModes_w
    % EXAMPLE - the basis function
    psi_w{n}    = matlabFunction(psi_w_sym(n),'Vars',[x y]);

    % EXAMPLE - and the first partial derivative of the basis function
    %   w.r.t to x and y
    %   note we use the diff() function BEFORE we transform the symbolic
    %   variable
    psi_w_x{n}  = matlabFunction(diff(psi_w_sym(n),x,1),'Vars',[x y]);
    psi_w_y{n}  = matlabFunction(diff(psi_w_sym(n),y,1),'Vars',[x y]);

    psi_w_xx{n}  = matlabFunction(diff(psi_w_sym(n),x,2),'Vars',[x y]);
    psi_w_yy{n}  = matlabFunction(diff(psi_w_sym(n),y,2),'Vars',[x y]);



    % TODO: implement the mixed derivative w.r.t to x and y
    %       by applying the same idea (use diff TWICE)
    % ---------- YOUR CODE HERE - START ----------
    psi_w_xy{n}  = matlabFunction(diff(diff(psi_w_sym(n),x,1),y,1),'Vars',[x y]);
    % ---------- YOUR CODE HERE - END ----------


    % TODO: implement the bi-harmonic operator \del^4 phi_n
    %    \del^4 psi_w_sym(n) = 
    %    psi_w_sym(n)_xxxx + 2 * psi_w_sym(n)_xxyy + psi_w_sym(n)_yyyy
    % ---------- YOUR CODE HERE - START ----------
    % derive fourth order derivatives and biharmonic function
    psi_w_xxxx{n}  = matlabFunction(diff(psi_w_sym(n),x,4),'Vars',[x y]);
    psi_w_yyyy{n}  = matlabFunction(diff(psi_w_sym(n),y,4),'Vars',[x y]);
    psi_w_xxyy{n}  = matlabFunction(diff(diff(psi_w_sym(n),y,2),x,2),'Vars',[x y]);
    psi_w_bihar{n}  = @(x,y) psi_w_xxxx{n}(x,y) + 2 * psi_w_xxyy{n}(x,y) + psi_w_yyyy{n}(x,y);
    % ---------- YOUR CODE HERE - END ----------

end


% evaluate shape functions at the mesh points ----------------------------
% initialize cell arrays
psi_w_mesh        = repmat({zeros(size(xMesh))}, 1, NModes_w);
psi_w_x_mesh      = repmat({zeros(size(xMesh))}, 1, NModes_w);
psi_w_y_mesh      = repmat({zeros(size(xMesh))}, 1, NModes_w);
psi_w_xx_mesh     = repmat({zeros(size(xMesh))}, 1, NModes_w);
psi_w_yy_mesh     = repmat({zeros(size(xMesh))}, 1, NModes_w);
psi_w_xy_mesh     = repmat({zeros(size(xMesh))}, 1, NModes_w);
psi_w_bihar_mesh  = repmat({zeros(size(xMesh))}, 1, NModes_w);

for i=1:NModes_w
    % we do this to significantly reduce calculation time
    psi_w_mesh{i}        = psi_w{i}(xMesh,yMesh);
    psi_w_x_mesh{i}      = psi_w_x{i}(xMesh,yMesh);
    psi_w_y_mesh{i}      = psi_w_y{i}(xMesh,yMesh);
    psi_w_xx_mesh{i}     = psi_w_xx{i}(xMesh,yMesh);
    psi_w_yy_mesh{i}     = psi_w_yy{i}(xMesh,yMesh);
    psi_w_xy_mesh{i}     = psi_w_xy{i}(xMesh,yMesh);
    psi_w_bihar_mesh{i}  = psi_w_bihar{i}(xMesh,yMesh);
end
% from this point we work with cell arrays that end in _mesh
% these are the INTEGRAND functions that we will integrate to obtain
% the structural tensors


%% Calculate the structural tensors
% Three structural tensors are provided as examples
% the simplest one is the mass matrix struct_mat_M_not_scaled
% a more complicated one is the stiffness matrix struct_mat_K_not_scaled
% and the most complicated is struct_mat_L_not_scaled and also provided
% you have to implement struct_mat_A_not_scaled, struct_mat_B_not_scaled
% and struct_mat_Q_not_scaled

% EXAMPLE - calculate the mass matrix
%           NOTE that it is NOT SCALED by material properties
%           we do this to be able to avoid recalculating it when 
%           varying material, thickness, or other parameters
% M_ni ------------------------------------------------------------------
struct_mat_M_not_scaled = zeros(NModes_w, NModes_w); % always initialize first
for n = 1:NModes_w
    for i = 1:NModes_w
        integrand  =  psi_w_mesh{n} .* psi_w_mesh{i}; % this is an element-wise product, 
        % pay close attention to the INDICES (n,i), the matrix or tensor is NOT ALWAYS SYMMETRIC 
        struct_mat_M_not_scaled(n,i) = trapz(yInterval,trapz(xInterval,integrand,2)) ;
    end
end

% K_ni ------------------------------------------------------------------
struct_mat_K_not_scaled = zeros(NModes_w, NModes_w);
for n = 1:NModes_w
    for i = 1:NModes_w
        integrand  =  psi_w_mesh{n} .* psi_w_bihar_mesh{i};
        struct_mat_K_not_scaled(n,i) = trapz(yInterval,trapz(xInterval,integrand,2)) ;
    end
end

% L_nis ------------------------------------------------------------------
struct_mat_L_not_scaled = zeros(NModes_w, NModes_w, NModes_w);
for n = 1:NModes_w
    for i = 1:NModes_w
        for s = 1:NModes_w
            integrand  =  psi_w_mesh{n} .* ...
                ( ...
                psi_w_yy_mesh{i} .* psi_w_xx_mesh{s} + psi_w_xx_mesh{i} .* psi_w_yy_mesh{s} ...
                - 2 * psi_w_xy_mesh{i} .* psi_w_xy_mesh{s} ...
                );
            struct_mat_L_not_scaled(n,i,s) = trapz(yInterval,trapz(xInterval,integrand,2)) ;
        end
    end
end

% A_ns ------------------------------------------------------------------
% TODO: implement the calculation of struct_mat_A_not_scaled(n,s)
struct_mat_A_not_scaled = zeros(NModes_w, NModes_w);
for n = 1:NModes_w
    for s = 1:NModes_w
        % ---------- YOUR CODE HERE - START ----------
        integrand =  psi_w_mesh{n} .* psi_w_bihar_mesh{s};
        struct_mat_A_not_scaled(n,s) = trapz(yInterval, trapz(xInterval, integrand, 2));
        % ---------- YOUR CODE HERE - END ----------
        
    end
end


% B_nik ------------------------------------------------------------------
% TODO: implement the calculation of struct_mat_B_not_scaled(n,i,k)
struct_mat_B_not_scaled = zeros(NModes_w, NModes_w, NModes_w);
for n = 1:NModes_w
    for i = 1:NModes_w
        for k = 1:NModes_w
            % ---------- YOUR CODE HERE - START ----------
            integrand  =  psi_w_mesh{n} .* ( ...
                psi_w_xy_mesh{i} .* psi_w_xy_mesh{k} - psi_w_xx_mesh{i} .* psi_w_yy_mesh{k});
            struct_mat_B_not_scaled(n,i,k) = trapz(yInterval, trapz(xInterval, integrand, 2));
            % ---------- YOUR CODE HERE - END ----------

        end
    end
end


% Q_n ------------------------------------------------------------------
% TODO: implement the calculation of struct_mat_Q_not_scaled(n)
% calculate Q_not_scaled - it's a vector, not a matrix/tensor!
static_pressure_distribution = ones(size(xMesh)); % the distribution of external load - uniform
struct_mat_Q_not_scaled = zeros(NModes_w,1);
for n = 1:NModes_w
    % ---------- YOUR CODE HERE - START ----------
    integrand  =  static_pressure_distribution .* psi_w_mesh{n};
    struct_mat_Q_not_scaled(n) = trapz(yInterval, trapz(xInterval, integrand, 2));
    % ---------- YOUR CODE HERE - END ----------

end


% remove small terms, order of magnitude comparison
% we do this because numerical integration is approximate and we get small
% valued noise where exact zeros are expected
eps_small_terms = 10;
struct_mat_M_not_scaled = delete_small_terms(struct_mat_M_not_scaled, eps_small_terms);
struct_mat_K_not_scaled = delete_small_terms(struct_mat_K_not_scaled, eps_small_terms);
struct_mat_A_not_scaled = delete_small_terms(struct_mat_A_not_scaled, eps_small_terms);
struct_mat_B_not_scaled = delete_small_terms(struct_mat_B_not_scaled, eps_small_terms);
struct_mat_L_not_scaled = delete_small_terms(struct_mat_L_not_scaled, eps_small_terms);
struct_mat_Q_not_scaled = delete_small_terms(struct_mat_Q_not_scaled, eps_small_terms);


%% rescale the structural tensors, use material properties
% this is where we apply the material properties and set the THICKNESS
struct_mat_M = struct_mat_M_not_scaled * m_s;
struct_mat_K = struct_mat_K_not_scaled * D;
struct_mat_A = struct_mat_A_not_scaled  / E / h;
struct_mat_B = struct_mat_B_not_scaled;
struct_mat_L = struct_mat_L_not_scaled;

% useful definitions - will be used in ODE solution
struct_mat_Minv = inv(struct_mat_M);


% ------------------------------------------------------------------
%% Solve the algebraic problem for the in-plane equilibrium equations
% TODO: use the solution shown in class
%       use tensorprod(A,B,DIMA,DIMB) to do the tensor product
%       and obtain the L2 tensor
%       a partial solution is given in this part


% ---------- YOUR CODE HERE - START ----------
struct_mat_A_inv = inv(struct_mat_A);
struct_mat_B2    = tensorprod(struct_mat_A_inv,struct_mat_B,2,1);
struct_mat_L2    = tensorprod(struct_mat_L,struct_mat_B2, 3, 1);
% ---------- YOUR CODE HERE - END ----------


% Aw_ni ------------------------------------------------------------------
% TODO: implement the calculation of struct_mat_Aw_not_scaled
struct_mat_Aw_not_scaled = zeros(NModes_w, NModes_w);
for n = 1:NModes_w
    for i = 1:NModes_w
        integrand  = psi_w_mesh{n} .* psi_w_x_mesh{i};
        struct_mat_Aw_not_scaled(n,i) =...
            trapz(yInterval,trapz(xInterval,integrand,2)) ;
    end
end

