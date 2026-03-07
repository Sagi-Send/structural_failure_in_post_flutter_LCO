function [vm_upper, vm_lower] = von_mises( ...
    Q, Psi_xx, Psi_yy, Psi_xy, struct_mat_B2, h, nu, D)

% Q: [Nt x N], Psi_**: [N x nPts]
[Nt, N] = size(Q);

% curvatures at points over time (nPts x Nt)
w_xx = (Q * Psi_xx).';
w_yy = (Q * Psi_yy).';
w_xy = (Q * Psi_xy).';

% bending moments (nPts x Nt)
Mxx = -D * (w_xx + nu * w_yy);
Myy = -D * (w_yy + nu * w_xx);
Mxy = -D * (1 - nu) * w_xy;

% Airy coefficients c(t): F(x,y,t) = sum c_n(t) psi_n(x,y)
% c = (A^{-1}B) : (q ⊗ q) = struct_mat_B2(q,q)
Ccoef = zeros(Nt, N);
for it = 1:Nt
    q = Q(it,:).';                                % [N x 1]
    tmp = tensorprod(struct_mat_B2, q, 3, 1);    % -> [N x N]
    c   = tensorprod(tmp, q, 2, 1);              % -> [N x 1]
    Ccoef(it,:) = c.';
end

% Airy second derivatives at points over time (nPts x Nt)
F_xx = (Ccoef * Psi_xx).';
F_yy = (Ccoef * Psi_yy).';
F_xy = (Ccoef * Psi_xy).';

% stress resultants from Airy
Nxx = F_yy;
Nyy = F_xx;
Nxy = -F_xy;

% membrane stresses
sxx_m = Nxx / h;
syy_m = Nyy / h;
sxy_m = Nxy / h;

% bending stresses at z = ±h/2
coef = 6 / h^2;   % (12z/h^3) with z=±h/2
sxx_b = coef * Mxx;
syy_b = coef * Myy;
sxy_b = coef * Mxy;

% upper (+h/2) and lower (-h/2)
sxxU = sxx_m + sxx_b;   syyU = syy_m + syy_b;   sxyU = sxy_m + sxy_b;
sxxL = sxx_m - sxx_b;   syyL = syy_m - syy_b;   sxyL = sxy_m - sxy_b;

% von Mises (plane stress)
vm_upper = sqrt(sxxU.^2 - sxxU.*syyU + syyU.^2 + 3*sxyU.^2);
vm_lower = sqrt(sxxL.^2 - sxxL.*syyL + syyL.^2 + 3*sxyL.^2);

end
