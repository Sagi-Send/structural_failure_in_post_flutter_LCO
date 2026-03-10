function f = rhs_func_aero(t, y, NModes_w, struct_mat_Minv, struct_mat_K, ...
                        struct_mat_L2, Aw, Awdot, include_nonlinearity)
    if nargin < 9 || isempty(include_nonlinearity)
        include_nonlinearity = true;
    end

    q    = y(1:NModes_w,1);
    qdot = y(NModes_w + 1: NModes_w * 2,1);

    if include_nonlinearity
        Lq   = tensorprod(struct_mat_L2, q, 4, 1);
        Lqq  = tensorprod(Lq, q, 3, 1);
        nonlinear_term = tensorprod(Lqq, q, 2, 1);
    else
        nonlinear_term = zeros(NModes_w,1);
    end

    f(1:NModes_w,1) = qdot;
    f(NModes_w + 1: NModes_w * 2,1) = -struct_mat_Minv * ( ...
        (struct_mat_K + Aw) * q + Awdot * qdot - nonlinear_term);
end
