function f = rhs_func_aero(t, y, NModes_w, struct_mat_Minv, struct_mat_K, ...
                        struct_mat_L2, Aw, Awdot)
    q    = y(1:NModes_w,1);
    qdot = y(NModes_w + 1: NModes_w * 2,1);

    % evaluate the nonlinear stiffness at q
    Lq   = tensorprod(struct_mat_L2,q,4,1);

    % TODO: complete the calculation of the nonlinear stiffness term
    % ---------- YOUR CODE HERE - START ----------
    Lqq  = tensorprod(Lq , q, 3, 1);   % (n,i)
    Lqqq = tensorprod(Lqq, q, 2, 1);   % (n)
    % ---------- YOUR CODE HERE - END ----------

    f(1:NModes_w,1) = qdot;


    % TODO: complete the calculation of RHS forces term
    % ---------- YOUR CODE HERE - START ----------
    f(NModes_w + 1: NModes_w * 2,1) = -1 * struct_mat_Minv * ( ...
        (struct_mat_K + Aw) * q + Awdot * qdot - Lqqq);
    % ---------- YOUR CODE HERE - END ----------
end
