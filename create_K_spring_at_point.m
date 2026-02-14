function [struct_mat_K_spring_not_scaled] = create_K_spring_at_point...
    (psi_w, x_spring, y_spring)
    % (Kc_not_scaled)_{nk} = psi_n(xc,yc) * psi_k(xc,yc)
    % so that later: Kc = K_spring * Kc_not_scaled
    
    NModes_w = length(psi_w);

    % Evaluate each mode shape at the spring location
    psi_at_c = zeros(NModes_w,1);
    for n = 1:NModes_w
        psi_at_c(n) = psi_w{n}(x_spring, y_spring);
    end
    struct_mat_K_spring_not_scaled = psi_at_c * psi_at_c.';
end
