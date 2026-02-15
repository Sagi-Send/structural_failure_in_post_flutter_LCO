function w_phys = modal2physical(q, x_point, y_point, psi_w)
    % q: is the modal displacement vector of dimension [T x NModes_w]
    %    where T is the time series length, it can also be equal to 1 for a
    %    single deformed shape solution

    % x_point and x_point: are of dimension [num_of_xy_points x 1]

    % psi_w: is a cell array of length NModes_w
    %   such that at psi_w{n}(x_point, y_point) we get the modal
    %   displacement w of the n-th mode at the points (x_point, y_point)
    %   the output is a vector.

    % Here we implement a function that transforms the time series 
    %       of modal displacement q to time series of physical displacement
    %       and return w_phys of dimension [T x num_of_xy_points]

    % we obtain the the deformation in physical coordinates in two steps
    % 1) get the physical deformation on the mesh grid for each mode
    num_of_xy_points = length(x_point);
    psi_w_at_xy = zeros(length(psi_w), num_of_xy_points);
    for n = 1:length(psi_w)
        % psi_w_at_xy is [NModes_w x num_of_xy_points]
        psi_w_at_xy(n,:) = psi_w{n}(x_point, y_point);
    end

    % and step 2) multiply by modal coordinates
    % q is [T x NModes_w]
    % psi_w_at_xy is [NModes_w x num_of_xy_points]
    
    % w_phys - of dimension [T x num_of_xy_points]
    w_phys = q * psi_w_at_xy; 
    
end
