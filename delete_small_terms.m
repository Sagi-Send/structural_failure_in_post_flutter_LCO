function [X_clean] = delete_small_terms(X, eps_small_terms)
    %           Delete small terms terms to get rid of numerical zeros
    %           because we intergrate numerically using trapz, we never obtain 
    %           zero values. BUT, many of the elements of our matrices are actually zero.
    %           To address this, we can do the following removal of small terms.
    %           It is suggested to wrap the following in a function.
    % eps_small_terms: order of magnitude difference between the largest and smallest absolute value in the matrix
    temp1 = max(abs(X),[],'all'); % maximum value
    temp2 = abs(X) ./ temp1;      % normalized and abs matrix
    X_clean = X;                  % we keep the original, for debug
    X_clean(temp2 < 10^(-eps_small_terms)) = 0;
end
