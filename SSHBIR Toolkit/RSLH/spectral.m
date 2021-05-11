function C = spectral(S, num_clusters)
% ?????
% ??Normalized????
% ??  : W              : N-by-N ??, ?????
%        sigma          : ?????,sigma????0
%        num_clusters   : ???
%
% ??  : C : N-by-1?? ????????
%
    format long
    m = size(S, 1);
%     %???????  ????????????????????????
%     W = W.*W;   %??
%     W = -W/(2*sigma*sigma);
%     S = full(spfun(@exp, W)); % ???S???????????????????????????????

    %?????D
    S=double(S);
    D = full(sparse(1:m, 1:m, sum(S))); %????D??????S????????????????????D

    % ???????? Do laplacian, L = D^(-1/2) * S * D^(-1/2)
    L = eye(m)-(D^(-1/2) * S * D^(-1/2)); %??????

    % ????? V
    %  eigs 'SM';????????
    [V, ~] = eigs(L, num_clusters, 'SM');
    % ??????k-means
    C=kmeans(V,num_clusters);
end
