function C = spectral(S, num_clusters)
    format long
    m = size(S, 1);
    S=double(S);
    D = full(sparse(1:m, 1:m, sum(S)));

    L = eye(m)-(D^(-1/2) * S * D^(-1/2));

    [V, ~] = eigs(L, num_clusters, 'SM');
    C=kmeans(V,num_clusters);
end
