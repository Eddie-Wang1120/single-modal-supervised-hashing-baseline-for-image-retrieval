% use PCA to reduce X to dimensionality of q.
function B = PCA (X, q)
  % number of data points
  n = size(X, 1);

  % covariance matrix
  Sigma = X' * X / n;

  % compute SVD
  [U, S, V] = svd(Sigma);

  % compute mapped value in dimension of q
  B = X * U(:, 1: q);
end
