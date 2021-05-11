% use PCA to reduce X to dimensionality of q.
function B = PCA (X, q)
  % number of data points
  n = size(X, 1);
%keyboard;
  % covariance matrix
  Sigma = X' * X / n;
%keyboard;
  % compute SVD
  [U, S, V] = svd(Sigma);
%keyboard;
  % compute mapped value in dimension of q
  B = X * U(:, 1: q);
end
