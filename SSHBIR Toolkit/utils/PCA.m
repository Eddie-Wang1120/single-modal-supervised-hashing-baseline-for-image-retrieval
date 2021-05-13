
function B = PCA (X, q)
  n = size(X, 1);

  Sigma = X' * X / n;

  [U, S, V] = svd(Sigma);

  B = X * U(:, 1: q);
end
