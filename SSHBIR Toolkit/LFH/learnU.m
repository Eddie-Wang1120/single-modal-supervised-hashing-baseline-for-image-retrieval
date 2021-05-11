function [Ubest, MAPIter, LIter] = learnU (U0, S, beta, maxIter, convergeThresh, calcMAP)

  % initialization
  [N, Q] = size(U0);
  iND = find(~eye(N));

  U = U0;
  T = U * U' / 2;
  A = 1 ./ (1 + exp(-T));
  L = sum(S(iND) .* T(iND)) - sum(logExpTrick(T(iND))) - trace(T) / beta;

  MAP = 0;
  MAPIter(1) = MAP;
  maxMAP = MAP;
  iterBest = 1;
  Ubest = U;
  LIter(1) = L;

  Lp = L; bU = L; bL = L;

  p1 = timerStart();
  for iter = 1: maxIter

    % update U
    for i = 1: N
      j = [1: i - 1, i + 1: N];
      Gi = ((S(i, j) - A(i, j)) + (S(j, i) - A(j, i))') * U(j, :) / 2 - U(i, :) / beta;
      Hi = -U(j, :)' * U(j, :) / 8 - eye(Q) / beta;
      U(i, :) = U(i, :) - Gi / Hi;
      T(i, :) = U(i, :) * U' / 2;
      T(:, i) = T(i, :)';
      A(i, :) = 1 ./ (1 + exp(-T(i, :)));
      A(:, i) = A(i, :)';
    end

    % compute objective
    %L = sum(S(iND) .* T(iND)) - sum(logExpTrick(T(iND))) - trace(T) / beta;

    % record converge process
    %MAP = calcMAP(U);
    %MAPIter(iter + 1) = MAP;
    %if MAP > maxMAP
    %  maxMAP = MAP;
    %  iterBest = iter + 1;
      Ubest = U;
    %end
    %LIter(iter + 1) = L;

    % check for convergence
    %bU = max(bU, L);
    %bL = min(bL, L);
    %if abs(L - Lp) / (bU - bL) < convergeThresh
    %  break;
    %end
    %Lp = L;

  end
  fprintf('  Number of iterations: %d\n', iter);
  t1 = timerStop(p1);
  fprintf('  Average time per iteration: %.4gs\n', t1.etime / iter);
  fprintf('  Best iteration found at %d with MAP %.4g\n', iterBest - 1, maxMAP);

end
