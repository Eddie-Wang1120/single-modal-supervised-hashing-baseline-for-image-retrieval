function Ubest = learnUStochasticQuick2 (U0, calcS, beta, sampleColumn, maxIter)

  [N, Q] = size(U0);

  U = U0;
  
  Ubest = U;

  for iter = 2: maxIter + 1

    % sample a subset of columns
    Sc = randsample(N, sampleColumn);

    % update rows of U not in Sc
    S = calcS(Sc);
    T = U * U(Sc, :)' / 2;
    A = 1 ./ (1 + exp(-T));
    ix = setdiff(1: N, Sc);
    G = (S(ix, :) - A(ix, :)) * U(Sc, :) - U(ix, :) / beta; % NQ^2
    H = - U(Sc, :)' * U(Sc, :) / 8 - eye(Q) / beta;
    U(ix, :) = U(ix, :) - G / H; 
	Ubest=U;
   
  end

end
