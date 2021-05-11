% A LFH wrapper for preparing appropriate data for the core implementation
% Input:
%   dataset: a structure containing dataset info
%   method: a structure containing method info
%   codeLength: length of the binary codes
% Output:
%   B1: binary encoding of the trainning data
%   B2: binary encoding of the testing data
%   t1: training time
%   t2: testing time
function [B1, B2, t1, t2] = LFH_nstop (dataset, method, codeLength)

  % set default values for method properties
  if ~isfield(method, 'learnImpl')
    method.learnImpl = 'default';
  end
  if ~isfield(method, 'normalize')
    method.normalize = false;
  end
  if ~isfield(method, 'debug')
    method.debug = true;
  end
  if ~isfield(method, 'selectParam')
    method.selectParam = false;
  end
  if method.selectParam && ~isfield(method, 'selectParamMethod')
    method.selectParamMethod = 'linear';
  end
  if ~isfield(method, 'initMethod')
    method.initMethod = 'PCA';
  end
  if ~isfield(method, 'roundingMethod')
    method.roundingMethod = 'sign';
  end
  if ~isfield(method, 'kernelNum')
    method.kernelNum = 0;
  end
  if method.kernelNum > 0
    if ~isfield(method, 'kernelEqualVariance')
      method.kernelEqualVariance = false;
    end
    if ~isfield(method, 'kernelEqualLength')
      method.kernelEqualLength = false;
    end
  end

  % start training timer
  timerTrain = tic;
  timeDebug = 0;

 
  % define fundamental variables
  N1 = length(dataset.indexTrain);
  X1 = dataset.traindata(dataset.indexTrain, :);
  N2 = length(dataset.indexTest);
  X2 = dataset.traindata(dataset.indexTest, :);
  Q = codeLength;
  D = size(X1, 2);


  % NT used for linear regression
  lambdaNorm = method.lambda * N1 / D;
  %NT = learnNT(X1, lambdaNorm);
  NT = (X1' * X1 + lambdaNorm * eye(size(X1, 2))) \ X1';

  % initialize U

  switch method.initMethod
    case 'PCA'
      if D > Q
			U0 = PCA(X1, Q);
      else
        idx = mod([0: Q - 1], D) + 1;
        U0 = X1(:, idx);
      end
    case 'random'
      U0 = randn(N1, Q);
  end

  % learn optimal U
  switch method.learnImpl
      
    case 'stochastic'
      betaNorm = method.beta / Q;
      calcS = @(Sc) calcNeighbor(dataset, dataset.indexTrain, dataset.indexTrain(Sc));
      U1 = learnUStochastic(U0, calcS, betaNorm, Q, method.maxIter);
      
  end

  % plot convergence curve of objective function

  % normalize U1 to have unit length
  if method.normalize
    lenU = sqrt(sum(U1 .^ 2, 2));
    U1 = bsxfun(@rdivide, U1, lenU);
  end

 



  % obtain the binary codes of training data
  W = learnW(U1, NT);
  B1 = rounding(U1, method.roundingMethod);
  t1 = toc(timerTrain);

  % out-of-sample extension
  timerTest = tic;
  U2 = X2 * W;
  B2 = rounding(U2, method.roundingMethod);
  t2 = toc(timerTest) / N2;

end


function NT = learnNT (X, lambda)

  NT = (X' * X + lambda * eye(size(X, 2))) \ X';

end

function W = learnW (U, NT)

  W = NT * U;

end
