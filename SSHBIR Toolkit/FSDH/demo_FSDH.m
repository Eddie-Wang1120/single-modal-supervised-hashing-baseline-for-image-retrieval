function [MAP,TIME] = demo_FSDH(exp_data, bits)


%addpath [liblinear-1.91/windows/] % for hinge loss



% dataset = 'cifar_10_gist.mat';

% prepare_dataset(dataset);
% addpath('./testbed/');
% load(['./testbed/',dataset]);

traindata = exp_data.traindata;
traingnd = exp_data.traingnd;
testdata = exp_data.testdata;
testgnd = exp_data.testgnd;
cateTrainTest = exp_data.cateTrainTest;

traindata = double(traindata); %traindata -> traindataset 59000 records
testdata = double(testdata);% testdata -> testdataset 1000 records

if sum(traingnd == 0)
    traingnd = traingnd + 1;  % 59000 train label records
    testgnd = testgnd + 1; % 1000 test label records
end

%Ntarin = row of train
Ntrain = size(traindata,1); %Ntrain -> row of train
% Use all the training data
X = traindata;  % X = traindataset

% get anchors
n_anchors = 1000; % number of anchors
% rand('seed',1);
% randomly produce anchors
anchor = X(randsample(Ntrain, n_anchors),:); % assign X random(not different) row to anchor 


% % determin rbf width sigma
% Dis = EuDist2(X,anchor,0);
% % sigma = mean(mean(Dis)).^0.5;
% sigma = mean(min(Dis,[],2).^0.5);
% clear Dis
sigma = 0.4; % for normalized data
PhiX = exp(-sqdist(X,anchor)/(2*sigma*sigma)); % compute RBF(sqdist between anchor and traindata)
PhiX = [PhiX, ones(Ntrain,1)];

%φ(x) = [exp(||x − a1||2/σ), ··· , exp(||x−am||2/σ)]
% RBF
Phi_testdata = exp(-sqdist(testdata,anchor)/(2*sigma*sigma)); clear testdata
Phi_testdata = [Phi_testdata, ones(size(Phi_testdata,1),1)];
Phi_traindata = exp(-sqdist(traindata,anchor)/(2*sigma*sigma)); clear traindata;
Phi_traindata = [Phi_traindata, ones(size(Phi_traindata,1),1)];


% learn G and F / set function and parameter od G step and F step
maxItr = 5; % maximum Iteration
gmap.lambda = 1; gmap.loss = 'L2'; % loss function
Fmap.type = 'RBF'; % function type
Fmap.nu = 1e-5; %  penalty parm for F term
Fmap.lambda = 1e-2;

time=tic();

%% run algo
nbits = bits;

% Init Z
randn('seed',3);

%init a random matrix of row od train, column of bits
% Initialize b
Zinit=sign(randn(Ntrain,nbits)); % return a matrix where has 59000 row and nbits column all truns out 1 and -1

B = Zinit;
X = PhiX;

% Initialize Y
Y = zeros(Ntrain,10);
for i = 1:Ntrain
    Y(i,traingnd(i,1)) = 1;
end

[W, ~, ~] = RRC(Y, B, gmap.lambda);

[P, ~, ~] = RRC(X, B, Fmap.lambda);
% 
debug = 0;
[~, F, H] = FSDH(W,P,X,Y,B,gmap,Fmap,[],maxItr,debug);

time=toc(time);
TIME = time;

%% evaluation
% display('Evaluation...');

AsymDist = 0; % Use asymmetric hashing or not

if AsymDist 
    H = H > 0; % directly use the learned bits for training data
else
    H = Phi_traindata*F.W > 0;
end

tH = Phi_testdata*F.W > 0;

hammRadius = 2;



B = compactbit(H);
tB = compactbit(tH);


hammTrainTest = hammingDist(tB, B)';
% hash lookup: precision and reall
Ret = (hammTrainTest <= hammRadius+0.00001);
[PRE, REC] = evaluate_macro(cateTrainTest, Ret);

% hamming ranking: MAP
[~, HammingRank]=sort(hammTrainTest,1);
MAP = cat_apcal(traingnd,testgnd,HammingRank);
end












