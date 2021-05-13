function [MAP,TIME] = demo_SDH(exp_data, bits)



traindata = exp_data.traindata;
traingnd = exp_data.traingnd;
testdata = exp_data.testdata;
testgnd = exp_data.testgnd;
cateTrainTest = exp_data.cateTrainTest;

traindata = double(traindata); %traindata -> traindataset 59000 records
testdata = double(testdata);% testdata -> testdataset 1000 records

if sum(traingnd == 0)
    traingnd = traingnd + 1;
    testgnd = testgnd + 1;
end

Ntrain = size(traindata,1); 
X = traindata;
label = double(traingnd);


n_anchors = 1000;
anchor = X(randsample(Ntrain, n_anchors),:);


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

t1 = tic();

%% run algo
nbits = bits;

% Init Z
randn('seed',3);

Zinit=sign(randn(Ntrain,nbits));


debug = 0;
[~, F, H] = SDH(PhiX,label,Zinit,gmap,Fmap,[],maxItr,debug);

t1=toc(t1);
TIME = t1;


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












