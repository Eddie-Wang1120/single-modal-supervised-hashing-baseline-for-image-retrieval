function [MAP,TIME] = demo_FSDH(exp_data, bits)


traindata = exp_data.traindata;
traingnd = exp_data.traingnd;
testdata = exp_data.testdata;
testgnd = exp_data.testgnd;
cateTrainTest = exp_data.cateTrainTest;

traindata = double(traindata);
testdata = double(testdata);

if sum(traingnd == 0)
    traingnd = traingnd + 1;
    testgnd = testgnd + 1;
end

Ntrain = size(traindata,1);
X = traindata;

n_anchors = 1000;
anchor = X(randsample(Ntrain, n_anchors),:);


sigma = 0.4; 
PhiX = exp(-sqdist(X,anchor)/(2*sigma*sigma));
PhiX = [PhiX, ones(Ntrain,1)];

Phi_testdata = exp(-sqdist(testdata,anchor)/(2*sigma*sigma)); clear testdata
Phi_testdata = [Phi_testdata, ones(size(Phi_testdata,1),1)];
Phi_traindata = exp(-sqdist(traindata,anchor)/(2*sigma*sigma)); clear traindata;
Phi_traindata = [Phi_traindata, ones(size(Phi_traindata,1),1)];


maxItr = 5;
gmap.lambda = 1; gmap.loss = 'L2';
Fmap.type = 'RBF';
Fmap.nu = 1e-5;
Fmap.lambda = 1e-2;

time=tic();

%% run algo
nbits = bits;

randn('seed',3);

Zinit=sign(randn(Ntrain,nbits)); 

B = Zinit;
X = PhiX;

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

AsymDist = 0;

if AsymDist 
    H = H > 0;
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












