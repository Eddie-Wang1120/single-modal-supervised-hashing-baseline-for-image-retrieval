function [MAP,TIME] = demo_POSH(exp_data, bits)
nbits_set = bits;

traindata = exp_data.traindata;
traingnd = exp_data.traingnd;
testdata = exp_data.testdata;
testgnd = exp_data.testgnd;


% label correction
u_label = unique(traingnd);
Y_tr = bsxfun(@eq, traingnd, u_label');
[~,traingnd] = max(Y_tr,[],2);
Y_te = bsxfun(@eq, testgnd, u_label');
[~,testgnd] = max(Y_te,[],2);
clear Y_tr Y_te

exp_data.traingnd = traingnd;
exp_data.testgnd = testgnd;
WtrueTestTraining = bsxfun(@eq, testgnd, traingnd');
exp_data.traindata = double(traindata');
exp_data.testdata = double(testdata');

n_anchors = 1000;
anchor = traindata(randperm(length(traingnd), n_anchors),:);
Dis = EuDist2(traindata,anchor,0);
sigma = mean(min(Dis,[],2).^0.5);
% sigma = 0.4; 
clear Dis
feaTrain = exp(-sqdist_sdh(traindata,anchor)/(2*sigma*sigma));
feaTest = exp(-sqdist_sdh(testdata,anchor)/(2*sigma*sigma)); 
m = mean(feaTrain);
feaTrain = bsxfun(@minus, feaTrain, m);
feaTest = bsxfun(@minus, feaTest, m);

t1 = tic();

    nbits = nbits_set;

    X=double([feaTrain ; feaTest]);
    data_our.indexTrain=1:size(exp_data.traindata,2);
    data_our.indexTest=size(exp_data.traindata,2)+1:size(exp_data.traindata,2) + size(exp_data.testdata,2);
    data_our.X=normEqualVariance(X)';
    data_our.label=exp_data.traingnd;

    pars.nbits    = nbits;
    pars.Iter_num = 5;
    pars.eta      = 10; 
    pars.lambda   =  .01;

    [B_trn,B_tst]= POSH_fun(data_our,pars);
    
    t1=toc(t1);
TIME = t1;

   %% Evaluation
    B1 = compactbit(B_trn);
    B2 = compactbit(B_tst);
    DHamm = hammingDist(B2, B1);
    [~, orderH] = sort(DHamm, 2);
    MAP = calcMAP(orderH, WtrueTestTraining);
end
