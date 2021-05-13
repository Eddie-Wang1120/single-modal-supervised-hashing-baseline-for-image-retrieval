function [MAP,PRE,REC] = train_FSSH( exp_data,nbits )


%% Preprocessing


X = exp_data.traindata;
n_anchors = 1000; sigma = 0.4; 
anchor = X(randsample( size(exp_data.traindata,1) , n_anchors),:);
Phi_testdata = exp(-sqdist_sdh(exp_data.testdata,anchor)/(2*sigma*sigma));
Phi_traindata = exp(-sqdist_sdh(exp_data.traindata,anchor)/(2*sigma*sigma));


X=[Phi_traindata ; Phi_testdata];
data_our.indexTrain=1:size(exp_data.traindata,1);
data_our.indexTest=size(exp_data.traindata,1)+1:size(exp_data.traindata,1) + size(exp_data.testdata,1);
data_our.X=normZeroMean(X);
data_our.X=normEqualVariance(X);


data_our.label=exp_data.traingnd;


%% Training

[U_logical_trn,U_logical_tst]= FSSH(data_our,nbits);



%% Evaluation

B_compact_trn = compactbit(U_logical_trn);
B_compact_tst = compactbit(U_logical_tst);
DHamm = hammingDist(B_compact_tst, B_compact_trn);

hammRadius = 2;

% hash lookup: precision and reall
Ret = (DHamm <= hammRadius+0.00001);
[PRE, REC] = evaluate_macro(exp_data.WTT, Ret);


[~, orderH] = sort(DHamm, 2);
WtrueTestTraining = exp_data.WTT; 
MAP = calcMAP(orderH, WtrueTestTraining);

end

