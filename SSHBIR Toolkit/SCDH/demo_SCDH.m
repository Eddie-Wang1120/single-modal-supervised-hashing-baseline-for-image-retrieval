function [MAP,TIME] = demo_SCDH(exp_data, bits)
%% 加载数据
db_data = construct_data(exp_data);

%% 超参数
alpha = 0.1;
beta = 1;
lambda = 10;

%% kernel computing
[Ntrain,dim] = size(db_data.train_data);
[Ntest,~] = size(db_data.test_data);

db_data.train_data = normalize(db_data.train_data);
db_data.test_data = normalize(db_data.test_data);
% get anchors
n_anchors = min(2000,Ntrain);
rand('seed',0);
anchor = db_data.train_data(randsample(Ntrain, n_anchors),:);
sigma = 0.4; % for normalized data
Phi_testdata = exp(-sqrt(sqdist(db_data.test_data,anchor))/(2*sigma*sigma));
Phi_testdata = [Phi_testdata, ones(Ntest,1)];
Phi_traindata = exp(-sqrt(sqdist(db_data.train_data,anchor))/(2*sigma*sigma));
Phi_traindata = [Phi_traindata, ones(Ntrain,1)];

db_data.train_data = Phi_traindata;
db_data.test_data = Phi_testdata;

t1 = tic();

%% training
params.X = db_data.train_data;
params.test_data = db_data.test_data;
params.epchos = 10;
params.train_label = db_data.train_label;
params.test_label = db_data.test_label;

params.b = bits;
params.alpha = alpha;
params.beta = beta;
params.lambda = lambda;

params = initialize(params);
params = solve(params);

t1=toc(t1);
TIME = t1;

%% testing
B_tst = hash_func(params.P,params.test_data);
B_trn = params.B;
B_trn(B_trn<0) = 0;
B_trn = compactbit(B_trn);
Dhamm = hammingDist(B_tst, B_trn);
[~, HammingRank]=sort((Dhamm'),1);
[MAP] = cal_map(params.train_label,params.test_label,HammingRank);
end