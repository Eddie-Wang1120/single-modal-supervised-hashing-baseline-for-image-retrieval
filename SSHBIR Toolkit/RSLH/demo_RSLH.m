function [MAP,TIME] = demo_RSLH(exp_data, bits)

len = bits;

traindata = exp_data.traindata;
traingnd = exp_data.traingnd;
testdata = exp_data.testdata;
testgnd = exp_data.testgnd;

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


IX = feaTrain;
Ntrain = size(IX,1);
LX_tr = zeros(Ntrain,10);

for i = 1:Ntrain
    LX_tr(i, traingnd(i,1)+1) = 1;
end

I_query = feaTest;
Ntest = size(testgnd,1);
L_query = zeros(Ntest,10);

for i = 1:Ntest
    L_query(i, testgnd(i,1)+1) = 1;
end


n_tr = 5000;

random = randperm(Ntrain,n_tr);

% Use all the training data
I_tr = IX(random,:);  % X = traindataset
L_tr = LX_tr(random,:);


IX(random,:) = [];
LX_tr(random,:) = [];

I_re = IX;
L_re = LX_tr;

t1 = tic();

 [~,MAP]=RSLH(len,I_tr,I_query,I_re,L_tr,L_query,L_re);

t1=toc(t1);
TIME = t1;

end