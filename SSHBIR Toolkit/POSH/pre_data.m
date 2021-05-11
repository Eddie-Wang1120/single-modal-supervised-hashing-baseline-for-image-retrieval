function [trfea, ttfea, cateTrainTest] = pre_data(traingnd,testgnd,traindata,testdata, feature)
traingnd = double(traingnd);
testgnd = double(testgnd);
traindata = double(traindata);
testdata = double(testdata);

% label correction
u_label = unique(traingnd);
Y_tr = bsxfun(@eq, traingnd, u_label');
[~,traingnd] = max(Y_tr,[],2);
Y_te = bsxfun(@eq, testgnd, u_label');
[~,testgnd] = max(Y_te,[],2);
clear Y_tr Y_te

Ntrain = size(traindata,1);
n_test = size(testdata,1);

if ~exist('cateTrainTest', 'var')
    cateTrainTest = bsxfun(@eq, traingnd, testgnd');
end

switch feature
    case 'raw'
        feaTrain = traindata;
        feaTest = testdata;
    case 'RBF'        
        % get anchors
        n_anchors = 1000;
        rand('seed',100);
        anchor = traindata(randperm(Ntrain, n_anchors),:);
        % sigma = 0.4;
        Dis = EuDist2(traindata,anchor,0);
        sigma = mean(min(Dis,[],2).^0.5);
        clear Dis
        trfea = exp(-sqdist(traindata,anchor)/(2*sigma*sigma));
        ttfea = exp(-sqdist(testdata,anchor)/(2*sigma*sigma)); 
        m = mean(trfea);
        trfea = bsxfun(@minus, trfea, m);
        ttfea = bsxfun(@minus, ttfea, m);
        
end

m = mean(trfea);
trfea = bsxfun(@minus, trfea, m);
ttfea = bsxfun(@minus, ttfea, m);

clear traindata testdata