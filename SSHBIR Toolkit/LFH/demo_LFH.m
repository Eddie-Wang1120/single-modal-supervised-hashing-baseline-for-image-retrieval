function [MAP,TIME] = demo_LFH(exp_data, bits)

%Load Dataset
traindata = exp_data.traindata;
traingnd = exp_data.traingnd;
testgnd = exp_data.testgnd;

%Preprocessing
dataset.traindata=normZeroMean(traindata);
dataset.traindata=normEqualVariance(traindata);

%Data Partition
num_dataset=size(traindata,1);
num_test=1000;
perm=randperm(num_dataset);
dataset.traindata=dataset.traindata(perm,:);

dataset.indexTrain=1:num_dataset-num_test;
dataset.indexTest=num_dataset-num_test+1:num_dataset;

%For multi-label cases, it should be 'tag'. (refer to calcNeighbor.m)
dataset.neighborType='traingnd';
dataset.traingnd=traingnd(perm);

%Parameter of LFH
method.learnImpl='stochastic';
method.maxIter=50;
method.beta=3e1;
method.lambda=1e0;

%Length of binary-codes
codeLength=bits;

time = tic();
[B1,B2,t1,t2] = LFH(dataset,method,codeLength);
time=toc(time);
TIME = time;

%Evaluation
dataset.neighborTest=calcNeighbor2(dataset,dataset.indexTest,dataset.indexTrain);
[distH, orderH] = calcHammingRank(B1, B2);

[MAP, succRate] = calcMAP2(orderH, dataset.neighborTest);



end