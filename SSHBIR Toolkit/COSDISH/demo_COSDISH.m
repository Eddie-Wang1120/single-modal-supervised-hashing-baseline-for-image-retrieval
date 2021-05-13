
function [MAP,TIME] = demo_COSDISH(exp_data, bits)

traindata = exp_data.traindata;
traingnd = exp_data.traingnd;

dataset.traindata=normZeroMean(traindata);
dataset.traindata=normEqualVariance(traindata);

num_dataset=size(traindata,1);
num_test=1000;
perm=randperm(num_dataset); 
dataset.traindata=dataset.traindata(perm,:);

dataset.indexTrain=1:num_dataset-num_test;
dataset.indexTest=num_dataset-num_test+1:num_dataset;

dataset.neighborType='traingnd';
dataset.traingnd=traingnd(perm);


codeLength=bits;

t1=tic();
[B1,B2] = COSDISH(dataset,codeLength); 
t1=toc(t1);


TIME = t1;

dataset.neighborTest=calcNeighbor2(dataset,dataset.indexTest,dataset.indexTrain);
[distH, orderH] = calcHammingRank(B1, B2);
[MAP, succRate] = calcMAP2(orderH, dataset.neighborTest);




end