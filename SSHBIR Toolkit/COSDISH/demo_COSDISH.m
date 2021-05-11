%This is the code for paper 'Column Sampling based Discrete Supervised Hashing, AAAI'16'
%Code Author: Wang-Cheng Kang (Kwc.Oliver@gmail.com)
function [MAP,TIME] = demo_COSDISH(exp_data, bits)

%Load Dataset
%the dataset should contain two varialbes: feature matrix 'X' (each row represents a point) and
%semantic information: 'label'(for single label) or 'tag'(for multi-label)
% load('../CIFAR-10.mat');

traindata = exp_data.traindata;
traingnd = exp_data.traingnd;

%Preprocessing
dataset.traindata=normZeroMean(traindata);
dataset.traindata=normEqualVariance(traindata);

%Data Partition
num_dataset=size(traindata,1);
num_test=1000;
perm=randperm(num_dataset); %random split
dataset.traindata=dataset.traindata(perm,:);

dataset.indexTrain=1:num_dataset-num_test;
dataset.indexTest=num_dataset-num_test+1:num_dataset;

%For multi-label cases, it should be 'tag'. (refer to calcNeighbor.m)
%single
dataset.neighborType='traingnd';
dataset.traingnd=traingnd(perm);

%multi
%dataset.neighborType='tag';
%dataset.tag=tag(perm,:);



%Length of binary-codes
codeLength=bits;

t1=tic();
[B1,B2] = COSDISH(dataset,codeLength); 
t1=toc(t1);


TIME = t1;

%Evaluation
dataset.neighborTest=calcNeighbor(dataset,dataset.indexTest,dataset.indexTrain);
[distH, orderH] = calcHammingRank(B1, B2);
[MAP, succRate] = calcMAP(orderH, dataset.neighborTest);




end