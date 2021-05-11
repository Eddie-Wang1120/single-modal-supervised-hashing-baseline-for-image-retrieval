%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This is the code for paper 'fast scalable supervised hashing'.
% This is the code for variant FSSH_ts.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [MAP,TIME] = demo_FSSH(exp_data, bits)

nbits_set = [bits];

%% Load dataset
traindata = exp_data.traindata;
traingnd = exp_data.traingnd;
testdata = exp_data.testdata;
testgnd = exp_data.testgnd;


train_data_cifar = traindata';
test_data_cifar = testdata';
train_label_cifar = traingnd + 1;
test_label_cifar = testgnd + 1;



exp_data.traingnd=train_label_cifar;
exp_data.testgnd=test_label_cifar;
cateTrainTest = bsxfun(@eq, train_label_cifar, test_label_cifar');
exp_data.WTT=cateTrainTest';
exp_data.traindata = double(train_data_cifar');
exp_data.testdata = double(test_data_cifar');



for ii=1:length(nbits_set)
    
    nbits=nbits_set(ii);
    
    time = tic();
    % FSSH_ts
    [MAP,PRE,REC] =train_FSSH(exp_data, nbits);
    time=toc(time);
    TIME = time;
    
    
end
end

