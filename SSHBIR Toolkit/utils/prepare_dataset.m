function prepare_dataset(dataset)
load(['./datasets/',dataset]);

traindata = normalize(double(traindata));
testdata  = normalize(double(testdata));

cateTrainTest = bsxfun(@eq, traingnd, testgnd');

save(['testbed/',dataset],'traindata','testdata','traingnd','testgnd','cateTrainTest', '-v7.3');

clear;


