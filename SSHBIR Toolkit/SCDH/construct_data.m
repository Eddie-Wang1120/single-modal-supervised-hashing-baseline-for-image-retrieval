function exp_data = construct_data(theexp_data)

traindata = theexp_data.traindata;
traingnd = theexp_data.traingnd;
testdata = theexp_data.testdata;
testgnd = theexp_data.testgnd;

traingnd = traingnd+1;
testgnd = testgnd+1;
train_label = zeros(59000,10);
test_label = zeros(1000,10);
for i=1:length(train_label)
    train_label(i,traingnd(i)) = 1;
end
for i=1:length(test_label)
    test_label(i,testgnd(i)) = 1;
end

XX = [traindata; testdata];
XX = double(XX);
exp_data.train_data = traindata;
exp_data.test_data = testdata;
exp_data.db_data = XX;
exp_data.train_label = train_label;
exp_data.test_label = test_label;
