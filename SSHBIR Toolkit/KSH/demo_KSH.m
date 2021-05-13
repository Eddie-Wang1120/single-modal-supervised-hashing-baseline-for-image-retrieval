function [MAP,TIME] = demo_KSH(exp_data, bits)
traindata = exp_data.traindata;
traingnd = exp_data.traingnd;
testdata = exp_data.testdata;
testgnd = exp_data.testgnd;

[n,d] = size(traindata);
tn = size(testdata,1);
range = 10; % number of returned neighbors
m = 300;    % number of anchors
r = bits;     % number of hash bits
trn = 2000; % number of labeled training samples
load label_index_2k; % indexes of labeled samples
load sample_300;     % indexes of anchors



%% Kernel-Based Supervised Hashing (KSH)
% kernel computing 
t1 = tic();
anchor = traindata(sample',:);
KTrain = kshsqdist(traindata',anchor');
sigma = mean(mean(KTrain,2));
KTrain = exp(-KTrain/(2*sigma));
mvec = mean(KTrain);
KTrain = KTrain-repmat(mvec,n,1);

% pairwise label matrix
trngnd = traingnd(label_index');
temp = repmat(trngnd,1,trn)-repmat(trngnd',trn,1);
S0 = -ones(trn,trn);
tep = find(temp == 0);
S0(tep) = 1;
clear temp;
clear tep;
S = r*S0;

% projection optimization
KK = KTrain(label_index',:);
RM = KK'*KK; 
A1 = zeros(m,r);
flag = zeros(1,r);
for rr = 1:r
    if rr > 1
        S = S-y*y';
    end
    
    LM = KK'*S*KK;
    [U,V] = eig(LM,RM);
    eigenvalue = diag(V)';
    [eigenvalue,order] = sort(eigenvalue,'descend');
    A1(:,rr) = U(:,order(1));
    tep = A1(:,rr)'*RM*A1(:,rr);
    A1(:,rr) = sqrt(trn/tep)*A1(:,rr);
    clear U;    
    clear V;
    clear eigenvalue; 
    clear order; 
    clear tep;  
    
    [get_vec, cost] = OptProjectionFast(KK, S, A1(:,rr), 500);
    y = double(KK*A1(:,rr)>0);
    ind = find(y <= 0);
    y(ind) = -1;
    clear ind;
    y1 = double(KK*get_vec>0);
    ind = find(y1 <= 0);
    y1(ind) = -1;
    clear ind;
    if y1'*S*y1 > y'*S*y
        flag(rr) = 1;
        A1(:,rr) = get_vec;
        y = y1;
    end
end

% encoding
Y = single(A1'*KTrain' > 0);
tep = find(Y<=0);
Y(tep) = -1;

t1=toc(t1);
TIME = t1;
save ksh_48 Y A1 anchor mvec sigma;
clear tep; 
clear get_vec;
clear y;
clear y1;
clear S;
clear KK;
clear LM;
clear RM;


load ksh_48;
%% test
% encoding
KTest = kshsqdist(testdata',anchor');
KTest = exp(-KTest/(2*sigma));
KTest = KTest-repmat(mvec,tn,1);
tY = single(A1'*KTest' > 0);
tep = find(tY<=0);
tY(tep) = -1;
clear tep;

% search
sim = Y'*tY;
[temp,order] = sort(sim,1,'descend');
MAP = cat_apcal(traingnd,testgnd,order);

end