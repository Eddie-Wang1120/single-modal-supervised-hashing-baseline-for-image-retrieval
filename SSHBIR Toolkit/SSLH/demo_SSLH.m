function [MAP,TIME] = demo_SSLH(exp_data, bits)
nbits = bits;

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

% I_query = I_te;
% L_query = L_te;
% 
% I_re = I_db;
% L_re = L_db;

X=I_tr;
Y=L_tr;

t1 = tic();

% 哈希码长度
len = nbits;
% 训练集数据和标签种类数
[n, c] = size(Y);
% X的特征维度
[~, doI] = size(X);
% alpha=3;
lamda = 1e-4;
cita=1e-4;
v = 1e-4;
gama=1e-4;

delt=2;

I = eye(len);
% 初始化B
A = rand(n,len);
B = sign(A-0.5);
% 初始化D
D = eye(n);

% 求初始W
W = (B'*D*B + lamda*I)\B'*D*Y;
% 求初始P
P_temp = pinv(X'*X+ lamda*eye(doI))*X';
P = P_temp*B;
U=P;
% W2=rand(c,len)-0.5;

dl=6;
% loss=zeros(dl,1);
for loops=1:dl
    
    
    % B-step
%     difference = 1;
    M = D*Y*W'+ v*X*P;

    
    for i=1:10
        B=zeros(size(B));
        for m=1:len
            temp = [1:m-1, m+1:len];
            Btemp = B(:,temp); % n*(len-1)
            Wtemp = W(temp,:); % (len-1)*c
            Utemp = U(temp,:);
            w = W(m,:); % 1*c
            q = M(:,m); % n*1
            uk = U(m,:);
            b = (q - D*Btemp*Wtemp*w'-gama*Btemp*Utemp*uk'-cita*Btemp*(ones(1,len-1)')); % n*1
            b(b>=0) = 1;
            b(b<0) = -1;
            
            B(:,m) = b;
        end
    end

% W-step
W = (B'*D*B + lamda*eye(len))\B'*D*Y;

P = P_temp*B;


U=(X'*B)*pinv(B'*B+ lamda*eye(len));


% D-step
E=Y-B*W;
for ii=1:n
    D(ii,ii) = exp(-norm(E(ii,:),'fro')^2/delt^2);
end
end

B_te=I_query*P>0;
B_re=I_re*P>0;

t1=toc(t1);
TIME = t1;

[~,MAP]=EvaPreK(5000,L_re,L_query,B_te,B_re);

end