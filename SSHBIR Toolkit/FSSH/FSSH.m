function [B_train , B_test ] = FSSH (data_our, nbits)


param_mu = 10000;  param_theta=0.01;
Iter_num = 5;



N1 = length(data_our.indexTrain);
X = data_our.X(data_our.indexTrain, :);
X2 = data_our.X(data_our.indexTest, :);
y=data_our.label(data_our.indexTrain, :);

if isvector(y)
    Y = sparse(1:length(y), double(y), 1); Y = full(Y);
else
    Y = y;
end

% SY=S*Y; YtY=Y'*Y;
load('pre_computed.mat');
A=X'*SY;
XtX=X'*X;  invYY=inv(Y'*Y);  invXX=inv(X'*X);


% %%%%%% B - initialize %%%%%%
B=randn(N1,nbits)>0; B=B*2-1;
% %%%%%% G - initialize %%%%%%
G=randn(size(Y,2),nbits);  %G=Y'*B;
% %%%%%% W - initialize %%%%%%
W=randn(size(X,2),nbits);  %W=X'*B;

%-----------------------------------------------------training---------------------------------
for iter=1:Iter_num
    
    
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    % ----------------------- B-step -----------------------%
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    B=sign(param_mu*Y*G+param_theta*X*W);

    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    % ----------------------- G-step -----------------------%
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    G=invYY*(param_mu*Y'*B+A'*W)*inv(param_mu*eye(nbits)+W'*XtX*W);

    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    % ----------------------- W-step -----------------------%
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    W=invXX*(param_theta*X'*B+A*G)*inv(param_theta*eye(nbits)+G'*YtY*G);

end


B_train=B>0;

%-----------------------------------------------------Out-of-Sample-------------------------------------
 NT = (X' * X + 1 * eye(size(X, 2))) \ X';
 W=NT*B;
 B_test=X2*W>0;


end



