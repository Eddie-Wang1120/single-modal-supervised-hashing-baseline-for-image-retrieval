
 function[Precision,MAP] =RSLH(len,I_tr,I_query,I_re,L_tr,L_query,L_re)

I_tr=I_tr';
L_tr=L_tr';
Y=L_tr;


[c,n] = size(Y);

[doI,~] = size(I_tr);

lamda =1e-6;
alpha=7;

beta =1e-4;
gama=1e-2;

W=rand(len,c)-0.5;

B=sign(rand(len,n)-0.5);

H=sign(rand(len,n)-0.5);



dl=6;
S=L_tr'*L_tr>0;
S=2*S-1;


rr=doI;
[U, ~, V] = svd(I_tr*S);
R2=U*V(:,1:rr)'; 

S2=S*R2';

% save('S.mat','S');
% load('S.mat');

P_temp=(I_tr*I_tr'+lamda*diag(ones(doI,1)))\(I_tr);
P=P_temp*B';


for loops=1:dl 

Q1=W*Y+beta*H*R2'*S2'+gama*H;     

[U, ~, V] = svd(Q1);
B=U*V(:,1:len)'; 

Q2=W*Y+beta*B*S+gama*B;   

H=sign(Q2);

W=(alpha*H*Y'+B*Y')/(alpha*Y*Y'+eye(c));

end

P = P_temp*H';
B_te=I_query*P>0;
B_re=I_re*P>0;
% save('H_RSLH.mat','H');


[Precision,MAP]=EvaPreK(5000,L_re,L_query,B_te,B_re);

end