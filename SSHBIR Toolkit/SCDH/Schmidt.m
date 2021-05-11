function B= Schmidt(A)
% input:A=[a1,a2,...ap]
% output:经过正交化处理的矩阵B

[n,p]=size(A);
B=zeros(n,p);

%根据正交化的计算公式
B(:,1) = A(:,1);
if p>=2
    for i=2:p
        B(:,i)=A(:,i);  %bi的初始值
        for j=1:i-1
            B(:,i)=B(:,i)-(B(:,j)'*A(:,i))/(B(:,j)'*B(:,j))*B(:,j);
        end
        %单位化
        B(:,i)=B(:,i)/norm(B(:,i));
    end
end

