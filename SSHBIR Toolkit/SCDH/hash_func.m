function [ data ] = hash_func( P,data )
%HASH �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
data = sgn(data*P);
data(data<0) = 0;
data = compactbit(data);
end

