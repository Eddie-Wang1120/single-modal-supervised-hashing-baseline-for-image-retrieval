function [ B ] = sgn( A )
%SGN �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    B = ones(size(A));
    q = A<0;
    B(q) = -1;
end

