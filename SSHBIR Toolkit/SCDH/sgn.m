function [ B ] = sgn( A )
%SGN 此处显示有关此函数的摘要
%   此处显示详细说明
    B = ones(size(A));
    q = A<0;
    B(q) = -1;
end

