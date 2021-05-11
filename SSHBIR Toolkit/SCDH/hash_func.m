function [ data ] = hash_func( P,data )
%HASH 此处显示有关此函数的摘要
%   此处显示详细说明
data = sgn(data*P);
data(data<0) = 0;
data = compactbit(data);
end

