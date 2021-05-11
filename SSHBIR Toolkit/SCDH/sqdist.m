function [ dist ] = sqdist( x1,x2 )
%SQDIST 此处显示有关此函数的摘要
%   此处显示详细说明
a = sum(x1.^2,2);
b = sum(x2.^2,2);
dist = repmat(a,1,size(x2,1)) + repmat(b',size(x1,1),1) - 2*x1*x2';
end

