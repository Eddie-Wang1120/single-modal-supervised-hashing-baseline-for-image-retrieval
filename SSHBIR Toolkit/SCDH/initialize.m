function [ params ] = initialize(params)
%%  ³õÊ¼»¯ P,Z,B
[n,d] = size(params.X);
rand('seed',7);
params.P = rand(d,params.b);
randn('seed',7);
params.Z = randn(n,params.b);
params.B = sgn(params.Z);
end

