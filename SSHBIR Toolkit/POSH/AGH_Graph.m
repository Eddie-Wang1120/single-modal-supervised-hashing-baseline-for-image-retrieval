%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Z, M] = AGH_Graph(X, Anchor, s)
[n,dim] = size(X);
m = size(Anchor,1);

Z = zeros(n,m);
Dis = sqdist_sdh(X,Anchor);
clear X;
clear Anchor;

val = zeros(n,s);
pos = val;
for i = 1:s
    [val(:,i),pos(:,i)] = min(Dis,[],2);
    tep = (pos(:,i)-1)*n+[1:n]';
    Dis(tep) = 1e60; 
end
clear Dis tep;

sigma = mean(val(:,s).^0.5);

val = exp(-val/(1/1*sigma^2));
val = repmat(sum(val,2).^-1,1,s).*val; %% normalize
tep = (pos-1)*n+repmat([1:n]',1,s);
Z([tep]) = [val];
Z = sparse(Z);
clear tep val pos;

M = diag(sum(Z).^-1);
end
