function [W, labels, E] = RRC(tr_dat, tr_labels, lambda)
%G-step : tr_dat -> Zinitt , tr_labels -> label(traindata) matrix , gmap.lambda = 1
%F-step : tr_dat -> Phix , tr_labels -> Zinit , Fmap.lambda = le-2
%projection matrix computing
%regularized least squares problem
if size(tr_dat,1) < size(tr_dat,2)
    Proj_M = tr_dat'/(tr_dat*tr_dat'+lambda*eye(length(tr_labels))); 
    %Proj_M = G-step : (B*B'+lambda*I)^-1*B
    %(eye(length(tr_labels))->Identity Matrix
    
    %Proj_M = F-step : (o(x)*o(x)')^-1*o(x)*B'
    
else
    Proj_M = (tr_dat'*tr_dat+lambda*eye(size(tr_dat,2)))\tr_dat';
end
if isvector(tr_labels)
    Y = sparse(1:length(tr_labels), double(tr_labels), 1); Y = full(Y);
else
    Y = tr_labels;
end
W = Proj_M * Y; % W = Proj_M * Y
%-------------------------------------------------------------------------
%testing
if nargout > 1
    [~,labels] = max(tr_dat*W, [], 2);
end
if nargout > 2
    E = sum(sum((Y - tr_dat*W).^2)) + lambda*sum(sum(W.^2));
end
