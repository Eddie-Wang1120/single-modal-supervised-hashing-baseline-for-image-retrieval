function [G, WF, B] = FSDH(W,P,X,Y,B,gmap,Fmap,tol,maxItr,debug)
% ---------- Argument defaults ----------
if ~exist('debug','var') || isempty(debug)
    debug=1;
end
if ~exist('tol','var') || isempty(tol)
    tol=1e-5;
end
if ~exist('maxItr','var') || isempty(maxItr)
    maxItr=1000;
end
nu = Fmap.nu;
% ---------- End ----------
G.W = W;
WF.W = P; WF.nu = nu;
i = 0; 
while i < maxItr
    i=i+1;  
    
    if debug,fprintf('Iteration  %03d: ',i);end
    
    % B-step
  
        F = X*P;
   
    switch gmap.loss
        case 'L2'
            B = sign(nu*F + Y*W);  
    end
    
    % G-step
    switch gmap.loss
    case 'L2'
        [W, ~, ~] = RRC(Y, B, gmap.lambda);
    end
    G.W = W;
    % F-step 
    P0 = P;
    
    [P, ~, ~] = RRC(X, B, Fmap.lambda);  
    
    WF.W = P; WF.nu = nu;
    
    bias = norm(B-X*P,'fro');
    
    if bias < tol*norm(B,'fro')
            break;
    end 
    
    
    if norm(P-P0,'fro') < tol * norm(P0)
        break;
    end
    
    
end
