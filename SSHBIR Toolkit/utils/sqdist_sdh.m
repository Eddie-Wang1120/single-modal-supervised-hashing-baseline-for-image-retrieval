
function sqd = sqdist_sdh(X,Y,w)

if nargin==1	
  x = sum(X.^2,2); sqd = max(bsxfun(@plus,x,bsxfun(@plus,x',-2*X*X')),0);
  return
end

if ~exist('Y','var') | isempty(Y) Y = X; eqXY = 1; else eqXY=0; end;
  
if exist('w','var') & ~isempty(w)
  h = sqrt(w(:)'); X = bsxfun(@times,X,h);
  if eqXY==1 Y = X; else Y = bsxfun(@times,Y,h); end;
end

x = sum(X.^2,2);
if eqXY==1 y = x'; else y = sum(Y.^2,2)'; end;
sqd = max(bsxfun(@plus,x,bsxfun(@plus,y,-2*X*Y')),0);

