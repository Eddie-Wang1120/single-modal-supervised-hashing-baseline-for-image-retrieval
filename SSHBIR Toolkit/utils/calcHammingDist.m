
function D = calcHammingDist (B1, B2)

  if ~exist('B2', 'var')
    B2 = B1;
  end

  P1 = sign(B1 - 0.5);
  P2 = sign(B2 - 0.5);

  R = size(P1, 2);
  
  a=size(B1,1);b=size(B2,1);
  D = round((R - P1 * P2') / 2);

end
