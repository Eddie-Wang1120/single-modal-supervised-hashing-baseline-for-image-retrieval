function B = rounding (U, method)

  switch method
    case 'mean'
      md = mean(U);
      B = bsxfun(@gt, U, md);
    case 'median'
      md = median(U);
      B = bsxfun(@gt, U, md);
    case 'sign'
      B = U > 0;
    otherwise
      fprintf('Unknown rounding method\n');
  end

end
