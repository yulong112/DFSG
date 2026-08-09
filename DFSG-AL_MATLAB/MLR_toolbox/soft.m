function y = soft(x,T)
%        y = soft(x,T)
% soft thresholding  function; the solution of the l2-l1 problem
%
%   y = arg min (1/2)||u-x||^2_2 + T ||u||_1
%            u
%
%  x - matrix
%  T - threshold
%

y = max(abs(x) - T, 0);
y = y./(y+T) .* x;

