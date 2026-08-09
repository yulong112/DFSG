function Filter1D = fFilter1D(FilterType, Direction, Para1, Para2)
%   1. FilterType = 'GaussianDerivative', 
%       Para1 is the scale, 
%       Para2:  controlling the N-sigma rule.
%       Direction: direction of filtering
 
%

if strcmpi(FilterType,'GaussianDerivative')
    Scale = Para1;
    
    MaskSize = ceil(Para2*Scale); % 3 \sigma rule
    X = [-MaskSize:MaskSize]'; % voxel lattice
    
    %%% envelop construction
    Temp =  (X.*X)/(Scale^2);
    GaussianFilter = (exp(-0.5*Temp))/((2*pi)^0.5*Scale);
    Filter1DTemp = (-X/(Scale^2)) .* GaussianFilter;
    
    if strcmpi(Direction,'X')
        Filter1D = Filter1DTemp';
    end
    if strcmpi(Direction,'Y')
       Filter1D = Filter1DTemp;
    end
    if strcmpi(Direction,'Z')
        Filter1D(1,1,:) = Filter1DTemp;
    end
end

