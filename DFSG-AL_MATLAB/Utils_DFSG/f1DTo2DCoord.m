function [TotalYCoord,TotalXCoord] = f1DTo2DCoord(ImgSize,TotalCoord1D)
% ImgSize(1): Height
%ImgSize(2): Width

TotalYCoord=[];% in case that TotalCoord1D is empty, which arises while treat all samples as training sample.
TotalXCoord=[];

for i = 1: length(TotalCoord1D)
    Coord1D = TotalCoord1D(i);
    YCoord = mod(Coord1D,ImgSize(1));% modulous after division
    
    if YCoord==0
        XCoord = floor(Coord1D/ImgSize(1));
        YCoord = ImgSize(1);
    else
        XCoord = floor(Coord1D/ImgSize(1))+1;
    end
    TotalXCoord(i) = XCoord;
    TotalYCoord(i) = YCoord;
end