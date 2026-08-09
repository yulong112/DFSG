function [neighbor_superpixel_values, centroidal_Loc]=Superpixel_Neighbor_Searching(superpixel_value,superpixel_Locs,superpixel_img,Height,Width)

neighbor_superpixel_values=[];
YCoordSet=[];
XCoordSet=[];
for search_i=1:length(superpixel_Locs)
    centerloc=superpixel_Locs(search_i);
    [TempYCoord, TempXCoord] = f1DTo2DCoord([Height, Width],centerloc);
    YCoordSet=[YCoordSet;TempYCoord];
    XCoordSet=[XCoordSet;TempXCoord];
    
    candidateXCoord=[TempXCoord;TempXCoord;TempXCoord-1;TempXCoord+1];
    candidateYCoord=[TempYCoord-1;TempYCoord+1;TempYCoord;TempYCoord];
    borderFilter=(candidateXCoord>=1)&(candidateXCoord<=Width)&(candidateYCoord>=1)&(candidateYCoord<=Height);
    candidateXCoord=candidateXCoord(borderFilter);
    candidateYCoord=candidateYCoord(borderFilter);
    
    candidateLocSet_tmp=(candidateXCoord-1)*Height+candidateYCoord;
    
    tmp_neighbor_superpixel_values = superpixel_img(candidateLocSet_tmp);
    
    tmp_neighbor_superpixel_values=unique(tmp_neighbor_superpixel_values);
    tmp_neighbor_superpixel_values=tmp_neighbor_superpixel_values(tmp_neighbor_superpixel_values~=superpixel_value);
    
    neighbor_superpixel_values=[neighbor_superpixel_values;tmp_neighbor_superpixel_values];
    
end

neighbor_superpixel_values=unique(neighbor_superpixel_values);

centroidal_Loc_YCoord=floor(mean(YCoordSet));
centroidal_Loc_XCoord=floor(mean(XCoordSet));

% centroidal_Loc=(centroidal_Loc_XCoord-1)*Height+centroidal_Loc_YCoord;
centroidal_Loc=[centroidal_Loc_XCoord, centroidal_Loc_YCoord];
