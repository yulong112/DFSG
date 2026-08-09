function [Superpixel_Weighted_Feature, centroidal_Loc]=Superpixel_Weighted_Features_Compute(superpixel_value,superpixel_Locs,superpixel_img,Height,Width,Superpixel_Mean_Features,kernel_h)

[neighbor_superpixel_values, centroidal_Loc]=Superpixel_Neighbor_Searching(superpixel_value,superpixel_Locs,superpixel_img,Height,Width);

feature_weight_sum=0;
Superpixel_Weighted_Feature=0;

center_Mean_Feature=Superpixel_Mean_Features(superpixel_value,:);

neighbor_superpixel_values=[neighbor_superpixel_values;superpixel_value];
for search_i=1:length(neighbor_superpixel_values)
    neighbor_superpixel_value=neighbor_superpixel_values(search_i);
    neighbor_Mean_Feature=Superpixel_Mean_Features(neighbor_superpixel_value,:);
    
    distance = sum((neighbor_Mean_Feature-center_Mean_Feature).^2,2);
    feature_weight=exp( -distance / kernel_h^2 );
    feature_weight_sum=feature_weight_sum+feature_weight;
    
    Superpixel_Weighted_Feature=Superpixel_Weighted_Feature+feature_weight*neighbor_Mean_Feature;
end
Superpixel_Weighted_Feature=Superpixel_Weighted_Feature/feature_weight_sum;
