function [block_data,block_label] = Class_Data_RandomSampling(ALL_data,AvailableLabels,AvailableLoc,class_i,rows,cols)
class_i_id= AvailableLabels==class_i;
class_i_loc=AvailableLoc(class_i_id);
sample_class_i_loc=fRandomSampling(class_i_loc,rows*cols,0);
%                 ALL_labels = reshape(TruthMap, Width*Height, 1); %3D±ä 2D
%                 sample_class_i_label=ALL_labels(sample_class_i_loc);
%                 unique(sample_class_i_label)
sample_class_i_data=ALL_data(sample_class_i_loc,:);
block_data=reshape(sample_class_i_data,rows,cols,[]);
block_label=ones(rows,cols)*class_i;