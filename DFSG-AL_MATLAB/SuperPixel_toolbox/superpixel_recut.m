
function superpix_img2 = superpixel_recut(HyperCube, superpix_img, TrainSamLoc, TrainLabels)

[nl ns nb] = size(HyperCube);

PredLabels=MLR_Classifier_fun(HyperCube,TrainSamLoc,TrainLabels,ToVector(HyperCube));

superpix_img2=superpix_img;
sp_num=max(superpix_img);
sp_num_tmp=sp_num;
for select_i=1:sp_num
    pos_idx=find(superpix_img==select_i);
    blk_prelabels=PredLabels(pos_idx);
%     hist_labels=tabulate(blk_prelabels);
%     unique_labels=hist_labels(:,1);
%     unique_labels=unique_labels(hist_labels(:,2)>0);
%     total_unique=length(unique_labels);
%     if total_unique>1;
%         for unique_i=2:total_unique
%             
%             hist_labels=tabulate(blk_prelabels);
%             unique_labels=hist_labels(:,1);
%             unique_labels=unique_labels(hist_labels(:,2)>0);
%             count_labels=hist_labels(:,2);
%             count_labels=count_labels(count_labels>0);
%             [~,min_idx]=min(count_labels);
%             min_pos_vec=find(blk_prelabels==unique_labels(min_idx));
%             for each_min=1:length(min_pos_vec)
%                 min_pos_idx=pos_idx(min_pos_vec(each_min));
%                 pos_adj=[min_pos_idx+1 min_pos_idx-1 min_pos_idx-ns min_pos_idx+ns];
%                 [~,adj_idx]=ismember(pos_adj,pos_idx);
%                 adj_label_vec=PredLabels(pos_idx(adj_idx(adj_idx>0)));
%                 find_same=find(adj_label_vec==unique_labels(min_idx));
% %                 if sum(adj_idx)>0
%                 if length(find_same)==0
%                     length_vec=[];
%                     for adj_i=1:length(adj_label_vec)
%                         length_vec(adj_i)=length(find(blk_prelabels==adj_label_vec(adj_i)));
%                     end
%                     [~,max_idx]=max(length_vec);
%                     blk_prelabels(min_pos_vec(each_min))=adj_label_vec(max_idx);
%                 end
%             end
%         end
%     end
    unique_labels=unique(blk_prelabels);
    if length(unique_labels)>1;
        for unique_i=2:length(unique_labels)
            idx_i=find(blk_prelabels==unique_labels(unique_i));
            superpix_img2(pos_idx(idx_i))= sp_num_tmp+1;
            sp_num_tmp=sp_num_tmp+1;
        end
    end
end
now_sp_num=max(superpix_img2)