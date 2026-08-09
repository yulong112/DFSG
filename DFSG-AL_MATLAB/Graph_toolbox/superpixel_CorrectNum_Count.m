function [SCR,sp_num,count_error]=superpixel_CorrectNum_Count(superpix_img,TruthMap1D)

%% 观察超像素分割的优劣：输出存在不同类的超像素块个数
sp_num=max(superpix_img);
labels_all=TruthMap1D;
count_error=0;
for sp_i=1:sp_num
    sp_idx= superpix_img==sp_i;
    unique_label=unique(labels_all(sp_idx));
    unique_label=setdiff(unique_label,0);
    if length(unique_label)>1;
        count_error=count_error+1;
    end
end
SCR=count_error*1.0/double(sp_num);
