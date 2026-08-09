function nodes_to_add = Connect_pixel_btwsp(nodes_to_super,superpix_img, distance, param2, flag2)
N=size(distance,1);
nodes_to_add = false(N);
for i = 1:size(nodes_to_super,1)
    similar_set=find(nodes_to_super(i,:)>0);
    similar_set=[similar_set i];
    if length(similar_set)>1
        for i_set=2:length(similar_set)
            for j_set=1:i_set-1
                sp_idx1=find(superpix_img==similar_set(i_set));
                sp_idx2=find(superpix_img==similar_set(j_set));
                dist_mat=distance(sp_idx1,sp_idx2);
                idx_connect=[];
                if strcmp(param2,'DIST')||strcmp(param2,'DIST2')||strcmp(param2,'DIST3')
                    [~,idx_connect]=min(dist_mat,[],2);
                else
                    [~,idx_connect]=max(dist_mat,[],2);
                end
                nodes_to_add((sp_idx2(idx_connect)-1)*N+sp_idx1)=true;
            end
        end
        
    else
        printtex='³öÏÖ¹ÂÁ¢³¬ÏñËØ¿é,³¬ÏñËØ¿éĞòºÅ£º'+num2str(i)
    end
end
nodes_to_add( nodes_to_add ~= nodes_to_add' ) = true;
