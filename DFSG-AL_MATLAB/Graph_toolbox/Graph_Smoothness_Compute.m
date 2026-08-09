function smoothness=Graph_Smoothness_Compute(Laplace_Mat,F_Truth_Labels)
no_classes=size(F_Truth_Labels,2);
smoothness2=[];
for select_i=1:no_classes
    smoothness2=[smoothness2,F_Truth_Labels(:,select_i)'*Laplace_Mat*F_Truth_Labels(:,select_i)];
end
smoothness=sum(smoothness2);
