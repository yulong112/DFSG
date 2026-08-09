function seg = alfaExpansion(eLabel,labels,initiaLabelling,beta,cliques);
% implements 'alfa-expansion algorithm' as described in 
% "Fast Approximate Energy Minimization via Graph Cuts", in IEEE
% Transactions on Patterns Analysis and M.I., vol 23, Nov. 2001
% eLabel: stappel of matrices correponding to energy associated to each class
% labels: vector containing all possible labels: 0, 1,...
% initialLabelling: initial segmentation; value is not important
% beta: parameter representing strenght of clique interaction
% cliques: vector containing cliques to be taken into accout for pixel interaction

success = false;
nrIter =0;
labelling = initiaLabelling;

while( ~success )
    for l=1:length(labels)
        alfa = labels(l);
        [flow,newLabelling]= gcAlfaExpansion(alfa,eLabel,labelling,beta,cliques);
        if (Energy(newLabelling, eLabel, beta, cliques)<Energy(labelling, eLabel, beta, cliques))
            labelling = newLabelling;
            success = true;     
        end    
    end
    success = ~success;    
    nrIter = nrIter+1;
end

seg = labelling;

