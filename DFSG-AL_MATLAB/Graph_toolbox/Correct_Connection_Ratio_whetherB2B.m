function [Correct_ratio_SG_withB2B,Correct_ratio_SG_withoutB2B,CRCNum_SG]=...
            Correct_Connection_Ratio_whetherB2B(S_W,TruthMap,nodes_to_RC,nodes_to_add,nodes_to_stastic)
nodes_to_RC=find(nodes_to_RC~=0);
TruthMap1D=TruthMap(:);

maxweight=max(max(S_W));
thresh_weight=maxweight*0.002;
        
flag_withoutB2B=0;
if exist('nodes_to_add','var')
    flag_withoutB2B=1;
end

%% 计算图连接的正确率
% Correct_ratio_SG_withB2B=0;
Correct_ratio_SG_vec=[];
TotalCorrectNum_SG=0;
TotalConnectNum_SG=0;
TotalCorrectWeight_SG=0;
TotalConnectWeight_SG=0;

% Correct_ratio_SG_withoutB2B=0;
TotalCorrectWeight_SG_withoutB2B=0;
TotalConnectWeight_SG_withoutB2B=1;

CRCNum_SG=0;


if exist('nodes_to_stastic','var')
    nodes_to_stastic_flags=sum(nodes_to_stastic);
    frontsceneID=TruthMap1D.*nodes_to_stastic_flags';
    frontsceneID=find(frontsceneID>0);
else
    frontsceneID=find(TruthMap1D>0);
end

for frontsceneID_i=frontsceneID'
    centerLabel=TruthMap1D(frontsceneID_i);
    
    %SGG-withB2B的连接可信度统计
    connection_SG=S_W(frontsceneID_i,:);
    connectedID_SG=find(connection_SG>0);
    SGLabels=TruthMap1D(connectedID_SG);
    connectedID_SG=connectedID_SG(SGLabels~=0);
    correctID_SG=connectedID_SG(TruthMap1D(connectedID_SG)==centerLabel);
    totaldegree=sum(connection_SG(connectedID_SG));
    correctdegree=sum(connection_SG(correctID_SG));
    if totaldegree>0
        Correct_ratio_SG_vec=[Correct_ratio_SG_vec correctdegree/totaldegree];
    end
    TotalCorrectWeight_SG=TotalCorrectWeight_SG+correctdegree;
    TotalConnectWeight_SG=TotalConnectWeight_SG+totaldegree;
    TotalCorrectNum_SG=TotalCorrectNum_SG+length(correctID_SG);
    TotalConnectNum_SG=TotalConnectNum_SG+length(connectedID_SG);
    
    if ismember(frontsceneID_i,nodes_to_RC)
        matching_set=Searching_SameClassBlock_LocSet(frontsceneID_i,TruthMap);
%         RCFilter=ismember(correctID_SG,nodes_to_RC);
%         correctID_SG=correctID_SG(RCFilter);
        %SG的遥相关统计
        CRCFilter=~ismember(correctID_SG,matching_set);
        CRCID_SG=correctID_SG(CRCFilter);
        CRCID_SG=CRCID_SG(connection_SG(CRCID_SG)>thresh_weight);
        CRCNum_SG=CRCNum_SG+length(CRCID_SG);
        
        if ~isempty(CRCID_SG)
            %% 显示遥相关效果
            [center_y, center_x] = f1DTo2DCoord(size(TruthMap),frontsceneID_i);
            figure;
            imagesc(TruthMap)
            hold on
            for line_idx=1:length(CRCID_SG)
                [TempYCoord, TempXCoord] = f1DTo2DCoord(size(TruthMap),CRCID_SG(line_idx));
                line([center_x,TempXCoord],[center_y,TempYCoord],'color','k','LineWidth',2,'Marker','o','MarkerSize',2,'MarkerEdgeColor','r');
            end
        end
    end
    
    if flag_withoutB2B==1
        %SGG-withoutB2B的连接可信度统计
        connectedID_SG_withoutB2B=find(sparse(TruthMap1D~=0)&(nodes_to_add(frontsceneID_i,:)==false)');
        correctID_SG_withoutB2B=connectedID_SG_withoutB2B(TruthMap1D(connectedID_SG_withoutB2B)==centerLabel);
        totaldegree=sum(connection_SG(connectedID_SG_withoutB2B));
        correctdegree=sum(connection_SG(correctID_SG_withoutB2B));
        %     if totaldegree>0
        %         Correct_ratio_SG_vec=[Correct_ratio_SG_vec correctdegree/totaldegree];
        %     end
        TotalCorrectWeight_SG_withoutB2B=TotalCorrectWeight_SG_withoutB2B+correctdegree;
        TotalConnectWeight_SG_withoutB2B=TotalConnectWeight_SG_withoutB2B+totaldegree;
        %     TotalCorrectNum_SG=TotalCorrectNum_SG+length(correctID_SG);
        %     TotalConnectNum_SG=TotalConnectNum_SG+length(connectedID_SG);
    end
end
Correct_ratio_SG=mean(Correct_ratio_SG_vec);
Correct_ratio_SG_withB2B=TotalCorrectWeight_SG/TotalConnectWeight_SG;
% Correct_ratio_SG2_withB2B=TotalCorrectNum_SG/TotalConnectNum_SG;
Correct_ratio_SG_withoutB2B=TotalCorrectWeight_SG_withoutB2B/TotalConnectWeight_SG_withoutB2B;

CRCNum_SG=CRCNum_SG/2;
