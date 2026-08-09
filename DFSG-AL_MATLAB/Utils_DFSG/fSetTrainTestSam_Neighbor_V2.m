function [TrainSam,TestSam,TrainLabels,TestLabels,TrainSamLoc,TestSamLoc] = ...
    fSetTrainTestSam_Neighbor_V2(HyperData,TruthMap,SelClassNo,NeighborMask,RandNo,TrainSamNum,TestSamNum)
%%%%%%%%%%%%%%%%
% if TestSamNum is of absence, it is set to be default value
% note: "SelClassNo = 1" denotes that the background is included
%          if 0 values in TruthMap correspond to background
%          eg. if unique(TruthMap) result [0 1 ... 16] and 0
%          value is referred to the label of background,
%          "SelClassNo = 2:17" is for the valid categories.
% SelClassNo:
% RandNo: No of seed for random number generating
% NeighborMask: to be utilized for sample selection in a neighbor
% TruthMap: 始终是二维label图
%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DataDim = ndims(HyperData);
if DataDim ==3
    [Height,Width,l] = size(HyperData);
    TempHyperData2D = reshape(HyperData, Width*Height, l); %3D变 2D
    Label1D = reshape(TruthMap, Width*Height,1);
end
if DataDim ==2
    TempHyperData2D = HyperData;
    Label1D = TruthMap(:);
    [Height, Width] = size(TruthMap);
end

TotalIdx = 1:Height*Width;
%Label1D = reshape(TruthMap, Width*Height,1);
ClassNo = unique(Label1D);                        %计算出类的数量

%            TempHyperData2D = HyperCube2D;
%%
%get samples from all classes, i.e. totally 17
% classes  including background (for IdianPine Imagery or some other data set).
for i = 1:length(ClassNo)
    LabelOneClass = find(Label1D == i-1);%%!!!%LabelOneClass为编号为i-1的位置
    AllSamWithLabel{i} = TempHyperData2D(LabelOneClass,:);%
    Locaction1D{i} = LabelOneClass;% Original index
    TempSamNum = size(AllSamWithLabel{i},1);
    AllLabels{i} = i*ones(TempSamNum,1);%set Label to be -1 and 1 for two-class problem
end

%% select class
% AllSamWithLabel = AllSamWithLabel(SelClassNo);
AllLabels = AllLabels(SelClassNo);
Locaction1D = Locaction1D(SelClassNo);
for tempNo = 1:length(AllLabels)
    AllLabels{tempNo} = tempNo*ones(length(AllLabels{tempNo}),1);
end
%% set the numbers of training and test samples
if TrainSamNum(1)<1
for tempNo = 1:length(AllLabels)
    TrainSamNum(tempNo) = TrainSamNum(tempNo)*length(AllLabels{tempNo});
end
TrainSamNum = round(TrainSamNum);
end


%% initialization of samples, labels and location of train and test set, respectively. 
for CellNo =1:length(find(NeighborMask(:)==1))
    TrainSam{CellNo} = [ ];
    TestSam{CellNo} = [ ];
end
TrainLabels = [ ];
TestLabels = [ ];
TrainSamLoc=[ ];
TestSamLoc=[ ];

%%  get train and test samples
for ClassNum = 1 : length(AllLabels)
    [Loc1DSubset1,Loc1DSubset2,LocalLoc1D1,LocalLoc1D2] = fRandomSampling...
        (Locaction1D{ClassNum},TrainSamNum(ClassNum),RandNo);%sample selection
    %Loc1DSubset1: index relative to total image (train sample);
    %LocalLoc1D1:local index relative to current class (train sample)
    [YCoordTrainSam, XCoordTrainSam] = f1DTo2DCoord([Height, Width],Loc1DSubset1);% coordinate of center point of mask
    TrainLabels=[TrainLabels;AllLabels{ClassNum}(LocalLoc1D1)];%label selection
    TrainSamLoc = [TrainSamLoc;Loc1DSubset1];
    
    if nargin == 6 % all the samples not belonging to train samples are set to be test samples
        [YCoordTestSam, XCoordTestSam] = f1DTo2DCoord([Height, Width],Loc1DSubset2);
        TestLabels=[TestLabels;AllLabels{ClassNum}(LocalLoc1D2)];%label selection
        TestSamLoc = [TestSamLoc;Loc1DSubset2];
    else% predeterminded number of samples not belonging to train samples are set to be test samples
        [YCoordTestSam, XCoordTestSam] = f1DTo2DCoord([Height, Width],Loc1DSubset2(1:TestSamNum(ClassNum)));
        TestLabels=[TestLabels;AllLabels{ClassNum}(LocalLoc1D2(1:TestSamNum(ClassNum)))];%label selection
        TestSamLoc = [TestSamLoc;Loc1DSubset2(1:TestSamNum(ClassNum))];
    end
    
    %for TrainSamNo = 1:length(YCoordTrainSam)
    TempCoordTrainSam = [YCoordTrainSam', XCoordTrainSam'];
    [SelCoord2DCellTrainSam,SelCoord1DCellTrainSam] = fGetNeighborCoord([Height, Width], TempCoordTrainSam, NeighborMask);
    TempCoordTestSam = [YCoordTestSam', XCoordTestSam'];
    [SelCoord2DCellTestSam,SelCoord1DCellTestSam] = fGetNeighborCoord([Height, Width], TempCoordTestSam, NeighborMask);
    
    for NeighborNo = 1:length(SelCoord1DCellTrainSam) % get samples
        %TempTotalIdx=TotalIdx;
        TempSelCoord1 = SelCoord1DCellTrainSam{NeighborNo};
        TempSelCoord2 = SelCoord1DCellTestSam{NeighborNo};
        %TempTotalIdx(TempSelCoord) = [];
        TrainSam{NeighborNo}=[TrainSam{NeighborNo};TempHyperData2D(TempSelCoord1,:)];%将同一邻域位置不同类的样本垒起来
        TestSam{NeighborNo}=[TestSam{NeighborNo};TempHyperData2D(TempSelCoord2,:)];
    end
    
end

%% adjust label in term of the total number of classes
% if it is two class problem, the labels are set to be 1 and -1;
% otherwise, the labels are set to be 1,2,....
if length(SelClassNo) ==2
    TrainLabels(find(TrainLabels ==2)) = -1;
    TestLabels(find(TestLabels ==2)) = -1;
end

function [SelCoord2DCell,SelCoord1DCell] = fGetNeighborCoord(ImgSize,CenterCoord,NeighborMask)
% ImgSize: size of total image
% CenterCoord: coordinate of center point of mask

[MaskSizeY,MaskSizeX] = size(NeighborMask);
PointNum = size(CenterCoord,1);
NeighborMask1d = NeighborMask(:);
CoordNum = length(find(NeighborMask1d==1));
SelCorrd  = zeros(CoordNum,2);

CoordNo = 1;
TempY = floor(MaskSizeY/2);
TempX = floor(MaskSizeX/2);

% set real location relative to real image
for i = 1:MaskSizeY
    for j = 1:MaskSizeX
        if NeighborMask(i,j)==1
            SelCorrd = CenterCoord+repmat([ i-1-TempY, j-1-TempX],PointNum,1);
            % tackle the coordinates out of the image
            TempIdx1 = find(SelCorrd(:,1)<=0);
            if length(TempIdx1)~=0
                SelCorrd(TempIdx1,1) = (-1)*SelCorrd(TempIdx1,1)+1;
            end
            
            TempIdx2 = find(SelCorrd(:,1)>ImgSize(1));
            if length(TempIdx2)~=0
                SelCorrd(TempIdx2,1) = ImgSize(1)- (SelCorrd(TempIdx2,1)-ImgSize(1))+1;
            end
            
            TempIdx3 = find(SelCorrd(:,2)<=0);
            if length(TempIdx3)~=0
                SelCorrd(TempIdx3,2) = (-1)*SelCorrd(TempIdx3,2)+1;
            end
            
            TempIdx4 = find(SelCorrd(:,2)>ImgSize(2));
            if length(TempIdx4)~=0
                SelCorrd(TempIdx4,2) = ImgSize(2)- (SelCorrd(TempIdx4,2)-ImgSize(2))+1;
            end
            
            SelCoord2DCell{CoordNo} = SelCorrd;
            SelCoord1DCell{CoordNo} = ImgSize(1)*(SelCorrd(:,2)-1)+SelCorrd(:,1);
            CoordNo = CoordNo+1;
        end
    end
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%% backup
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DataDim = ndims(HyperData);
% if DataDim ==3
%     [m,n,l] = size(HyperData);
%     TempHyperData2D = reshape(HyperData, m*n, l); %3D变 2D
% end
% if DataDim ==2
%     TempHyperData2D = HyperData;
% end
% %%Set %%%%%%%%%%%%%%%%%%%%%
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Height, Width] = size(TruthMap);
% TotalIdx = 1:Height*Width;
% Label1D = reshape(TruthMap, Width*Height,1);
% ClassNo = unique(Label1D);                        %计算出类的数量
% 
% %            TempHyperData2D = HyperCube2D;
% %%
% %get samples from all classes, i.e. totally 17
% % classes  including background (for IdianPine Imagery or some other data set).
% for i = 1:length(ClassNo)
%     LabelOneClass = find(Label1D == i-1);%%!!!%LabelOneClass为编号为i-1的位置
%     AllSamWithLabel{i} = TempHyperData2D(LabelOneClass,:);%
%     Locaction1D{i} = LabelOneClass;% Original index
%     TempSamNum = size(AllSamWithLabel{i},1);
%     AllLabels{i} = i*ones(TempSamNum,1);%set Label to be -1 and 1 for two-class problem
% end
% 
% %% select class
% % AllSamWithLabel = AllSamWithLabel(SelClassNo);
% AllLabels = AllLabels(SelClassNo);
% Locaction1D = Locaction1D(SelClassNo);
% for tempNo = 1:length(AllLabels)
%     AllLabels{tempNo} = tempNo*ones(length(AllLabels{tempNo}),1);
% end
% 
% %% initialization of samples, labels and location of train and test set, respectively. 
% for CellNo =1:length(find(NeighborMask(:)==1))
%     TrainSam{CellNo} = [ ];
%     TestSam{CellNo} = [ ];
% end
% TrainLabels = [ ];
% TestLabels = [ ];
% TrainSamLoc=[ ];
% TestSamLoc=[ ];
% 
% %%  get train and test samples
% for ClassNum = 1 : length(AllLabels)
%     [Loc1DSubset1,Loc1DSubset2,LocalLoc1D1,LocalLoc1D2] = fRandomSampling...
%         (Locaction1D{ClassNum},TrainSamNum(ClassNum),RandNo);%sample selection
%     %Loc1DSubset1: index relative to total image (train sample);
%     %LocalLoc1D1:local index relative to current class (train sample)
%     [YCoordTrainSam, XCoordTrainSam] = f1DTo2DCoord([Height, Width],Loc1DSubset1);% coordinate of center point of mask
%     TrainLabels=[TrainLabels;AllLabels{ClassNum}(LocalLoc1D1)];%label selection
%     TrainSamLoc = [TrainSamLoc;Loc1DSubset1];
%     
%     if nargin == 6 % all the samples not belonging to train samples are set to be test samples
%         [YCoordTestSam, XCoordTestSam] = f1DTo2DCoord([Height, Width],Loc1DSubset2);
%         TestLabels=[TestLabels;AllLabels{ClassNum}(LocalLoc1D2)];%label selection
%         TestSamLoc = [TestSamLoc;Loc1DSubset2];
%     else% predeterminded number of samples not belonging to train samples are set to be test samples
%         [YCoordTestSam, XCoordTestSam] = f1DTo2DCoord([Height, Width],Loc1DSubset2(1:TestSamNum(ClassNum)));
%         TestLabels=[TestLabels;AllLabels{ClassNum}(LocalLoc1D2(1:TestSamNum(ClassNum)))];%label selection
%         TestSamLoc = [TestSamLoc;Loc1DSubset2(1:TestSamNum(ClassNum))];
%     end
%     
%     %for TrainSamNo = 1:length(YCoordTrainSam)
%     TempCoordTrainSam = [YCoordTrainSam', XCoordTrainSam'];
%     [SelCoord2DCellTrainSam,SelCoord1DCellTrainSam] = fGetNeighborCoord([Height, Width], TempCoordTrainSam, NeighborMask);
%     TempCoordTestSam = [YCoordTestSam', XCoordTestSam'];
%     [SelCoord2DCellTestSam,SelCoord1DCellTestSam] = fGetNeighborCoord([Height, Width], TempCoordTestSam, NeighborMask);
%     
%     for NeighborNo = 1:length(SelCoord1DCellTrainSam) % get samples
%         %TempTotalIdx=TotalIdx;
%         TempSelCoord1 = SelCoord1DCellTrainSam{NeighborNo};
%         TempSelCoord2 = SelCoord1DCellTestSam{NeighborNo};
%         %TempTotalIdx(TempSelCoord) = [];
%         TrainSam{NeighborNo}=[TrainSam{NeighborNo};TempHyperData2D(TempSelCoord1,:)];
%         TestSam{NeighborNo}=[TestSam{NeighborNo};TempHyperData2D(TempSelCoord2,:)];
%     end
%     
% end
% 
% %% adjust label in term of the total number of classes
% % if it is two class problem, the labels are set to be 1 and -1;
% % otherwise, the labels are set to be 1,2,....
% if length(SelClassNo) ==2
%     TrainLabels(find(TrainLabels ==2)) = -1;
%     TestLabels(find(TestLabels ==2)) = -1;
% end
% 
% function [SelCoord2DCell,SelCoord1DCell] = fGetNeighborCoord(ImgSize,CenterCoord,NeighborMask)
% % ImgSize: size of total image
% % CenterCoord: coordinate of center point of mask
% 
% [MaskSizeY,MaskSizeX] = size(NeighborMask);
% PointNum = size(CenterCoord,1);
% NeighborMask1d = NeighborMask(:);
% CoordNum = length(find(NeighborMask1d==1));
% SelCorrd  = zeros(CoordNum,2);
% 
% CoordNo = 1;
% TempY = floor(MaskSizeY/2);
% TempX = floor(MaskSizeX/2);
% 
% % set real location relative to real image
% for i = 1:MaskSizeY
%     for j = 1:MaskSizeX
%         if NeighborMask(i,j)==1
%             SelCorrd = CenterCoord+repmat([ i-1-TempY, j-1-TempX],PointNum,1);
%             % tackle the coordinates out of the image
%             TempIdx1 = find(SelCorrd(:,1)<=0);
%             if length(TempIdx1)~=0
%                 SelCorrd(TempIdx1,1) = (-1)*SelCorrd(TempIdx1,1)+1;
%             end
%             
%             TempIdx2 = find(SelCorrd(:,1)>ImgSize(1));
%             if length(TempIdx2)~=0
%                 SelCorrd(TempIdx2,1) = ImgSize(1)- (SelCorrd(TempIdx2,1)-ImgSize(1))+1;
%             end
%             
%             TempIdx3 = find(SelCorrd(:,2)<=0);
%             if length(TempIdx3)~=0
%                 SelCorrd(TempIdx3,2) = (-1)*SelCorrd(TempIdx3,2)+1;
%             end
%             
%             TempIdx4 = find(SelCorrd(:,2)>ImgSize(2));
%             if length(TempIdx4)~=0
%                 SelCorrd(TempIdx4,2) = ImgSize(2)- (SelCorrd(TempIdx4,2)-ImgSize(2))+1;
%             end
%             
%             SelCoord2DCell{CoordNo} = SelCorrd;
%             SelCoord1DCell{CoordNo} = ImgSize(1)*(SelCorrd(:,2)-1)+SelCorrd(:,1);
%             CoordNo = CoordNo+1;
%         end
%     end
% end
