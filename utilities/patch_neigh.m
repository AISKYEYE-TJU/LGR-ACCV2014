function [patch4,ttls]=patch_neigh(Train_DAT,Image_row_NUM,Image_column_NUM,Train_NUM,pos_x,pos_y,ps,pw,trls)
%this function is used to get the neighbor patches at the given position 
%Input:
%Train_DAT:
%Image_row_NUM    : the height of the iimage 
%Image_column_NUM : the width of the image 
%Train_NUM        : the number of the training samples 
%pos_x            : the position of the patch 
%pos_y            : the position of the patch 
%ps               : patch size 
%pw               : neighbor size 
%trls             : the training labels           
%Output:
%patch4           : the obtained patches 
%ttls             : the label of the patches 

%% normorize the data
Train_DAT=reshape(Train_DAT,[Image_row_NUM*Image_column_NUM,Train_NUM]);
Train_DAT=Train_DAT./ repmat(sqrt(sum(Train_DAT.*Train_DAT)),[size(Train_DAT,1) 1]); % unit norm 2
Train_DAT=reshape(Train_DAT,[Image_row_NUM,Image_column_NUM,Train_NUM]);
%% divide each image into patches 
%square patches 
pm=1; 
po=ps-pm;
rowtem1=pos_x-pw;
rowtem2=pos_x+pw;
coltem1=pos_y-pw;
coltem2=pos_y+pw;
if rowtem1<1
    rowtem1=1;
end

if coltem1<1
    coltem1=1;
end

if rowtem2>Image_row_NUM-po
    rowtem2=Image_row_NUM-po;
end

if coltem2>Image_row_NUM-po
    coltem2=Image_row_NUM-po;
end


patch4=[];
ttls=[];
for k=1:Train_NUM
     s=0;
     patch=[];
     m=[];
     for i=rowtem1:rowtem2
         for j=coltem1:coltem2
           s=s+1;
           C4= Train_DAT((pm*(i-1)+1):(pm*i+po),(pm*(j-1)+1):(pm*j+po),k);  %patch for training set
           patch(:,s)=reshape(C4,[ps*ps,1])/norm(C4);
         end
     end
     patch4=[patch4,patch];
     ttls=[ttls;ones(s,1)*trls(k)];
end
