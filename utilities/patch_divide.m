function [patch,index]=patch_divide(Train_DAT,Image_row_NUM,Image_column_NUM,Train_NUM,ps,pm)
% This function is used to divide the image to a set of patches
%Input:
% Train_DAT        
% Image_row_NUM
% Image_column_NUM
% Train_NUM
% ps                patch size 
% pm                overlapped size 
%Output:
% patch             patches(patch vector* patch number* sample number)
% index             patch position
Train_DAT=reshape(Train_DAT,[Image_row_NUM*Image_column_NUM,Train_NUM]);
Train_DAT=Train_DAT./ repmat(sqrt(sum(Train_DAT.*Train_DAT)),[size(Train_DAT,1) 1]); % unit norm 2
Train_DAT=reshape(Train_DAT,[Image_row_NUM,Image_column_NUM,Train_NUM]);
po=ps-pm;
patch_row=fix((Image_row_NUM-po)/pm);                       % number of patches per row 
patch_col=fix((Image_column_NUM-po)/pm);                    % number of patches per column

%square patches 
patch=[];
po=ps-pm;
for k=1:Train_NUM
  for i=1:patch_row
    for j=1:patch_col  
      s=patch_row*(i-1)+j;
      index(s,1:2)=[pm*(i-1)+1,pm*(j-1)+1];
      C1= Train_DAT((pm*(i-1)+1):(pm*i+po),(pm*(j-1)+1):(pm*j+po),k);  %patch for training set
      patch(:,s,k)=reshape(C1,[ps*ps,1])/norm(C1);
    end
  end
end


