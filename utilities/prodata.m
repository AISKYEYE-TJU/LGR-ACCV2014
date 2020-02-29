function  tt_patch=prodata(tt_dat,projM,Image_row_NUM,Image_column_NUM,ps,pm,par)

tt_dat=reshape(tt_dat,[Image_row_NUM Image_column_NUM size(tt_dat,2)]);
tt_dat=imresize(tt_dat,[Image_row_NUM Image_column_NUM]);
tt_dat=reshape(tt_dat,[Image_row_NUM*Image_column_NUM size(tt_dat,3)]);
tt_dat       =    tt_dat./ repmat(sqrt(sum(tt_dat.*tt_dat)),[size(tt_dat,1) 1]); % unit norm 2
tt_dat=reshape(tt_dat,[Image_row_NUM Image_column_NUM size(tt_dat,2)]);

Test_NUM=size(tt_dat,3);
[tt_patch,index]=patch_divide(tt_dat,Image_row_NUM,Image_column_NUM,Test_NUM, ps,pm);
patch_num=size(tt_patch,2);
tt_patch=permute(tt_patch,[1 3 2]);

for k=1:patch_num 
 temp_patch = projM{k}'*tt_patch(:,:,k);
 tt_patch_temp(:,:,k)  =  temp_patch./( repmat(sqrt(sum(temp_patch.*temp_patch)), [par.nDim,1]) );
end

tt_patch=tt_patch_temp;