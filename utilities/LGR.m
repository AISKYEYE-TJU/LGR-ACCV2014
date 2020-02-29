function  accu=LGR(tr_dat,tt_dat,td_dat,tv_dat,trls,ttls,Image_row_NUM,Image_column_NUM,ps,pm,pw,kappa)

% tr_dat  training set 
% tt_dat  test set
% td_dat  reference set  (generic)
% tv_dat  validation set (generic)
% trls    training label
% ttls    test label
% Image_row_NUM  height of face image
% Image_column_NUM  width of face image
% ps     size of patches (default value 20 for 80*80 face image)
% pm     overlap size of patches (default 10 for 20*20 patch)
% pw     neighborhood size (default 1 )
% kappa  regularization parameter

%%
max_num=1;  %to improve the efficiency 
par.nDim=400;
Train_NUM=size(tr_dat,3);
Test_NUM=size(tt_dat,3);
class_num=length(unique(trls));

%divide test samples into patches 
[tt_patch,index]=patch_divide(tt_dat,Image_row_NUM,Image_column_NUM,Test_NUM, ps,pm);
patch_num=size(tt_patch,2);
tt_patch=permute(tt_patch,[1 3 2]);

for k=1:patch_num
 fprintf(['The number of patches ' num2str(k)]);   
 fprintf('\n')
 temp=[index(k,1),index(k,2)];
 [temp_patch,trlsd{k}]=patch_neigh(tr_dat,Image_row_NUM,Image_column_NUM,Train_NUM,temp(1),temp(2),ps,pw,trls); 
%[disc_set]  =  Eigenface_f(temp_patch,par.nDim);
 disc_set=eye(size(temp_patch,1));
 projM{k}=disc_set;
 temp_patch = disc_set'*temp_patch;
 tr_patch{k}  =  temp_patch./( repmat(sqrt(sum(temp_patch.*temp_patch)), [par.nDim,1]) );
 temp_patch = disc_set'*tt_patch(:,:,k);
 tt_patch_temp(:,:,k)  =  temp_patch./( repmat(sqrt(sum(temp_patch.*temp_patch)), [par.nDim,1]) ); 
end

tt_patch=tt_patch_temp;
tt_patch=permute(tt_patch,[1 3 2]);
clear tt_patch_temp  tr_dat tt_dat 

% reference set
td_patch=prodata(td_dat,projM,Image_row_NUM,Image_column_NUM,ps,pm,par);
for i=1:length(tv_dat)
    tv_patch{i}=prodata(tv_dat{i},projM,Image_row_NUM,Image_column_NUM,ps,pm,par);
end

for k=1:patch_num 
    temp_patch=[];
    for i=1:length(tv_dat)
    temp_patch=[temp_patch tv_patch{i}(:,:,k)-td_patch(:,:,k)];
    end
    dic_t{k}=temp_patch;
end
clear tv_patch td_patch tv_dat td_dat
%%
w=ones(1,patch_num);
pro=cell(1,patch_num);
proj_m=cell(1,patch_num);
for k=1:patch_num
   fprintf(['The number of patches ' num2str(k)]);   
   fprintf('\n')
   pro{k}=[tr_patch{k} dic_t{k}]'*[tr_patch{k} dic_t{k}]; 
   proj_m{k}=inv(w(k)* pro{k}+kappa*eye(size([tr_patch{k} dic_t{k}],2)))*w(k)* [tr_patch{k} dic_t{k}]';
end
clear pro

%%
for indTest=1:Test_NUM
fprintf(['The number of classifed samples ' num2str(indTest)]);
int_num=0;  
w=ones(1,patch_num);

while int_num<max_num 
   int_num=int_num+1;  
   %update the coefficients   
   coeff=cell(1,patch_num);
   for k=1:patch_num
       coeff{k}=proj_m{k}*tt_patch(:,k,indTest);
   end

   %update the sigma 
   error=zeros(1,patch_num);
   for k=1:patch_num
     error(k)=(norm(tt_patch(:,k,indTest)-[tr_patch{k} dic_t{k}]*coeff{k}))^2;
   end
   sigma2=mean(error)/2;
   %update the w
   w=exp(-error/sigma2/2)/sigma2;
end

% classification
error_patch=zeros(patch_num,class_num);
for m=1:patch_num
  train_num=size(tr_patch{m},2);  
  y_temp=dic_t{m}*coeff{m}(train_num+1:end);       
  for n=1:class_num
     d_temp=tr_patch{m}(:,trlsd{m}==n);
     train_num=size(tr_patch{m},2);
     coee_tem=coeff{m}(trlsd{m}==n);
     coee_temp=[coeff{m}(trlsd{m}==n);coeff{m}(train_num+1:end)];
     error_patch(m,n)=w(m)*(norm(tt_patch(:,m,indTest)-d_temp*coee_tem-y_temp))^2/sum(coee_temp.*coee_temp);    
  end
end
error=sum(error_patch);
[~,temp]=min(error);
label(indTest)=temp;
fprintf([' The label is ' num2str(temp)]);
fprintf('\n')
end

accu=sum(label==ttls)/length(ttls);
fprintf(['recogniton rate of LGR is ' num2str(accu)]);
fprintf('\n')
