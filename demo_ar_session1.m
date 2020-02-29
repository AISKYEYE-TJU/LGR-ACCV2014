
clear all;
clc;
addpath utilities;
tem_fd = cd;
par.d_fd          =   [cd '\database\'];

%        database='illumination'
%        database='expression' 
%        database='scarf'
%        database='glass'
%        database='disguise'
         database='disill'
               
%seting parameter
pro_sign          =   100;        
par.nDim          =   70;
lambda            =   0.01;      %test lambda
dnum              =   400;
Image_row_NUM=80;
Image_column_NUM=80;
ps=20;
pm=10;

par.nameDatabase  =   'AR_disguise';

% gallery set
load([par.d_fd 'AR_database']);
Tr_dataMatrix=reshape(Tr_dataMatrix,[165 120 size(Tr_dataMatrix,2)]);
Tr_dataMatrix=imresize(Tr_dataMatrix,[80 80]);
Tr_dataMatrix=reshape(Tr_dataMatrix,[80*80 size(Tr_dataMatrix,3)]);
Tt_dataMatrix=reshape(Tt_dataMatrix,[165 120 size(Tt_dataMatrix,2)]);
Tt_dataMatrix=imresize(Tt_dataMatrix,[80 80]);
Tt_dataMatrix=reshape(Tt_dataMatrix,[80*80 size(Tt_dataMatrix,3)]);

tr_dat = []; trls = [];
for ci = 1:80
    tr_dat = [tr_dat Tr_dataMatrix(:,1+7*(ci-1):1+7*(ci-1))];
    trls   = [trls repmat(ci,[1 1])];
end

% generic training set
td_dat = []; tdls = [];
for ci = 81:100
    td_dat = [td_dat Tr_dataMatrix(:,1+7*(ci-1):1+7*(ci-1))];
    tdls   = [tdls repmat(ci,[1 1])];
end

for vi=1:6
  temp =[];
  for ci = 81:100
    temp = [temp Tr_dataMatrix(:,vi+1+7*(ci-1):vi+1+7*(ci-1))];
  end
  tv_dat{vi}=temp;
end
clear Tr_dataMatrix Tr_sampleLabels Tt_dataMatrix Tt_sampleLabels;


load([par.d_fd 'AR_database_Occlusion.mat']);
Tr_dataMatrix=reshape(Tr_dataMatrix,[165 120 size(Tr_dataMatrix,2)]);
Tr_dataMatrix=imresize(Tr_dataMatrix,[80 80]);
Tr_dataMatrix=reshape(Tr_dataMatrix,[80*80 size(Tr_dataMatrix,3)]);
Tt_dataMatrix=reshape(Tt_dataMatrix,[165 120 size(Tt_dataMatrix,2)]);
Tt_dataMatrix=imresize(Tt_dataMatrix,[80 80]);
Tt_dataMatrix=reshape(Tt_dataMatrix,[80*80 size(Tt_dataMatrix,3)]);

for vi=7:12
  temp =[];
  for ci = 81:100
    temp = [temp Tr_dataMatrix(:,vi-6+6*(ci-1):vi-6+6*(ci-1))];
  end
  tv_dat{vi}=temp;
end

% testing set
tt_dat_sunglass = []; ttls_sunglass = [];
tt_dat_scarf = []; ttls_scarf = [];
%% scarf and sunglass 
for ci = 1:80
    tt_dat_sunglass = [tt_dat_sunglass Tr_dataMatrix(:,1+6*(ci-1):3+6*(ci-1))];
    ttls_sunglass   = [ttls_sunglass repmat(ci,[1 3])];
    
    tt_dat_scarf    = [tt_dat_scarf Tr_dataMatrix(:,4+6*(ci-1):6+6*(ci-1))];
    ttls_scarf      = [ttls_scarf repmat(ci,[1 3])];
end

%% disguise
tt_dat_dis = []; ttls_dis = [];
for ci = 1:80
    tt_dat_dis = [tt_dat_dis Tr_dataMatrix(:,1+6*(ci-1):1+6*(ci-1))];
    tt_dat_dis = [tt_dat_dis Tr_dataMatrix(:,4+6*(ci-1):4+6*(ci-1))];
    ttls_dis   = [ttls_dis repmat(ci,[1 2])];
end

%% disillu
tt_dat_dil = []; ttls_dil = [];
for ci = 1:80
    tt_dat_dil = [tt_dat_dil Tr_dataMatrix(:,2+6*(ci-1):3+6*(ci-1))];
    tt_dat_dil = [tt_dat_dil Tr_dataMatrix(:,5+6*(ci-1):6+6*(ci-1))];
    ttls_dil   = [ttls_dil repmat(ci,[1 4])];
end
clear Tr_dataMatrix Tr_sampleLabels Tt_dataMatrix Tt_sampleLabels;

load([par.d_fd 'AR_database']);
Tr_dataMatrix=reshape(Tr_dataMatrix,[165 120 size(Tr_dataMatrix,2)]);
Tr_dataMatrix=imresize(Tr_dataMatrix,[80 80]);
Tr_dataMatrix=reshape(Tr_dataMatrix,[80*80 size(Tr_dataMatrix,3)]);
Tt_dataMatrix=reshape(Tt_dataMatrix,[165 120 size(Tt_dataMatrix,2)]);
Tt_dataMatrix=imresize(Tt_dataMatrix,[80 80]);
Tt_dataMatrix=reshape(Tt_dataMatrix,[80*80 size(Tt_dataMatrix,3)]);

%% expression
tt_dat_exp = []; ttls_exp = [];
for ci = 1:80
    tt_dat_exp = [tt_dat_exp Tr_dataMatrix(:,2+7*(ci-1):4+7*(ci-1))];
    ttls_exp   = [ttls_exp repmat(ci,[1 3])];
end

%% illumination
tt_dat_ill = []; ttls_ill = [];
for ci = 1:80
    tt_dat_ill = [tt_dat_ill Tr_dataMatrix(:,5+7*(ci-1):7+7*(ci-1))];
    ttls_ill   = [ttls_ill repmat(ci,[1 3])];
end
clear Tr_dataMatrix Tr_sampleLabels Tt_dataMatrix Tt_sampleLabels;

%% variation dictionary
for vi=1:length(tv_dat)
   tv_dat{vi}=double(tv_dat{vi});
end

switch database 
    case 'illumination'
        tt_dat=tt_dat_ill;
        ttls=ttls_ill;
    case 'expression'        
        tt_dat=tt_dat_exp;
        ttls=ttls_exp;        
    case 'scarf'  
        tt_dat=tt_dat_scarf;
        ttls=ttls_scarf;            
    case 'glass'  
        tt_dat=tt_dat_sunglass;
        ttls=ttls_sunglass;   
    case 'disguise'  
        tt_dat=tt_dat_dis;
        ttls=ttls_dis;            
    case 'disill'  
        tt_dat=tt_dat_dil;
        ttls=ttls_dil;        
end


tr_dat=double(tr_dat);
tt_dat=double(tt_dat);
td_dat=double(td_dat);

tr_dat       =    tr_dat./ repmat(sqrt(sum(tr_dat.*tr_dat)),[size(tr_dat,1) 1]); % unit norm 2
tt_dat       =    tt_dat./ repmat(sqrt(sum(tt_dat.*tt_dat)),[size(tt_dat,1) 1]); % unit norm 2
tr_dat=reshape(tr_dat,[80 80 size(tr_dat,2)]);
tt_dat=reshape(tt_dat,[80 80 size(tt_dat,2)]);

Image_row_NUM=size(tr_dat,1);
Image_column_NUM=size(tr_dat,2);
ps=20;
pm=10;
pw=1;
kappa=0.005;
Rec=LGR(tr_dat,tt_dat,td_dat,tv_dat,trls,ttls,Image_row_NUM,Image_column_NUM,ps,pm,pw,kappa);