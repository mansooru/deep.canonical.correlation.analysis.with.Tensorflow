%% Simulated data
clear all; clc;
% Setting initial information
n = 5000;
p = 1024;
q = 1024;

max_zero = 150;
max_block =80;
u0_block = sb_vector(p,max_zero,max_block);

if length(u0_gt) > p
    u0_gt(q+1:end)=[];
end

max_zero = 150;
max_block = 80;
v0_gt = sb_vector(q,max_zero,max_block);

if length(v0_gt) > q
    v0_gt(q+1:end)=[];
end

stem(v0_gt);
figure; stem(u0_gt);

v0_gt = v0_gt>0;
u0_gt = u0_gt>0;
    
% Generate X and Y
n = 5000;
z =  zscore(rand(n,1));
noise_level = ceil(rand*1);
input_X1  =z*u0_gt + randn(n,q) .*rand(n,q)*4;
label_X1= u0_gt';

input_X2 = z*v0_gt + randn(n,q).*rand(n,q)*4;
label_X2 = v0_gt';

input_X1 = zscore(input_X1);
input_X2 = zscore(input_X2);
H1 = get_connectivity(input_X1,2);
H2 = get_connectivity(input_X2,2);

save('train_X1_dataset.ver5.mat','input_X1','label_X1','H1')
save('train_X2_dataset.ver5.mat','input_X2','label_X2','H2')

clear input_X1 input_X2 label_X1 label_X2 i
% Generate X and Y
n = 1000;
z1 =  zscore(rand(n,1));
input_X1 = z1*u0_gt + randn(n,q) .*rand(n,q)*3;
label_X1 = u0_gt';

input_X2 = z1*v0_gt + randn(n,q).*rand(n,q)*3;
label_X2 = v0_gt';


input_X1 = zscore(input_X1);
input_X2 = zscore(input_X2);
save('test_X1_dataset.ver5.mat','input_X1','label_X1')
save('test_X2_dataset.ver5.mat','input_X2','label_X2')

























