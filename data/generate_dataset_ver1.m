%% Simulated data
clear all; clc;
% Setting initial information
n = 5000;
p = 1024;
q = 1024;

max_zero = 300;
max_block =40;
u0_gt = sb_vector(p,max_zero,max_block);

if length(u0_gt) > p
    u0_gt(q+1:end)=[];
end

max_zero = 300;
max_block = 40;
v0_gt = sb_vector(q,max_zero,max_block);

if length(v0_gt) > q
    v0_gt(q+1:end)=[];
end

stem(v0_gt);
figure; stem(u0_gt);

% Generate X and Y



max_zero = 100;
max_block =100;
u0_block = sb_vector(p,max_zero,max_block);

if length(u0_block) > p
    u0_block(q+1:end)=[];
end

max_zero = 100;
max_block = 100;
v0_block = sb_vector(q,max_zero,max_block);

if length(v0_block) > q
    v0_block(q+1:end)=[];
end

figure; stem(v0_block);
figure; stem(u0_block);

n = 5000;
z =  zscore(rand(n,1));
z1 =  zscore(rand(n,1));
z2 =  zscore(rand(n,1));

input_X1  =z*u0_gt + z1*u0_block + randn(n,q) .*rand(n,q)*10;
label_X1= u0_gt';
input_X2 = z*v0_gt + z2*v0_block  + randn(n,q).*rand(n,q)*10;
label_X2 = v0_gt';

% cor_X1 = corr(input_X1);
% cX1 = cor_X1 - diag(diag(cor_X1));
% 
% cor_X2 = corr(input_X2);
% cX2 = cor_X2 - diag(diag(cor_X2));
% 
% corr(input_X2,input_X1);

input_X1 = zscore(input_X1);
input_X2 = zscore(input_X2);
H1 = get_connectivity(input_X1,2);
H2 = get_connectivity(input_X2,2);

save('train_X1_dataset.ver5.mat','input_X1','label_X1','H1')
save('train_X2_dataset.ver5.mat','input_X2','label_X2','H2')










