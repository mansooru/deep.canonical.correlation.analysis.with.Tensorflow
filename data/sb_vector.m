function sb_vec = sb_vector(n, max_zero, max_block)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% input
% n = total length of the sparse signal
% max_zero = maximum size of continuous zeros
% max_block = maximum size of non-zero elements block
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
min_zero = ceil(0.1*max_zero); 
min_block = ceil(0.1*max_block); 

t=0;
sb_vec=[];
while t <= n
    x = rand; %generate a value between 0 to1
    v = 0;
    u = 0;
    if x <= 0.5 %you can change this condition depending upon your condition
        v = ceil(rand*max_zero); %generating a continuous zero series
        if  v < min_zero
            sb_vec = [sb_vec zeros(min_zero,1)'];
            v = min_zero;
        else
            sb_vec = [sb_vec zeros(v,1)'];
        end
    else
        u = ceil(rand*max_block); %to generate non zero element block
        if  u < min_block
            sb_vec=[sb_vec rand(min_block,1)'];
             u = min_block;
        else
            sb_vec=[sb_vec rand(u,1)'];
        end
        
        v=ceil(rand*max_zero); %to generate immediate zeros after a non zero block
        if v < min_zero
            sb_vec=[sb_vec zeros(min_zero,1)'];
             v = min_zero;
        else
            sb_vec=[sb_vec zeros(v,1)'];
        end
    end
    t = t+v+u;
end

