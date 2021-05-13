function [ap] = cal_map(traingnd,testgnd, IX)
% average precision (AP) calculation 

[numtrain, numtest] = size(IX);

apall = zeros(1,numtest);
for i = 1 : numtest
    y = IX(:,i);
    x=0;
    p=0;
    new_label = traingnd * testgnd(i,:)';
    new_label(new_label>0) = 1;
    
    num_return_NN = numtrain;
    for j=1:num_return_NN
        if new_label(y(j))==1
            x=x+1;
            p=p+x/j;
        end
    end  
    if p==0
        apall(i)=0;
    else
        apall(i)=p/x;
    end
    
    
end

ap = mean(apall);
