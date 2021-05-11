 function [ap] = cat_precal(kkk,traingnd,testgnd, IX)
% ap=apcal(score,label)
% average precision (AP) calculation 


% IX=HammingRank;
[numtrain, numtest] = size(IX);

apall = zeros(1,numtest);
for i = 1 : numtest
    y = IX(:,i);
    x=0;

    new_label=zeros(1,numtrain);
%     new_label(traingnd==testgnd(i))=1;


    new_label(sum(traingnd .* (repmat(testgnd(i,:),numtrain,1)),2)>0)=1;
 
    num_return_NN = kkk;%5000; % only compute MAP on returned top 5000 neighbours.
    for j=1:num_return_NN
        if new_label(y(j))==1
            x=x+1;
        end
    end  
        apall(i)=x/kkk;    
end

ap = mean(apall);
  end