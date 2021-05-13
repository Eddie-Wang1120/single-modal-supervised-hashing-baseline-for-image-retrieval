 function[num,MAP] = EvaPreK(kkk,traingnd,testgnd,H,tH)
B = compactbit(H);
tB = compactbit(tH);

hammTrainTest = hammingDist(tB, B)';

[~, HammingRank]=sort(hammTrainTest,2);
num = cat_precal(kkk,traingnd,testgnd,HammingRank');
MAP = cat_apcal(traingnd,testgnd,HammingRank');

 end