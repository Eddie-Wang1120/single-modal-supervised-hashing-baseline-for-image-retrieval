 function[num,MAP] = EvaPreK(kkk,traingnd,testgnd,H,tH)
B = compactbit(H);
tB = compactbit(tH);

hammTrainTest = hammingDist(tB, B)';
% hash lookup: precision and reall
% Ret = (hammTrainTest <= hammRadius+0.00001);
% [Pre, Rec] = evaluate_macro(cateTrainTest, Ret)

% hamming ranking: MAP
[~, HammingRank]=sort(hammTrainTest,2);
num = cat_precal(kkk,traingnd,testgnd,HammingRank');
MAP = cat_apcal(traingnd,testgnd,HammingRank');

 end