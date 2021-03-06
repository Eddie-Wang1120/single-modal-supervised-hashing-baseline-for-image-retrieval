%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Single-Modal Supervised Hashing Methods for Large-Scale Image 
% Retrival Codes Collection
% Version 1.1
% Updated on 5/13/2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this is the main script to start the demo.
% you can get the mean Average Precision (mAP) curves and Time curves 
% from 10 different supervised hashing methods.
% the script uses CIFAR10-Gist512 as database.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author:
%       Jin-Heng Wang
%       github:Eddie-Wang1120
%       e-mail:wangjinheng1120@163.com
%
%       Yu-Wei Zhan
%       e-mail:zhanyuweilif@gmail.com
%
%       Xingbo Liu
%       e-mail:sclxb@mail.sdu.edu.cn
%
%       Xin Luo
%       e-mail:luoxin@sdu.edu.cn
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the papers for these methods are all provided on my github project 
% introduction.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all; clear all;

addpath('./utils/');
load('./CIFAR-10.mat');
db_name = 'CIFAR10';

exp_data.traindata = traindata;
exp_data.traingnd = traingnd;
exp_data.testdata = testdata;
exp_data.testgnd = testgnd;
exp_data.cateTrainTest = cateTrainTest;

% input the hashmethod name
% must use capital letters
hashmethods = {'COSDISH','FSDH','FSSH','KSH','LFH','POSH','RSLH','SCDH','SDH','SSLH'};

nhmethods = length(hashmethods);

% input the bit number
loopnbits = [4,8,16,32,64,128];
nbits = length(loopnbits);

MAPres = ones(nhmethods, nbits);
TIMEres = ones(nhmethods, nbits);

for i = 1:nhmethods
    K = hashmethods{i};
    fprintf('......%s start...... \n', K);
    for j = 1:nbits
        bits = loopnbits(j);
        fprintf('Bits:%d\n', bits)
        [MAP,TIME] = demo(exp_data, bits, hashmethods{i});
        fprintf('MAP:%.6f\n',MAP);
        fprintf('TIME:%.6f\n',TIME);
        MAPres(i,j) = MAP;
        TIMEres(i,j) = TIME;
    end
end


%% plot drawing

line_width = 2;
marker_size = 8;
xy_font_size = 14;
legend_font_size = 12;
linewidth = 1.6;
title_font_size = xy_font_size;

% show MAP
figure('Color', [1 1 1]); hold on;
for i = 1: nhmethods
    map = MAPres(i, :);
    bits = loopnbits;
    
    p = plot(bits, map);
    color=gen_color(i);
    marker=gen_marker(i);
    set(p,'Color', color);
    set(p,'Marker', marker);
    set(p,'LineWidth', line_width);
    set(p,'MarkerSize', marker_size);
end

h1 = xlabel('Number of bits');
h2 = ylabel('mean Average Precision (mAP)');
title(db_name, 'FontSize', title_font_size);
set(h1, 'FontSize', xy_font_size);
set(h2, 'FontSize', xy_font_size);
axis square;
set(gca, 'xtick', bits);
set(gca, 'XtickLabel', bits);
set(gca, 'linewidth', linewidth);
hleg = legend(hashmethods);
set(hleg, 'FontSize', legend_font_size);
set(hleg, 'Location', 'best');
box on;
grid on;
hold off;

% show TIME
figure('Color', [1 1 1]); hold on;
for i = 1: nhmethods
    time = TIMEres(i, :);
    bits = loopnbits;
    
    p = plot(bits, time);
    color=gen_color(i);
    marker=gen_marker(i);
    set(p,'Color', color);
    set(p,'Marker', marker);
    set(p,'LineWidth', line_width);
    set(p,'MarkerSize', marker_size);
end

h1 = xlabel('Number of bits');
h2 = ylabel('Time');
title(db_name, 'FontSize', title_font_size);
set(h1, 'FontSize', xy_font_size);
set(h2, 'FontSize', xy_font_size);
axis square;
set(gca, 'xtick', bits);
set(gca, 'XtickLabel', bits);
set(gca, 'linewidth', linewidth);
hleg = legend(hashmethods);
set(hleg, 'FontSize', legend_font_size);
set(hleg, 'Location', 'best');
box on;
grid on;
hold off;

