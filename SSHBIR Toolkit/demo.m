%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the demo for start the hashing methods respectively
% don't modify unless add methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [MAP,TIME] = demo(exp_data, bits, method)

switch(method)
    % COSDISH hashing
    case 'COSDISH'
        addpath('./COSDISH/');
        [MAP,TIME] = demo_COSDISH(exp_data, bits);
        
    % LFH hashing
    case 'LFH'
        addpath('./LFH/');
        [MAP,TIME] = demo_LFH(exp_data, bits);
    
    % SDH hashing
    case 'SDH'
        addpath('./SDH/');
        [MAP,TIME] = demo_SDH(exp_data, bits);
        
    % FSSH hashing
    case 'FSSH'
        addpath('./FSSH/');
        [MAP,TIME] = demo_FSSH(exp_data, bits);
    
    % FSDH hashing
    case 'FSDH'
        addpath('./FSDH/');
        [MAP,TIME] = demo_FSDH(exp_data, bits);
    
    % SCDH hashing
    case 'SCDH'
        addpath('./SCDH/');
        [MAP,TIME] = demo_SCDH(exp_data, bits);
        
    % POSH hashing
    case 'POSH'
        addpath('./POSH/');
        [MAP,TIME] = demo_POSH(exp_data, bits);
        
    % SSLH hashing
    case 'SSLH'
        addpath('./SSLH/');
        [MAP,TIME] = demo_SSLH(exp_data, bits);
        
    % RSLH hashing
    case 'RSLH'
        addpath('./RSLH/');
        [MAP,TIME] = demo_RSLH(exp_data, bits);

                
    % KSH hashing
    case 'KSH'
        addpath('./KSH/');
        [MAP,TIME] = demo_KSH(exp_data, bits);


end
