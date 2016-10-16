function model = TrainDetector(faceDBFile, nonfaceDBFile, outFile, options)
%% function model = TrainDetector(faceDBFile, nonfaceDBFile, outFile, options)
% Train a Nomalized Pixel Difference (NPD) based face detector.
%
% Input:
%   <faceDBFile>: MAT file for the face images. It contains an array FaceDB
%   of size [objSize, objSize, numFaces].
%   <nonfaceDBFile>: MAT file for the nonface images.It contains the
%   following variables:
%       numSamples: the number of cropped nonface images of size [objSize,
%       objSize].
%       numNonfaceImgs: the number of big nonface images for bootstrapping.
%       NonfaceDB: an array of size [objSize, objSize, numSamples] 
%           containing the cropped nonface images. This is used in the 
%           begining stages of the detector training.
%       NonfaceImages: a cell of size [numNonfaceImgs, 1] containing the
%       big nonface images for bootstrapping.
%   <outFile>: the output file to store the training result.
%   [optioins]: optional parameters. See the beginning codes of this function
%    for the parameter meanings and the default values.
%
% Output:
%   model: output of the trained detector.
% 
% Example:
%     See TrainDetector_Demo.m.

objSize = 20; % size of the face detection template
numFaces = Inf; % the number of face samples to be used for training.
% Inf means to use all face samples.
negRatio = 1; % factor of bootstrap nonface samples. For example, negRatio=2 
% means bootstrapping two times of nonface samples w.r.t face samples.
finalNegs = 1000; % the minimal number of bootstrapped nonface samples. 
% The training will be stopped if there is no enough nonface samples in the
% final stage. This is also to avoid overfitting.
numThreads = 24; % the number of computing threads for bootstrapping

if nargin >= 3 && ~isempty(options)
    if isfield(options,'objSize') && ~isempty(options.objSize)
        objSize = options.objSize;
    end
    if isfield(options,'numFaces') && ~isempty(options.numFaces)
        numFaces = options.numFaces;
    end
    if isfield(options,'negRatio') && ~isempty(options.negRatio)
        negRatio = options.negRatio;
    end
    if isfield(options,'finalNegs') && ~isempty(options.finalNegs)
        finalNegs = options.finalNegs;
    end
    if isfield(options,'numThreads') && ~isempty(options.numThreads)
        numThreads = options.numThreads;
    end
    if isfield(options,'boostOpt') && ~isempty(options.boostOpt)
        boostOpt = options.boostOpt;
    end
end

boostOpt.numThreads = numThreads;

load(faceDBFile, 'FaceDB');
load(nonfaceDBFile, 'numNonfaceImgs', 'numSamples');

numFaces = min(numFaces, size(FaceDB, 4)); %#ok<NODEF>
if numFaces < size(FaceDB, 4)
    index = randperm(size(FaceDB, 4));
    FaceDB = FaceDB(:,:,:,index(1:numFaces));
end

numNegs = ceil(numFaces * negRatio);

fprintf('Extract face features\n');


faceFea=LOMO(FaceDB);
faceFea=faceFea';
faceFea=im2uint8(faceFea);
clear FaceDB;

if exist(outFile, 'file')
    load(outFile, 'model',  'trainTime', 'numStages', 'negIndex', 'numGridFace', 'numSlideFace');
else
    trainTime = 0;
    numStages = 0;
    model = [];
    negIndex = 1 : numSamples;
    numGridFace = ( rand(numNonfaceImgs, 1) + 1 ) * 1e6;
    numSlideFace = ( rand(numNonfaceImgs, 1) + 1 ) * 1e6;
end

NonfaceDB = [];
NonfaceImages = [];

fprintf('Start to train detector.\n');
T = length(model);

while true
    t0 = tic;
    
    [nonfaceFea, NonfaceDB, NonfaceImages, negIndex, numGridFace, numSlideFace] = ...
        BootstrapNonfaces(model, nonfaceDBFile, NonfaceDB, NonfaceImages, objSize, numNegs, negIndex, numGridFace, numSlideFace, numThreads);
    
    if size(nonfaceFea,1) < finalNegs
        fprintf('\n\nNo enough negative examples to bootstrap (nNeg=%d). The detector training is terminated.\n', size(nonfaceFea,1));
        trainTime = trainTime + toc(t0);
        fprintf('\nTraining time: %.0fs.\n', trainTime);
        break;
    end
    
    if size(nonfaceFea,1) == numNegs
        model = LearnGAB(faceFea, nonfaceFea, model, boostOpt);
    else
        NonfaceDB = [];
        NonfaceImages = [];
        boostOpt2 = boostOpt;
        boostOpt2.minNegRatio = finalNegs / size(nonfaceFea,1);
        model = LearnGAB(faceFea, nonfaceFea, model, boostOpt2);
    end
    
    clear nonfaceFea
    
    if length(model) == T
        fprintf('\n\nNo effective features for further detector learning.\n');
        break;
    end
    
    T = length(model);
    
    numStages = numStages + 1;
    trainTime = trainTime + toc(t0);

    try
        save(outFile, 'model',  'objSize', 'options', 'negRatio', 'numStages', 'trainTime', 'negIndex', 'numGridFace', 'numSlideFace', '-v7.3');
    catch exception
        fprintf('%s\n', exception.message);
        filename = userpath;
        filename = [filename(1:end-1) '\npd_model.mat'];
        fprintf('Save the results in %s instead.\n', filename);
        save(filename, 'model', 'objSize', 'options', 'negRatio', 'numStages', 'trainTime', 'negIndex', 'numGridFace', 'numSlideFace', '-v7.3');
    end

    far = prod([model.far]);
    fprintf('\nStage %d, #Weaks: %d, FAR: %.2g, Training time: %.0fs, Time per stage: %.0fs, Time per weak: %.3fs.\n\n', ...
        numStages, T, far, trainTime, trainTime / numStages, trainTime / T);
    
    if far <= boostOpt.maxFAR || T == boostOpt.maxNumWeaks || exist('boostOpt2', 'var')
        fprintf('\n\nThe detector training is finished.\n');
        break;
    end
end

clear faceFea nonfaceFea NonfaceDB NonfaceImages

try
    save(outFile, '-v7.3');
catch exception
    fprintf('%s\n', exception.message);
    filename = userpath;
    filename = [filename(1:end-1) '\npd_model.mat'];
    fprintf('Save the results in %s instead.\n', filename);
    save(filename, '-v7.3');
end
end


function [nonfaceFea, NonfaceDB, NonfaceImages, negIndex, numGridFace, numSlideFace, nonfacePatches] = BootstrapNonfaces(model, ...
    nonfaceDBFile, NonfaceDB, NonfaceImages, objSize, numNegs, negIndex, numGridFace, numSlideFace, numThreads)

numLimit = floor(numNegs / 1000);
dispStep = floor(numNegs / 10);
dispCount = 0;

if ~isempty(negIndex) && isempty(NonfaceDB)
    fprintf('Load NonfaceDB... ');
    t1 = tic;
    load(nonfaceDBFile, 'NonfaceDB');
    fprintf('done. %.0f seconds.\n', toc(t1));
end

if isempty(model)
    numNonfaces = size(NonfaceDB, 4);
    index = randperm(numNonfaces);
    nonfacePatches = NonfaceDB(:,:,:,index(1:numNegs));
else
    nonfacePatches = zeros(objSize, objSize,3,numNegs, 'uint8');
    T = length(model);
    t0 = tic;
    
    if ~isempty(negIndex) && ~isempty(NonfaceDB)
        numValid = length(negIndex);   
        batchsize=100000;
        batchnum=length(negIndex)/batchsize;
        fprintf('batchnum: %d\n',batchnum);
        negIndex_group2=[];
        negIndex_group=[];
        if batchnum>=1
            negIndex_group=[];
            for i=1:batchnum
                fprintf('This is the %dth batch\n',i);
                batch_negIndex=negIndex(batchsize*(i-1)+1:batchsize*i);
                nonfaceFea2=im2uint8(LOMO(NonfaceDB(:,:,:,batch_negIndex)));
                [~,passCount]=TestGAB(model,nonfaceFea2');
                tmp=batch_negIndex(passCount==T);
                negIndex_group=[negIndex_group;tmp'];
                clear nonfaceFea2
            end
            negIndex=negIndex(batchsize*floor(batchnum)+1:end);
        end
        
        nonfaceFea2=im2uint8(LOMO(NonfaceDB(:,:,:,negIndex)));
        [~,passCount]=TestGAB(model,nonfaceFea2');
        negIndex_group2=negIndex(passCount==T);
        clear nonfaceFea2;
        
        negIndex=[negIndex_group;negIndex_group2']';
        n = length(negIndex);
        
        fprintf('+%g of %.3g NonfaceDB samples. total: %d of %d. Time: %.0f seconds.\n', n, numValid, min(n,numNegs), numNegs, toc(t0));
        index = negIndex;
        
        if n > numNegs
            idx = randperm(n);
            index = index( idx(1:numNegs) );
            n = numNegs;
        end
        
        nonfacePatches(:,:,:,1:n) = NonfaceDB(:,:,:,index);
    else
        n = 0;
    end

    if n<numNegs
        nonfacePatches(:,:,:,n+1:end)=[];
    end
end

fprintf('Extract nonface features... ');
t1 = tic;
nonfaceFea= LOMO(nonfacePatches);
nonfaceFea=nonfaceFea';
nonfaceFea=im2uint8(nonfaceFea);
fprintf('done. %.4f seconds.\n', toc(t1));
end
