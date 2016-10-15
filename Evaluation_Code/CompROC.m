function [FAR, VR, posHist, negHist, threshold] = compROC( scoreMatrix, orientFlag, sampleMask, totalThresholds )

posDis = scoreMatrix( find( sampleMask == 1 ) );
negDis = scoreMatrix( find( sampleMask == -1 ) );
clear scoreMatrix sampleMask;

totalPosSamples = length(posDis);
totalNegSamples = length(negDis);

dmin = double( min( [posDis; negDis] ) );
dmax = double( max( [posDis; negDis] ) );

threshold = dmin + (dmax - dmin) * (1:totalThresholds) / totalThresholds;
[posHist,thr] = hist(posDis,threshold);
clear posDis;
[negHist,thr] = hist(negDis,threshold);
clear negDis;

if orientFlag == 0
    P = tril(ones(totalThresholds));
else
    P = triu(ones(totalThresholds));
end
    
FAR = ( P * negHist' ) / totalNegSamples;
VR  = ( P * posHist' ) / totalPosSamples;