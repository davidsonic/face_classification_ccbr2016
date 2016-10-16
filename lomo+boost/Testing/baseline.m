load('lomo_boost.mat','model');
load feat_test.mat;
save_dir='lomo_score.txt';
[Fx,~]=TestGAB(model,feat_test);
dlmwrite(save_dir,Fx);