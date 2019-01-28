%Description:
%This script is supposed to classify the patches generated for the camera
%model identification

%author: Simone Milani (simone.milani@dei.unipd.it) 
%date: 30/11/2017
%license: This project is released under the GNU Public License.
%

clear all;
close all;
close all hidden;

close all;
clear all;
%define path to useful functions
if ispc
	addpath('func_arc');
else
	addpath('./func_arc');
end;

%name of the camera models to be identified
vet_prefix={ 'Canon_Ixus70' 'Kodak_M1063' 'Nikon_CoolPixS710' 'Casio_EX-Z150' ...
    'FujiFilm_FinePixJ50' 'Nikon_D200' 'Nikon_D70s' }

%directory where dataset is to be found
str_dir='C:\\Users\\Riccardo\\Desktop\\DFProject\\Photos\\SmallTemporary\\';

%parameters to be used
Nclasses=7;
th_val1=0;
th_val2=10000;

%organize the generated data to fit to MATLAB CNN functions
label_train=[];
ntot_train=0;
label_test=[];
ntot_test=0;
%loop on classes
for c=1:Nclasses
    
    %prepare training data
    
    %load files
    eval(sprintf('load feat_cam_trainC%d res vet_tot_files Img',c));
    n=size(Img,4);
    
    %select good patches
    ind=select_patches(Img,th_val1);
    
    n=length(ind);
    
    ntot_train=ntot_train+n;  %count the total number of training patches
    label_train=[ label_train (ones(1,n)*c) ];  %select the labels
    
    %prepare test data
    
    eval(sprintf('load feat_cam_testC%d res vet_tot_files Img',c));
    
    %select images
    
    ind=select_patches(Img,th_val1);
    
    n=length(ind);
    
    ntot_test=ntot_test+n;  %count teh total number
    label_test=[ label_test (ones(1,n)*c) ];  %select labels
end;

%create two arrays of images to be used with MATLAB Neural Toolbox
Itrain=zeros(64,64,3,ntot_train);
Itest=zeros(64,64,3,ntot_test);
Ntrain=ntot_train;
Ntest=ntot_test;


ntot_train=0;
ntot_test=0;
for c=1:Nclasses
    
    %prepare training data
    
    %load data
    eval(sprintf('load feat_cam_trainC%d res vet_tot_files Img',c));

    ind=select_patches(Img,th_val1);
    
    n=length(ind);
    
    %save patches in the struct Itrain
    Itrain(:,:,:,ntot_train+1:ntot_train+n)=Img(:,:,:,ind);
    ntot_train=ntot_train+n;
    
    
    %prepare testing (evaluation) data
    
    %load data
    eval(sprintf('load feat_cam_testC%d res vet_tot_files Img',c));

    
    ind=select_patches(Img,th_val1);
    
    n=length(ind);
    
    %save the patches of the Test set in Itest
    Itest(:,:,:,ntot_test+1:ntot_test+n)=Img(:,:,:,ind);
    ntot_test=ntot_test+n;
end;


Ten_train=Itrain;
Ten_test=Itest;


labels=categorical(label_train);

label_t=categorical(label_test);

%%
%This part creates the CNN and do the classification
%%

%parameters for CNN
miniBatchSize = 256;
numValidationsPerEpoch = 2;
validationFrequency = floor(size(Itrain,4)/miniBatchSize/numValidationsPerEpoch);

%options for CNN
%for more information go to the link 
%https://it.mathworks.com/help/nnet/ref/trainingoptions.html?requestedDomain=www.mathworks.com
options = trainingOptions('sgdm',...
    'ExecutionEnvironment','gpu',...
     'LearnRateSchedule','piecewise',...
     'InitialLearnRate',.005,...
    'LearnRateDropFactor',0.2,...
    'LearnRateDropPeriod',20,...
    'MaxEpochs',20,...
    'ValidationData',{Ten_test,label_t'},...
    'ValidationFrequency',30,...
    'VerboseFrequency',30,...
    'Verbose',true,...
    'Plots','training-progress');


%structure of the CNN (list of layers)
layers = [
    imageInputLayer([64 64 3])

    convolution2dLayer(7,16,'Padding',3)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(7,32,'Padding',3)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(7,64,'Padding',3)
    batchNormalizationLayer
    reluLayer
    
%     maxPooling2dLayer(2,'Stride',2)
% 
%     convolution2dLayer(7,128,'Padding',3)
%     batchNormalizationLayer
%     reluLayer
    
%     maxPooling2dLayer(2,'Stride',2)
% 
%     convolution2dLayer(7,256,'Padding',3)
%     batchNormalizationLayer
%     reluLayer

    fullyConnectedLayer(Nclasses)
    softmaxLayer
    classificationLayer];
%%
%end of parameters for CNN
%%

%%
%train and test the network
%%

 
rng('default') % For reproducibility
net = trainNetwork(Ten_train,labels',layers,options);  %train
eval(sprintf('save net net'));
label_out = classify(net,Ten_test,'ExecutionEnvironment','gpu',...
    'MiniBatchSize',miniBatchSize);  %classify evaluatio test

accuracy = sum(label_out(:) == label_t(:))/numel(label_t)  %average accuracy

%Confusion matrix
conf_mat=zeros(Nclasses,Nclasses);  
for c1=1:Nclasses
    ind1=find(double(label_t(:))==c1);  %select values that belong to c1 classes
    for c2=1:Nclasses
        ind2=find(double(label_out(:))==c2);  %elements classified as c2
        ind=intersect(ind1,ind2);  %c1 elements classified as c2
        conf_mat(c1,c2)=length(ind);  %save the number
    end;
end;
%normalize the numbers into percentages
conf_mat=conf_mat./(sum(conf_mat,2)*ones(1,Nclasses))*100;
disp(conf_mat);  %visualize confusion matrix
