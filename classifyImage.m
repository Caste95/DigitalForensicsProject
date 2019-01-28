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
    'FujiFilm_FinePixJ50' 'Nikon_D200' 'Nikon_D70s' };

%parameters to be used
szp=64;  %size of the patches processed by CFA detection
th_val1=50;  %threshold minimum variance of patch

res=[];  %matrix of feature arrays

    %images to be processed by CNN
    Img=zeros(szp,szp,3,10);

    res=[];

        %read images
        I0=imread('Casio_EX-Z150_0_5165.jpg');
        [N0,M0,ch0]=size(I0);  %size of the image

        R=I0(:,:,1);  %R component
        G=I0(:,:,2);  %G component
        B=I0(:,:,3);  %B component

        Rc=ROFdenoise(R,14);  %denoise components
        Gc=ROFdenoise(G,14);
        Bc=ROFdenoise(B,14);

        %create difference image between original and denoised image
        D0=I0;
        D0(:,:,1)=double(R)-Rc;
        D0(:,:,2)=double(G)-Gc;
        D0(:,:,3)=double(B)-Bc;

        %for every image create 10 szbxszb patches to be classified

        cpatch=1;

        while cpatch <= 10

            %select (x,y) randomly inside each image (coordinates of the upper
            %left pixel in the pixel)
            x=randi(M0-szp,1,1);
            y=randi(N0-szp,1,1);

            %select the patch
            I=I0(y:y+szp-1,x:x+szp-1,:);
            D=D0(y:y+szp-1,x:x+szp-1,:);

            %components for each patch (block)
            rb=R(y:y+szp-1,x:x+szp-1);
            gb=G(y:y+szp-1,x:x+szp-1);
            bb=B(y:y+szp-1,x:x+szp-1);

            %compute variances of the signal in the patch (block)
            vr=var(double(rb(:)));
            vg=var(double(gb(:)));
            vb=var(double(bb(:)));

            %average variance
            avg_v=(vb+vr+vg)/3;

            %this operation is required in order to avoid using patches with
            %low or high variance (not significant in the camera model
            %identification)
            if (avg_v<th_val1)
                continue;
            end;

            %save the features in the structure Img
            Img(:,:,:,cpatch)=D;

            fprintf('Patch %d complete\n', cpatch);
            cpatch=cpatch+1; %increase counter of the patch
        end;

load net.mat
miniBatchSize = 256;
label_out = classify(net,Img,'ExecutionEnvironment','gpu',...
    'MiniBatchSize',miniBatchSize)
A = mode(double(label_out));
sorted = sort(label_out);
B = sorted(ceil(length(label_out)/2));
C = round(mean(double(label_out)));
fprintf('Value of mode: %d ==> %s\n',A,vet_prefix{A});
fprintf('Value of median: %d ==> %s\n',B,vet_prefix{B});
fprintf('Value of mean (rounded): %d ==> %s\n',C,vet_prefix{C});