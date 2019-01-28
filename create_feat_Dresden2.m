%Description:
%This script is supposed to generate the features sets for the identification of the camera
%
%author: Simone Milani (simone.milani@dei.unipd.it) 
%date: 30/11/2017
%license: This project is released under the GNU Public License.
%

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
str_dir='C:\\Users\\Riccardo\\Desktop\\DFProject\\Photos\\DatasetSeven\';

res=[];

vet_tot_files=[];

%parameters to be used
szp=64;  %size of the patches processed by CFA detection
th_val1=50;  %threshold minimum variance of patch
th_val2=100;   %threshold max values of patch

%cycle on each directory
for c=1:length(vet_prefix)
    
%find the image file names
if (~(ispc))
    eval(sprintf('vet_files=dir(''%s/%s*.JPG'')',str_dir,vet_prefix{c}));
else
    eval(sprintf('vet_files=dir(''%s/%s*.JPG'')',str_dir,vet_prefix{c}));
end;
%you can use also dir('*.JPG') depending on files

%total number of files
Nfiles=length(vet_files);  

%divide the total number between training and test
Ntrain=round(Nfiles*0.7);  %70 % images -> training
Ntest=Nfiles-Ntrain;  %30 % images -> test or evaluation

%randomly select which images belongs to training and which to test set
ind_files=randperm(Nfiles);
ind_filesTr=ind_files(1:Ntrain)+length(vet_tot_files);
ind_filesT=ind_files(Ntrain+1:Ntrain+Ntest)+length(vet_tot_files);

vet_tot_files=[ vet_tot_files ; vet_files ];

%%
%create training and test set
for ndir=1:2    %ndir: flags that signal whether you are considering test or training set
                %ndir=1: training; ndir=2: test
    
    res=[];  %matrix of feature arrays

    if ndir==1
        ind=ind_filesTr;  %training

    else
        ind=ind_filesT;  %test set

    end;

    %images to be processed by CNN
    Img=zeros(szp,szp,3,length(ind)*10);

    %iterate on each image
    cnt=1;
    res=[];
    for v=1:length(ind)   %loop on images

        %image name
        if ispc
            fname=sprintf('%s\\%s',str_dir,vet_tot_files(ind(v)).name);
        else
            fname=sprintf('%s/%s',str_dir,vet_tot_files(ind(v)).name);
        end;

        disp(fname);


        %read images
        I0=imread(fname);
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

            %compute eigenfilters features for the current block
            %to be used in a future implementation
            [N,M,ch]=size(I);
            psnr_vet=[ ];
            for bp=1:4
                for mth=1:15

                    [fa,fb,fc]=int_feat_diff(D,mth,bp,0); 

                    psnr_vet=[ psnr_vet fa' fb' fc' ];
                end;
            end;

            %save the features in the structure Img
            Img(:,:,:,cnt)=D;

            %save eigenfilters features in the matrix res
            res=[ res ; c v ind(v) cnt x y psnr_vet  ];

            cnt=cnt+1;
            cpatch=cpatch+1; %increase counter of the patch
    end;
                
        
    %save in a matrix where features array are placed along rows
    %structure of a row
    % index of camera | index of image | features by Milani | features by Gao (69)
    

    disp(v);
 
    end;
    
    %save the data in the .mat files
    if ndir==1
        eval(sprintf('save feat_cam_trainC%d res vet_tot_files Img',c));
    else
        eval(sprintf('save feat_cam_testC%d res vet_tot_files Img',c));
    end;
    

end;
    
end;

disp('Uff ... I finished!');


