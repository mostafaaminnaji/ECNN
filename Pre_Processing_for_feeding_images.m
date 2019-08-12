% C)Mostafa Amin-Naji, Babol Noshirvani University of Technology,
% My Official Website: www.Amin-Naji.com
% My Email: Mostafa.Amin.Naji@Gmail.com

% PLEASE CITE THE BELOW PAPER IF YOU USE THIS CODE

% M. Amin-Naji, A. Aghagolzadeh, and M. Ezoji, “Ensemble of CNN for Multi-Focus Image Fusion”, Information Fusion, vol. 51, pp. 21–214, 2019. 
% DOI: https://doi.org/10.1016/j.inffus.2019.02.003





im1=imread('G:\mostafa\test\flower1.tif');
im2=imread('G:\mostafa\test\flower2.tif');


tic
im1=rgb2gray(im1);
im2=rgb2gray(im2);
[m,n]=size(im1);
out1=zeros(m,n,3);
[m,n]=size(im2);
out2=zeros(m,n,3);
    
[Gx1, Gy1] = imgradientxy(double(im1));
[Gx2, Gy2] = imgradientxy(double(im2));

out1(:,:,1)=im1;
out1(:,:,2)=Gx1;
out1(:,:,3)=Gy1;

out2(:,:,1)=im2;
out2(:,:,2)=Gx2;
out2(:,:,3)=Gy2;
toc
out1=uint8(out1);
out2=uint8(out2);

imwrite(out1,'G:\mostafa\test\feed_flower1.tif');
imwrite(out2,'G:\mostafa\test\feed_flower2.tif');


