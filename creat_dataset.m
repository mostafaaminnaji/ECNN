% C)Mostafa Amin-Naji, Babol Noshirvani University of Technology,
% My Official Website: www.Amin-Naji.com
% My Email: Mostafa.Amin.Naji@Gmail.com

% PLEASE CITE THE BELOW PAPER IF YOU USE THIS CODE

% M. Amin-Naji, A. Aghagolzadeh, and M. Ezoji, “Ensemble of CNN for Multi-Focus Image Fusion”, Information Fusion, vol. 51, pp. 21–214, 2019. 
% DOI: https://doi.org/10.1016/j.inffus.2019.02.003






clear
f=1;


for k=1:500
input_adress=strcat('G:\Mostafa Amin Naji\Mostafa\DATASet\','COCO (',num2str(k),').jpg');
    im{k}=(imread(input_adress));
     if size(im{k},3)==3
      im{k}=rgb2gray(im{k});
    end
    im_blur1{k}=imgaussfilt(im{k},9,'FilterSize',9);

    
    [m,n]=size(((im{k})));
    
        x0=im{k};
    x1=im_blur1{k};
    [Gx0, Gy0] = imgradientxy(x0);
    [Gx1, Gy1] = imgradientxy(x1);

    

    for i = 1:floor(m/32)
    for j = 1:floor(n/32)
        f=f+1;
        
        im_Block(:,:,1) = x0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,1) = x1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block(:,:,2) = Gx0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,2) = Gx1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block (:,:,3)= Gy0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,3) = Gy1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block=uint8(im_Block);
        im_Block_Blurred=uint8(im_Block_Blurred);

        focus_unfocus=vertcat(im_Block,im_Block_Blurred);
        unfocus_focus=vertcat(im_Block_Blurred,im_Block);



        
                
%         im{k}=(im{k}-min(min(im{k})))/(max(max(im{k}))-min(min(im{k})))*255;
%      im_blur{k}=(im_blur{k}-min(min(im_blur{k})))/(max(max(im_blur{k}))-min(min(im_blur{k})))*255;
    
    
    addres_of_image=strcat('G:\mostafa','\focus&unfocus7\train\focus_unfocus\','sharp_blur_ver1_',num2str(f),'.tif');
    imwrite(focus_unfocus,addres_of_image);
    
    addres_of_image=strcat('G:\mostafa','\focus&unfocus7\train\unfocus_focus\','sharp_blur_ver1_',num2str(f),'.tif');
    imwrite(unfocus_focus,addres_of_image);
        
    end
    end
k
end  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k=500:1000
input_adress=strcat('G:\Mostafa Amin Naji\Mostafa\DATASet\','COCO (',num2str(k),').jpg');
    im{k}=(imread(input_adress));
     if size(im{k},3)==3
      im{k}=rgb2gray(im{k});
    end
%       im{k}=(imread(input_adress));
      
    im_blur1{k}=imgaussfilt(im{k},9,'FilterSize',11);

    
    [m,n]=size(((im{k})));
    
        x0=im{k};
    x1=im_blur1{k};
    [Gx0, Gy0] = imgradientxy(x0);
    [Gx1, Gy1] = imgradientxy(x1);




    for i = 1:floor(m/32)
    for j = 1:floor(n/32)
        f=f+1;
        
        im_Block(:,:,1) = x0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,1) = x1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block(:,:,2) = Gx0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,2) = Gx1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block (:,:,3)= Gy0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,3) = Gy1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block=uint8(im_Block);
        im_Block_Blurred=uint8(im_Block_Blurred);

        focus_unfocus=vertcat(im_Block,im_Block_Blurred);
        unfocus_focus=vertcat(im_Block_Blurred,im_Block);


        
                
%         im{k}=(im{k}-min(min(im{k})))/(max(max(im{k}))-min(min(im{k})))*255;
%      im_blur{k}=(im_blur{k}-min(min(im_blur{k})))/(max(max(im_blur{k}))-min(min(im_blur{k})))*255;
    
    
    addres_of_image=strcat('G:\mostafa','\focus&unfocus7\train\focus_unfocus\','sharp_blur_ver1_',num2str(f),'.tif');
    imwrite(focus_unfocus,addres_of_image);
    
    addres_of_image=strcat('G:\mostafa','\focus&unfocus7\train\unfocus_focus\','sharp_blur_ver1_',num2str(f),'.tif');
    imwrite(unfocus_focus,addres_of_image);
        
    end
    end
k
end  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k=1001:1500
input_adress=strcat('G:\Mostafa Amin Naji\Mostafa\DATASet\','COCO (',num2str(k),').jpg');
    im{k}=(imread(input_adress));
     if size(im{k},3)==3
      im{k}=rgb2gray(im{k});
    end
    im_blur1{k}=imgaussfilt(im{k},9,'FilterSize',13);
    
    

    
    [m,n]=size(((im{k})));
    
        x0=im{k};
    x1=im_blur1{k};
    [Gx0, Gy0] = imgradientxy(x0);
    [Gx1, Gy1] = imgradientxy(x1);



    for i = 1:floor(m/32)
    for j = 1:floor(n/32)
        f=f+1;
        
        im_Block(:,:,1) = x0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,1) = x1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block(:,:,2) = Gx0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,2) = Gx1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block (:,:,3)= Gy0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,3) = Gy1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block=uint8(im_Block);
        im_Block_Blurred=uint8(im_Block_Blurred);

        focus_unfocus=vertcat(im_Block,im_Block_Blurred);
        unfocus_focus=vertcat(im_Block_Blurred,im_Block);



        
                
%         im{k}=(im{k}-min(min(im{k})))/(max(max(im{k}))-min(min(im{k})))*255;
%      im_blur{k}=(im_blur{k}-min(min(im_blur{k})))/(max(max(im_blur{k}))-min(min(im_blur{k})))*255;
    
    
    addres_of_image=strcat('G:\mostafa','\focus&unfocus7\train\focus_unfocus\','sharp_blur_ver1_',num2str(f),'.tif');
    imwrite(focus_unfocus,addres_of_image);
    
    addres_of_image=strcat('G:\mostafa','\focus&unfocus7\train\unfocus_focus\','sharp_blur_ver1_',num2str(f),'.tif');
    imwrite(unfocus_focus,addres_of_image);
        
    end
    end
k
end  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k=1501:2000
input_adress=strcat('G:\Mostafa Amin Naji\Mostafa\DATASet\','COCO (',num2str(k),').jpg');
    im{k}=(imread(input_adress));
     if size(im{k},3)==3
      im{k}=rgb2gray(im{k});
    end
    im_blur1{k}=imgaussfilt(im{k},9,'FilterSize',15);

    
    [m,n]=size(((im{k})));
    
        x0=im{k};
    x1=im_blur1{k};
    [Gx0, Gy0] = imgradientxy(x0);
    [Gx1, Gy1] = imgradientxy(x1);


    for i = 1:floor(m/32)
    for j = 1:floor(n/32)
        f=f+1;
        
        im_Block(:,:,1) = x0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,1) = x1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block(:,:,2) = Gx0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,2) = Gx1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block (:,:,3)= Gy0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,3) = Gy1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block=uint8(im_Block);
        im_Block_Blurred=uint8(im_Block_Blurred);

        focus_unfocus=vertcat(im_Block,im_Block_Blurred);
        unfocus_focus=vertcat(im_Block_Blurred,im_Block);



        
                
%         im{k}=(im{k}-min(min(im{k})))/(max(max(im{k}))-min(min(im{k})))*255;
%      im_blur{k}=(im_blur{k}-min(min(im_blur{k})))/(max(max(im_blur{k}))-min(min(im_blur{k})))*255;
    
    
    addres_of_image=strcat('G:\mostafa','\focus&unfocus7\train\focus_unfocus\','sharp_blur_ver1_',num2str(f),'.tif');
    imwrite(focus_unfocus,addres_of_image);
    
    addres_of_image=strcat('G:\mostafa','\focus&unfocus7\train\unfocus_focus\','sharp_blur_ver1_',num2str(f),'.tif');
    imwrite(unfocus_focus,addres_of_image);
        
    end
    end
k
end  








%valid

for k=2001:2300
input_adress=strcat('G:\Mostafa Amin Naji\Mostafa\DATASet\','COCO (',num2str(k),').jpg');
    im{k}=(imread(input_adress));
     if size(im{k},3)==3
      im{k}=rgb2gray(im{k});
    end
    im_blur1{k}=imgaussfilt(im{k},9,'FilterSize',9);

    
    [m,n]=size(((im{k})));
    
        x0=im{k};
    x1=im_blur1{k};
    [Gx0, Gy0] = imgradientxy(x0);
    [Gx1, Gy1] = imgradientxy(x1);

    

    for i = 1:floor(m/32)
    for j = 1:floor(n/32)
        f=f+1;
        
        im_Block(:,:,1) = x0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,1) = x1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block(:,:,2) = Gx0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,2) = Gx1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block (:,:,3)= Gy0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,3) = Gy1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block=uint8(im_Block);
        im_Block_Blurred=uint8(im_Block_Blurred);

        focus_unfocus=vertcat(im_Block,im_Block_Blurred);
        unfocus_focus=vertcat(im_Block_Blurred,im_Block);



                        
%         im{k}=(im{k}-min(min(im{k})))/(max(max(im{k}))-min(min(im{k})))*255;
%      im_blur{k}=(im_blur{k}-min(min(im_blur{k})))/(max(max(im_blur{k}))-min(min(im_blur{k})))*255;
    
    
    addres_of_image=strcat('G:\mostafa','\focus&unfocus7\test\focus_unfocus\','sharp_blur_ver1_',num2str(f),'.tif');
    imwrite(focus_unfocus,addres_of_image);
    
    addres_of_image=strcat('G:\mostafa','\focus&unfocus7\test\unfocus_focus\','sharp_blur_ver1_',num2str(f),'.tif');
    imwrite(unfocus_focus,addres_of_image);
        
    end
    end
k
end  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k=2301:2600
input_adress=strcat('G:\Mostafa Amin Naji\Mostafa\DATASet\','COCO (',num2str(k),').jpg');
    im{k}=(imread(input_adress));
     if size(im{k},3)==3
      im{k}=rgb2gray(im{k});
    end
    im_blur1{k}=imgaussfilt(im{k},9,'FilterSize',11);

    
    [m,n]=size(((im{k})));
    
        x0=im{k};
    x1=im_blur1{k};
    [Gx0, Gy0] = imgradientxy(x0);
    [Gx1, Gy1] = imgradientxy(x1);

    


    for i = 1:floor(m/32)
    for j = 1:floor(n/32)
        f=f+1;
        
        im_Block(:,:,1) = x0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,1) = x1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block(:,:,2) = Gx0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,2) = Gx1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block (:,:,3)= Gy0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,3) = Gy1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block=uint8(im_Block);
        im_Block_Blurred=uint8(im_Block_Blurred);

        focus_unfocus=vertcat(im_Block,im_Block_Blurred);
        unfocus_focus=vertcat(im_Block_Blurred,im_Block);



        
                
%         im{k}=(im{k}-min(min(im{k})))/(max(max(im{k}))-min(min(im{k})))*255;
%      im_blur{k}=(im_blur{k}-min(min(im_blur{k})))/(max(max(im_blur{k}))-min(min(im_blur{k})))*255;
    
    
    addres_of_image=strcat('G:\mostafa','\focus&unfocus7\test\focus_unfocus\','sharp_blur_ver1_',num2str(f),'.tif');
    imwrite(focus_unfocus,addres_of_image);
    
    addres_of_image=strcat('G:\mostafa','\focus&unfocus7\test\unfocus_focus\','sharp_blur_ver1_',num2str(f),'.tif');
    imwrite(unfocus_focus,addres_of_image);
        
    end
    end
k
end  
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k=2601:2900
input_adress=strcat('G:\Mostafa Amin Naji\Mostafa\DATASet\','COCO (',num2str(k),').jpg');
    im{k}=(imread(input_adress));
     if size(im{k},3)==3
      im{k}=rgb2gray(im{k});
    end

    im_blur1{k}=imgaussfilt(im{k},9,'FilterSize',13);

    
    [m,n]=size(((im{k})));
    
        x0=im{k};
    x1=im_blur1{k};
    [Gx0, Gy0] = imgradientxy(x0);
    [Gx1, Gy1] = imgradientxy(x1);

    

    for i = 1:floor(m/32)
    for j = 1:floor(n/32)
        f=f+1;
        
        im_Block(:,:,1) = x0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,1) = x1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block(:,:,2) = Gx0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,2) = Gx1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block (:,:,3)= Gy0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,3) = Gy1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block=uint8(im_Block);
        im_Block_Blurred=uint8(im_Block_Blurred);

        focus_unfocus=vertcat(im_Block,im_Block_Blurred);
        unfocus_focus=vertcat(im_Block_Blurred,im_Block);



        
                
%         im{k}=(im{k}-min(min(im{k})))/(max(max(im{k}))-min(min(im{k})))*255;
%      im_blur{k}=(im_blur{k}-min(min(im_blur{k})))/(max(max(im_blur{k}))-min(min(im_blur{k})))*255;
    
    
    addres_of_image=strcat('G:\mostafa','\focus&unfocus7\test\focus_unfocus\','sharp_blur_ver1_',num2str(f),'.tif');
    imwrite(focus_unfocus,addres_of_image);
    
    addres_of_image=strcat('G:\mostafa','\focus&unfocus7\test\unfocus_focus\','sharp_blur_ver1_',num2str(f),'.tif');
    imwrite(unfocus_focus,addres_of_image);
        
    end
    end
k
end  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k=2901:3200
input_adress=strcat('G:\Mostafa Amin Naji\Mostafa\DATASet\','COCO (',num2str(k),').jpg');
    im{k}=(imread(input_adress));
     if size(im{k},3)==3
      im{k}=rgb2gray(im{k});
    end

      
    im_blur1{k}=imgaussfilt(im{k},9,'FilterSize',15);

   
    
    [m,n]=size(((im{k})));
    
        x0=im{k};
    x1=im_blur1{k};
    [Gx0, Gy0] = imgradientxy(x0);
    [Gx1, Gy1] = imgradientxy(x1);



    for i = 1:floor(m/32)
    for j = 1:floor(n/32)
        f=f+1;
        
        im_Block(:,:,1) = x0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,1) = x1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block(:,:,2) = Gx0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,2) = Gx1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block (:,:,3)= Gy0(32*i-31:32*i,32*j-31:32*j);
        im_Block_Blurred(:,:,3) = Gy1(32*i-31:32*i,32*j-31:32*j);
        
        im_Block=uint8(im_Block);
        im_Block_Blurred=uint8(im_Block_Blurred);

        focus_unfocus=vertcat(im_Block,im_Block_Blurred);
        unfocus_focus=vertcat(im_Block_Blurred,im_Block);



        
                
%         im{k}=(im{k}-min(min(im{k})))/(max(max(im{k}))-min(min(im{k})))*255;
%      im_blur{k}=(im_blur{k}-min(min(im_blur{k})))/(max(max(im_blur{k}))-min(min(im_blur{k})))*255;
    
    
    addres_of_image=strcat('G:\mostafa','\focus&unfocus7\test\focus_unfocus\','sharp_blur_ver1_',num2str(f),'.tif');
    imwrite(focus_unfocus,addres_of_image);
    
    addres_of_image=strcat('G:\mostafa','\focus&unfocus7\test\unfocus_focus\','sharp_blur_ver1_',num2str(f),'.tif');
    imwrite(unfocus_focus,addres_of_image);
        
    end
    end
k
end  

