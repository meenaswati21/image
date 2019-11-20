clc;
  close all;
 
  
  img = imread('E:\ip codes\project\test1.jpg'); %grabbing an image
  B = imresize(img, [400 300]); %resizing it
  G = rgb2gray(B);   %converting it into gray
  BW = edge(G,'canny',0.6); %finding the edges
  se = ones(25,25);
  IM2 = imdilate(BW,se); %dilating the image
  
  
  subplot(1,3,1); %plotting orignal resized image
  imshow(B);
  title('Resized orignal image');
  
  subplot(1,3,2);  %plotting edge detected image
  imshow(BW);
  title('Edge detection');
  
  subplot(1,3,3);  %plotting dilated image
  imshow(IM2);
  title('Dilated image');