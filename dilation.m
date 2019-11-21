clc;
close all;
 
  
  img = imread('example.jpg'); %grabbing an image
  subplot(1,3,1); %plotting orignal resized image
  imshow(img);
  G = rgb2gray(img); 
  subplot(1,3,2); %plotting orignal resized image
  imshow(G);
  se = ones(2,2);
  IM2 = imdilate(G,se); %dilating the image
  
  subplot(1,3,3);  %plotting dilated image
  imshow(IM2);
  title('Dilated image');