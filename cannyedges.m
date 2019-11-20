clc;
clear all;
close all;

i=imread('E:\ip codes\project\test1.jpg');

img=rgb2gray(i);
lpf = ones(5,5)/25;
% filtered_img = imfilter(img, lpf);

edge_detected = edge(img,'canny');
imshow(edge_detected);