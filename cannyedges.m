clc;
clear all;
close all;

i=imread('test1.jpg');
img=rgb2gray(i);
edge_detected = edge(img,'canny');
imshow(edge_detected);
