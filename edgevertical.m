clc;
clear all;
close all;

i=imread('E:\ip codes\project\test1.jpg');
img=rgb2gray(i);
[m, n]= size(img);

sig=1;
x = floor(-3*sig):ceil(3*sig);
G = exp(-0.5*x.^2/sig^2);
G = G/sum(G);
dG = -x.*G/sig^2;
img = imfilter(img,dG);

vertical=[-1 0 1 ; -2 0 2; -1 0 1];

% for i=2:m-1
%     for j=2:n-1
%          o= img(i-1:i+1, j-1: j+1).*vertical;
%         d = sum(o(:));
%       gy(i-1,j-1)=d;
%     end
% end
gy = imfilter(img,vertical);
figure;
subplot(2,2,1),imshow(i),title('original image');
subplot(2,2,2),imshow(gy),title('vertical edges');


horizontal=[1 2 1 ; 0 0 0; -1 -2 -1];
gx = imfilter(img,horizontal);
% for i=2:m-1
%     for j=2:n-1
%          o= img(i-1:i+1, j-1: j+1).*horizontal;
%         d = sum(o(:));
%       gx(i-1,j-1)=d;
%     end
% end

subplot(2,2,3),imshow(gx),title('horizontal edges of pothole');
gx1 = abs(gx);
gy1 = abs(gy);
g = gx1+gy1;

nms_g = imhmax(g,1);

subplot(2,2,4),imshow(nms_g),title('all edges of pothole');