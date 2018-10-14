clc;
clear all;
close all;

a=imread('C:\Users\sana\Desktop\blurry.jpg');
img= im2double(a);
imshow(a);
title('input image');
im1 = img(:,:,1);
im2 = img(:,:,2);
im3 = img(:,:,3);
[m, n] = size(im1);
M        = zeros(m, m);
N        = zeros(n, n);

for u = 0 : (m - 1)
    for x = 0 : (m - 1)
        M(u+1, x+1) = exp(-2 * pi * 1i / m * x * u);
    end    
end

for v = 0 : (n - 1)
    for y = 0 : (n - 1)
        N(y+1, v+1) = exp(-2 * pi * 1i / n * y * v);
    end    
end

DFT1= M * im1 * N;
DFT_S1=fftshift(DFT1);
mag_DFT1=100*log(1+abs(DFT_S1));
% figure;
% imagesc(100*log(1+abs(DFT_S1))); colormap(gray); 
% title('magnitude spectrum of R channel');
% figure;
% imagesc(angle(DFT1));  colormap(gray);
% title('phase spectrum of R channel');

DFT2 = M * im2 * N;
DFT_S2=fftshift(DFT2);
mag_DFT2=100*log(1+abs(DFT_S2));
 
% figure;
% imagesc(100*log(1+abs(DFT_S2))); colormap(gray);
% title('magnitude spectum of G channel');
% figure;
% imagesc(angle(DFT2));  colormap(gray);
% title('phase spectrum of G channel');
%  
DFT3 = M * im3 * N;
DFT_S3=fftshift(DFT3);
mag_DFT3=100*log(1+abs(DFT_S3));
% figure;
% imagesc(mag_DFT3); colormap(gray); 
% title('magnitude spectrum of B channel');
% figure;
% imagesc(angle(DFT3));  colormap(gray); 
% title('phase spectrum of B channel');

% inverse dft
M1  = zeros(m, m);
N1   = zeros(n, n);
 
for x1 = 0 : (m - 1)

    for u1 = 0 : (m - 1)
    M1(x1+1, u1+1) = exp(2 * pi * 1i / m * x1 * u1);
   end    
 end
 
 
 for y1 = 0 : (n - 1)
     for v1 = 0 : (n - 1)
         N1(v1+1, y1+1) = exp(2 * pi * 1i / n * y1 * v1);
     end    
 end
 
F(:,:,1)= M1 *(DFT_S1* N1 /(m*n));
F(:,:,2)= M1 * (DFT_S2* N1 /(m*n));
F(:,:,3)= M1 * (DFT_S3* N1 /(m*n));
 imshow(F);
 title('output image');

 %KERNEL
 
 
kernel=imread('C:\Users\sana\Desktop\Kernel.png');
% imshow(kernel);
% title('input kernel');
k= im2double(kernel(1:80,1:80));
% figure,imshow(k);
% title('output kernel');
average=sum(sum(k));
ker_avg = k/average;
[r1, c1] = size(ker_avg);
img_pad=padarray(ker_avg,[(360),(360)],0,'both');    
%padding the image
[r, c]= size(img_pad);
% figure;imshow(img_pad);
% title('padded image');
M        = zeros(r, r);
N        = zeros(c, c);
for u = 0 : (r - 1)
    for x = 0 : (r - 1)
        M(u+1, x+1) = exp(-2 * pi * 1i / r * x * u);
    end    
end

for v = 0 : (c - 1)
    for y = 0 : (c - 1)
        N(y+1, v+1) = exp(-2 * pi * 1i / c * y * v);
    end    
end
Ker1 = M * img_pad * N;
Ker_S1=fftshift(Ker1);
mag_Ker1=100*log(1+abs(Ker_S1));

%  figure;
%  imagesc(mag_Ker1); colormap(gray); 
%  title('magnitude kernel 1');
%  figure;
%  imagesc(angle(Ker1));  colormap(gray); 
%  title('phase spectrum kernel 1');
 trans_1=DFT_S1./Ker_S1;
 trans_2=DFT_S2./Ker_S1;
 trans_3=DFT_S3./Ker_S1;
trans_S1=fftshift(trans_1);
trans_S2=fftshift(trans_2);
trans_S3=fftshift(trans_3);
mag_trans_1=100*log(1+abs(trans_S1));
% imagesc(mag_trans_1); colormap(gray); 
% title('magnitude of trans_1');
% figure;
% imagesc(angle(trans_1));  colormap(gray); 
% title('phase spectrum trans_1');
% mag_trans_2=100*log(1+abs(trans_S2));
% imagesc(mag_trans_2); colormap(gray); 
% title('magnitude of trans_2');
% figure;
% imagesc(angle(trans_2));  colormap(gray); 
% title('phase spectrum trans_2');
% mag_trans_3=100*log(1+abs(trans_S3));
% imagesc(mag_trans_3); colormap(gray); 
% title('magnitude of trans_3');
% figure;
% imagesc(angle(trans_3));  colormap(gray); 
% title('phase spectrum trans_3');

%IDFT OF KERNEL

wM1  = zeros(m, m);
wN1   = zeros(n, n);

for x1 = 0 : (m - 1)
    for u1 = 0 : (m - 1)
        wM1(x1+1, u1+1) = exp(2 * pi * 1i / m * x1 * u1);
    end    
end
for y1 = 0 : (n - 1)
    for v1 = 0 : (n - 1)
        wN1(v1+1, y1+1) = exp(2 * pi * 1i / n * y1 * v1);
    end    
end
out(:,:,1) = M1 *trans_1 * N1/(r*c);
out_S(:,:,1) = ifftshift(out(:,:,1));
out(:,:,2) = M1 *trans_2 * N1/(r*c);
out_S(:,:,2) = ifftshift(out(:,:,2));
out(:,:,3) = M1 *trans_3 * N1/(r*c);
out_S(:,:,3) = ifftshift(out(:,:,3));
 figure;
 imshow(out_S);
 title('filtered output');

 
%designing the butterworth filter


p=inputdlg('Do');% cut-off frequencies
Do=str2num(p{1});

q=inputdlg('order');%order- order of the filter
order=str2num(q{1});

lpf=zeros(m,n);% low pass filter of image dimensions
dis=zeros(m,n);%distance matrix
for u=0: m-1
    for v=0:n-1
        dis(u+1,v+1) = ((u-m/2)^2 + (v-n/2)^2)^0.5;
        lpf(u+1,v+1)=1/(1+dis(u+1,v+1)/Do)^(2*order);
    end
end
dft1=trans_1.*lpf;
dft2=trans_2.*lpf;
dft3=trans_3.*lpf;
[m,n]=size(dft1);%all dft have thesame size.
M=zeros(m);
N=zeros(n);
for x=0:(m-1)
    for u=0:(m-1)
        M(x+1,u+1)=exp(i*2*pi*u*x/m);
    end
end
for y=0:n-1
    for v=0:n-1
        N(y+1,v+1)=exp(i*2*pi*v*y/n);
 
    end
end
idft1(:,:,1)=1/(m*n)*M*dft1*N;
idft(:,:,1)=abs(idft1(:,:,1));
idft(:,:,1)=ifftshift(idft(:,:,1));


idft2(:,:,2)=1/(m*n)*M*dft2*N;
idft(:,:,2)=abs(idft2(:,:,2));
idft(:,:,2)=ifftshift(idft(:,:,2));

idft3(:,:,3)=1/(m*n)*M*dft3*N;
idft(:,:,3)=abs(idft3(:,:,3));
idft(:,:,3)=ifftshift(idft(:,:,3));

 figure;imshow(idft);
 title('butterworth filtered output');
 
org_img=imread('C:\Users\sana\Desktop\GroundTruth.jpg');
   
error1 = (double(org_img(:,:,1)) - double(idft(:,:,1))) .^ 2;
error2 = (double(org_img(:,:,2)) - double(idft(:,:,2))) .^ 2;
error3 = (double(org_img(:,:,3)) - double(idft(:,:,3))) .^ 2;

m_error1 = sum(sum(error1)) / (m *n );
m_error2 = sum(sum(error2)) / (m *n );
m_error3 = sum(sum(error3)) / (m *n );

% Average mean square error of channels.
mse = (m_error1 + m_error2 + m_error3)/3;

% Calculate PSNR.
PSNR = 10 * log10( 255^2 / mse);
disp(PSNR);

% Calculating SSIM.
idft_b=im2double(idft);
org_imgb=im2double(org_img);
morg_img=mean(mean(mean(org_imgb, 1), 2), 3);
mout_img=mean(mean(mean(idft_b, 1), 2), 3);
one=ones(m,n);
varorg_img=mean(mean(mean((org_imgb - morg_img.*one).^ 2,1),2),3);
varout_img=mean(mean(mean((idft_b - mout_img.*one).^ 2,1),2),3);
cross=mean(mean(mean((org_imgb - morg_img.*one).*(idft_b - mout_img.*one),1),2),3);
stdev_org=varorg_img^0.5;
stdev_out=varout_img^0.5;
T1=1;
T2=0.01;
T3=0.01;
l=(2*morg_img*mout_img+T1)/(morg_img^2+mout_img^2+T1);
c=(2*stdev_org*stdev_out+T2)/(varorg_img+varout_img+T2);
s=(cross+T3)/(stdev_org*stdev_out + T3);
ssim=l*c*s;
disp(ssim);




 
%design of weiner filter


%kernel conjugate
con= conj(Ker_S1);
%magnitude of the kernel
mag = mag_Ker1.*mag_Ker1;

%taking value of C
b=inputdlg('Z');
C=str2num(b{1});
L=C.*ones(m,n);
WEIN1= (con./(L+mag)).*DFT_S1;
WEIN2 = (con./(L+mag)).*DFT_S2;
WEIN3 = (con./(L+mag)).*DFT_S3;

WEIN_S1=fftshift(WEIN1);
WEIN_S2=fftshift(WEIN2);
WEIN_S3=fftshift(WEIN3);

mag_WEIN1=100*log(1+abs(WEIN_S1));
% figure;
% imagesc(mag_WEIN1); colormap(gray); 
%  title('magnitude of WEIN_S1');
% figure;
% imagesc(angle(WEIN1));  colormap(gray); 
% title('phase spectrum of WEIN1');

%inverse dft
M1  = zeros(m, m);
N1   = zeros(n, n);

for x1 = 0 : (m - 1)
    for u1 = 0 : (m - 1)
        M1(x1+1, u1+1) = exp(2 * pi * 1i / m * x1 * u1);
    end    
end
for y1 = 0 : (n - 1)
    for v1 = 0 : (n - 1)
        N1(v1+1, y1+1) = exp(2 * pi * 1i / n * y1 * v1);
    end    
end
out(:,:,1) = M1 * WEIN_S1 * N1/(m*n);
out(:,:,1)=real(out(:,:,1));
out(:,:,1) = ifftshift(out(:,:,1)).*10000;
out(:,:,2) = M1 * WEIN_S2* N1/(m*n);
out(:,:,2)=real(out(:,:,2));
out(:,:,2) = ifftshift(out(:,:,2)).*10000;
out(:,:,3) = M1 * WEIN_S3* N1/(m*n);
out(:,:,3)=real(out(:,:,3));
out(:,:,3) = ifftshift(out(:,:,3)).*10000;
 figure;
 imshow(out);
 title(' WEINER filtered output');
 
werror1 = (double(org_img(:,:,1)) - double(out(:,:,1))) .^ 2;
werror2 = (double(org_img(:,:,2)) - double(out(:,:,2))) .^ 2;
werror3 = (double(org_img(:,:,3)) - double(out(:,:,3))) .^ 2;

wm_error1 = sum(sum(werror1)) / (m *n );
wm_error2 = sum(sum(werror2)) / (m *n );
wm_error3 = sum(sum(werror3)) / (m *n );

% Average mean square error of R, G, B.
wmse = (wm_error1 + wm_error2 + wm_error3)/3;

% Calculate PSNR (Peak Signal to noise ratio).
wPSNR = 10 * log10( 255^2 / wmse);
disp(wPSNR);


% Calculating SSIM.
widft_b=im2double(out);

wmorg_img=mean(mean(mean(org_imgb, 1), 2), 3);
wmout_img=mean(mean(mean(widft_b, 1), 2), 3);
one=ones(m,n);
wvarorg_img=mean(mean(mean((org_imgb - wmorg_img.*one).^ 2,1),2),3);
wvarout_img=mean(mean(mean((widft_b - wmout_img.*one).^ 2,1),2),3);
wcross=mean(mean(mean((org_imgb - wmorg_img.*one).*(widft_b - wmout_img.*one),1),2),3);
wstdev_org=wvarorg_img^0.5;
wstdev_out=wvarout_img^0.5;
T1=1;
T2=0.01;
T3=0.01;
wl=(2*wmorg_img*wmout_img+T1)/(wmorg_img^2+wmout_img^2+T1);
wc=(2*wstdev_org*wstdev_out+T2)/(wvarorg_img+wvarout_img+T2);
ws=(wcross+T3)/(wstdev_org*wstdev_out + T3);
wssim=wl*wc*ws;
disp(wssim);




%design of constrained least squares filter


gamma=inputdlg('Value of gamma');
E=str2num(gamma{1});
L=E.*ones(m,n);
%defining the laplacian operator
j= [0 -1 0 0; -1 4 -1 0; 0 -1  0 0; 0 0 0 0];
[rj, cj] = size(j);
j_pad = padarray(j,[((m-rj)/2),((n-cj)/2)],0,'both');    %padding the image
[rj, cj]= size(j_pad);


Mj  = zeros(rj, rj);
Nj  = zeros(cj, cj);

for uj = 0 : (rj - 1)
    for xj = 0 : (rj- 1)
        Mj(uj+1, xj+1) = exp(-2 * pi * 1i / rj * xj * uj);
    end    
 end

for vj= 0 : (cj - 1)
  for yj = 0 : (cj - 1)
         N(yj+1, vj+1) = exp(-2 * pi * 1i / cj * yj * vj);
     end    
end

J = Mj * j_pad * Nj;
O= fft2(j_pad);
J_S1=fftshift(O);
mag_0=abs(J(u,v));
mag_j=mag_0.*mag_0;
CLS1= (con./(mag+(L.*mag_j))).*DFT_S1;
CLS2= (con./(mag+(L.*mag_j))).*DFT_S2;
CLS3 = (con./(mag+(L.*mag_j))).*DFT_S3;
CLS_S1=fftshift(CLS1);
CLS_S2=fftshift(CLS2);
CLS_S3=fftshift(CLS3);
mag_CLS1=100*log(1+abs(CLS_S1));
%figure;
%imagesc(mag_CLS1); colormap(gray); 
%title('magnitude of CLS_S1');
%figure;
%imagesc(angle(CLS1));  colormap(gray); 
%title('phase spectrum CLS1');

%inverse dft
M1  = zeros(m, m);
N1   = zeros(n, n);

for x1 = 0 : (m - 1)
    for u1 = 0 : (m - 1)
        M1(x1+1, u1+1) = exp(2 * pi * 1i / m * x1 * u1);
    end    
end
for y1 = 0 : (n - 1)
    for v1 = 0 : (n - 1)
        N1(v1+1, y1+1) = exp(2 * pi * 1i / n * y1 * v1);
    end    
end
lout(:,:,1) = M1 * CLS1 * wN1/(m*n);
lout(:,:,1)=real(lout(:,:,1));
lout(:,:,1) = ifftshift(lout(:,:,1)).*10000;

lout(:,:,2) = M1 * CLS2 * N1/(m*n) ;
lout(:,:,2)=real(lout(:,:,2));
lout(:,:,2) = ifftshift(lout(:,:,2)).*10000;

lout(:,:,3) = M1 * CLS3 * N1/(m*n);
lout(:,:,3)=real(lout(:,:,3));
lout(:,:,3) = ifftshift(lout(:,:,3)).*10000;

figure;
imshow(lout);
title('constrained least squares filter');


lerror1 = (double(org_img(:,:,1)) - double(lout(:,:,1))) .^ 2;
lerror2 = (double(org_img(:,:,2)) - double(lout(:,:,2))) .^ 2;
lerror3 = (double(org_img(:,:,3)) - double(lout(:,:,3))) .^ 2;

lm_error1 = sum(sum(lerror1)) / (m *n );
lm_error2 = sum(sum(lerror2)) / (m *n );
lm_error3 = sum(sum(lerror3)) / (m *n );

% Average mean square error of channels.
lmse = (lm_error1 + lm_error2 + lm_error3)/3;

% Calculate PSNR (Peak Signal to noise ratio).
lPSNR = 10 * log10( 255^2 / lmse);
disp(lPSNR);



% Calculating SSIM.
cidft_b=im2double(lout);

cmorg_img=mean(mean(mean(org_imgb, 1), 2), 3);
cmout_img=mean(mean(mean(cidft_b, 1), 2), 3);
one=ones(m,n);
cvarorg_img=mean(mean(mean((org_imgb - cmorg_img.*one).^ 2,1),2),3);
cvarout_img=mean(mean(mean((cidft_b - cmout_img.*one).^ 2,1),2),3);
ccross=mean(mean(mean((org_imgb - cmorg_img.*one).*(cidft_b - cmout_img.*one),1),2),3);
cstdev_org=cvarorg_img^0.5;
cstdev_out=cvarout_img^0.5;
T1=0.0001;
T2=0.0009;
T3=T2/2;
cl=(2*cmorg_img*cmout_img+T1)/(cmorg_img^2+cmout_img^2+T1);
cc=(2*cstdev_org*cstdev_out+T2)/(cvarorg_img+cvarout_img+T2);
cs=(ccross+T3)/(cstdev_org*cstdev_out + T3);
cssim=cl*cc*cs;
disp(cssim);