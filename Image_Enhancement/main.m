close all
F=imread('lena512.bmp');
%%%%%%%%%%%%%%%%%First part of P1%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% imhist(F);
% figure
% title("Original")

G=min(max(round(imadd(immultiply(F, 0.2), 50)),0),255); %Lowering contrast
% imhist(G)
% figure

%to view the image in lower contrast)
% imagesc(GG,[0 150])
% figure
% imagesc(GG)
% 
% imhist(histeq(F,255))
% figure
% 
% counts=imhist(F);
% p_r = counts./(512*512) .* 255;
% T_r = round(cumsum(p_r));
% s_k = zeros(256,1);
% for i = 1:256
%     s_k(T_r(i)+1) = s_k(T_r(i)+1) + counts(i);
% end
% 
% bar([0:255],s_k);


%%%%%%%%%%%%%%%%%%%Second part of P1%%%%%%%%%%%%%%%%%%%%%%%%%%5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Gaussian%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = mynoisegen('gaussian', 512, 512, 0, 64);
im_gaus=uint8(min(max((round(double(F) + n)),0),255));
% imhist(im_gaus);
% figure
% imhist(F);


%mean filter
% meanfilter=1/9.*[1 1 1; 1 1 1 ; 1 1 1];
% imgausmean=conv2(im_gaus,meanfilter,'same');
% imhist(uint8(imgausmean))
% % figure
% % imhist(F)


%Median Filter
% imgausmed=medfilt2(im_gaus);
% imhist(imgausmed)
% figure
% imshow(im_gaus)
% figure
% imhist(F)



%%%%%%%%%%%%%%%%%%%%%%%Salt&pepper%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
im_saltp = F;
n = mynoisegen('saltpepper', 512, 512, .05, .05);
im_saltp(n==0) = 0;
im_saltp(n==1) = 255;
% imhist(im_saltp);
% figure
% imhist(F);


%mean filter
meanfilter=1/9.*[1 1 1; 1 1 1 ; 1 1 1];
imsaltpepper=conv2(im_saltp,meanfilter,'same');
% imhist(uint8(imsaltpepper))
% figure
% imshow(im_saltp)
% figure
% imshow(F)


%Median Filter
% saltpmed=medfilt2(im_saltp);
% imhist(saltpmed)
% figure
% imhist(im_saltp)
% figure
% imhist(F)


%Part 3 of P1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 h = myblurgen('gaussian', 8);
 g = min(max(round(conv2(F, h, 'same')), 0), 255);
%  GG=log(fft2(g));
%  imagesc(abs(fftshift(GG)));
%  figure
%  FF=log(fft2(F));
%  imagesc(abs(fftshift(FF)));
%  figure
% %comment
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Before degradation we 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NoiseVar=0.0833; % data from the project definition 
GGRow=g(:);
GGVar=var(double(GGRow)); 
GGPWR=(sum(double(GGRow).*double(GGRow)))/(512*512);
K=NoiseVar/GGPWR;

FFTh=fft2(h,529,529);
%Shifting to center
[x,y] = meshgrid(0:512+17-1);
xy0 = (17-1)/2;
FFTHm = exp((1j*2*pi)/(529)*xy0*(x+y));
H = FFTh .* FFTHm;
%Wiener's Filter
FFTh_conj=conj(H);
FFTh_abs=abs(H .* FFTh_conj);
Restore=FFTh_conj ./ (FFTh_abs+0.01);

% tapering the picture in order to cancel the edge effect
Corner=fspecial('gaussian',50,10);
g_t=edgetaper(g,Corner);
FFTg_t=fft2(g_t,529,529); 
%using the filter we have calculated before in order to restore the original FFT
Restoring=Restore .* FFTg_t;
image = uint8(ifft2(Restoring)) ;
% Running a median filter in order to reduce the noise 
Restored= medfilt2(image);
Restored=Restored(1:512, 1:512);
% imshow(Restored,[]);
% figure
% imshow();


%Blurred man
% F2=imread('man512_outoffocus.bmp')
% 
% NoiseVar=0.0833; % data from the project definition 
% GGRow=F2(:);
% GGVar=var(double(GGRow)); 
% GGPWR=(sum(double(GGRow).*double(GGRow)))/(512*512);
% K=NoiseVar/GGPWR;
% 
% FFTh=fft2(h,529,529);
% %Shifting to center
% [x,y] = meshgrid(0:512+17-1);
% xy0 = (17-1)/2;
% FFTHm = exp((1j*2*pi)/(529)*xy0*(x+y));
% H = FFTh .* FFTHm;
% %Wiener's Filter
% FFTh_conj=conj(H);
% FFTh_abs=abs(H .* FFTh_conj);
% Restore=FFTh_conj ./ (FFTh_abs+K);
% 
% % tapering the picture in order to cancel the edge effect
% Corner=fspecial('gaussian',50,10);
% g_t2=edgetaper(F2,Corner);
% FFTg_t=fft2(g_t2,529,529); 
% %using the filter we have calculated before in order to restore the original FFT
% Restoring=Restore .* FFTg_t;
% image = uint8(ifft2(Restoring)) ;
% % Running a median filter in order to reduce the noise 
% Restored= medfilt2(image);
% Restored=Restored(1:512, 1:512);
% imshow(Restored,[]);
% figure
% imshow(F2);


%Boat
F3=imread('boats512_outoffocus.bmp');

NoiseVar=0.0833; % data from the project definition 
GGRow=F3(:);
GGVar=var(double(GGRow)); 
GGPWR=(sum(double(GGRow).*double(GGRow)))/(512*512);
K=NoiseVar/GGPWR;

FFTh=fft2(h,529,529);
%Shifting to center
[x,y] = meshgrid(0:512+17-1);
xy0 = (17-1)/2;
FFTHm = exp((1j*2*pi)/(529)*xy0*(x+y));
H = FFTh .* FFTHm;
%Wiener's Filter
FFTh_conj=conj(H);
FFTh_abs=abs(H .* FFTh_conj);
Restore=FFTh_conj ./ (FFTh_abs+K);

% tapering the picture in order to cancel the edge effect
Corner=fspecial('gaussian',50,10);
g_t3=edgetaper(F3,Corner);
FFTg_t=fft2(g_t3,529,529); 
%using the filter we have calculated before in order to restore the original FFT
Restoring=Restore .* FFTg_t;
image = uint8(ifft2(Restoring)) ;
% Running a median filter in order to reduce the noise 
Restored= medfilt2(image);
Restored=Restored(1:512, 1:512);
imshow(Restored,[]);
figure
imshow(F3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%