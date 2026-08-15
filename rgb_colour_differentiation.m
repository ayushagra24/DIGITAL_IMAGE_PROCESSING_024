clc;
clear;
close all;

img = imread('colors.jpg');

R = img(:,:,1);
G = img(:,:,2);
B = img(:,:,3);

red_img = zeros(size(img), 'uint8');
green_img = zeros(size(img), 'uint8');
blue_img = zeros(size(img), 'uint8');

red_img(:,:,1) = R;
green_img(:,:,2) = G;
blue_img(:,:,3) = B;

figure;

subplot(4,1,1);
imshow(img);
title('Original Image');

subplot(4,1,2);
imshow(red_img);
title('Red Channel');

subplot(4,1,3);
imshow(green_img);
title('Green Channel');

subplot(4,1,4);
imshow(blue_img);
title('Blue Channel');
