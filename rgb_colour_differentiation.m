clc;
clear;
close all;

% Read image
img = imread('colors.jpg');

% Separate RGB channels
R = img(:,:,1);
G = img(:,:,2);
B = img(:,:,3);

% Find red, green and blue pixels
red = (R > G) & (R > B) & (R > 100);
green = (G > R) & (G > B) & (G > 100);
blue = (B > R) & (B > G) & (B > 100);

% Display results
figure;

subplot(2,2,1);
imshow(img);
title('Original Image');

subplot(2,2,2);
imshow(red);
title('Red Pixels');

subplot(2,2,3);
imshow(green);
title('Green Pixels');

subplot(2,2,4);
imshow(blue);
title('Blue Pixels');