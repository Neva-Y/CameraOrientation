% Name: Xing Yang Goh
% ID: 1001969
clc
clear
close all
tic

img1 = rgb2gray(imread("file1.png"));
img2 = rgb2gray(imread("file2.png"));

%% Part 1
%SIFT takes scale spaces to the next level. You take the original image, and generate progressively blurred out images. Then, you resize the original image to half size. And you generate blurred out images again. And you keep repeating.
%Gaussian Blur
SIFTfeatures1_1 = detectSIFTFeatures(img1, NumLayersInOctave=1);
SIFTfeatures2_1 = detectSIFTFeatures(img2, NumLayersInOctave=1);

figure();
imshow(img1);
hold on;
plot(SIFTfeatures1_1.selectStrongest(10))
title("Image 1, 10 Points, 1 Layer in Octave")

figure();
imshow(img2);
hold on;
plot(SIFTfeatures2_1.selectStrongest(10))
title("Image 2, 10 Points, 1 Layer in Octave")

% Increase number of layers in octave to 5, higher number of layers means
% that larger features in the image are identified since an increased
% Gaussian blur is applied to the image, which will lead to smoothing
% of the noise. This is an important step for the Difference of Gaussians
% to find out interesting key points such as corners due to the sensitivity
% of the Laplacian of Gaussian to noise.
SIFTfeatures1_5 = detectSIFTFeatures(img1, NumLayersInOctave=5);
SIFTfeatures2_5 = detectSIFTFeatures(img2, NumLayersInOctave=5);

figure();
imshow(img1);
hold on;
plot(SIFTfeatures1_5.selectStrongest(10))
title("Image 1, 10 Points, 5 Layer in Octave")

figure();
imshow(img2);
hold on;
plot(SIFTfeatures2_5.selectStrongest(10))
title("Image 2, 10 Points, 5 Layer in Octave")

%% Part 2
SIFTfeatures1_3 = detectSIFTFeatures(img1, NumLayersInOctave=3);
SIFTfeatures2_3 = detectSIFTFeatures(img2, NumLayersInOctave=3);

% Create the feature vectors for the images using the SIFT points identified
% with 3 layers in the octaves
[features1_3, valid_points1] = extractFeatures(img1, SIFTfeatures1_3);
[features2_3, valid_points2] = extractFeatures(img2, SIFTfeatures2_3);

% Perform L2 normalisation to the feature vectors
normFeatures1_3 = vecnorm(features1_3, 2, 2);
normFeatures2_3 = vecnorm(features2_3, 2, 2);

processedFeatures1_3 = [];
for i = 1:length(features1_3)
    processedFeatures1_3(i,:) = features1_3(i,:)/normFeatures1_3(i);
end

processedFeatures2_3 = [];
for i = 1:length(features2_3)
    processedFeatures2_3(i,:) = features2_3(i,:)/normFeatures2_3(i);
end

% Find the corresponding feature index from image 1 to image 2 with the
% lowest sum of square distance (SSD)
image1match2 = zeros(length(processedFeatures1_3), 1);
for i = 1:length(processedFeatures1_3)
    lowestSSD = inf;
    for j = 1:length(processedFeatures2_3)
        if sum((processedFeatures1_3(i,:) - processedFeatures2_3(j,:)).^2 ) <  lowestSSD
            lowestSSD = sum((processedFeatures1_3(i,:) - processedFeatures2_3(j,:)).^2 );
            image1match2(i) = j;
        end
    end
end

% Find the corresponding feature index from image 2 to image 1 with the
% lowest sum of square distance
image2match1 = zeros(length(processedFeatures2_3), 1);
for i = 1:length(processedFeatures2_3)
    lowestSSD = inf;
    for j = 1:length(processedFeatures1_3)
        if sum((processedFeatures2_3(i,:) - processedFeatures1_3(j,:)).^2 ) <  lowestSSD
            lowestSSD = sum((processedFeatures2_3(i,:) - processedFeatures1_3(j,:)).^2 );
            image2match1(i) = j;
        end
    end
end

% Compare the two corresponding lowest SSD matches and find points that are
% uniquely match to each other (lowest SSD corresponding to each other)
uniqueMatch1to2 = [];
uniqueCount = 0;
for i = 1:length(image1match2)
    if image2match1(image1match2(i)) == i
        uniqueCount = uniqueCount + 1;
        uniqueMatch1to2(uniqueCount, 1) = i;
        uniqueMatch1to2(uniqueCount, 2) = image1match2(i);
    end
end

% Using the matchFeature function with SSD metric to find unique matches
uniqueMatch1to2_true = matchFeatures(processedFeatures1_3, processedFeatures2_3, 'Unique', true, 'MatchThreshold', 100, 'MaxRatio', 1.0, 'Metric', 'SSD');

% Estimate the geometric transformation between the two images by mapping 
% the inliers of the matched points between the two images with a
% projective transformation since there is only a change in the POV of the
% observer, preserving the incidence and cross-ratio between the images.
matchedPoints1 = valid_points1(uniqueMatch1to2(:,1),:);
matchedPoints2 = valid_points2(uniqueMatch1to2(:,2),:);
[tform, inlier] = estimateGeometricTransform2D(matchedPoints1,matchedPoints2,'projective');

matchedPoints1_true = valid_points1(uniqueMatch1to2_true(:,1),:);
matchedPoints2_true = valid_points2(uniqueMatch1to2_true(:,2),:);
[tform_true, inlier_true] = estimateGeometricTransform2D(matchedPoints1_true,matchedPoints2_true,'projective');

% Plot all the matched points
figure(); 
showMatchedFeatures(img1,img2,matchedPoints1,matchedPoints2);
title("Unique correspondences of all points using manual matching")

% Plot the inlier matched points that are used to estimate the
% geometric tranformation
matchedPoints1_inlier = matchedPoints1(inlier,:);
matchedPoints2_inllier = matchedPoints2(inlier,:);
figure(); 
showMatchedFeatures(img1,img2,matchedPoints1_inlier,matchedPoints2_inllier);
title("Unique correspondences of inliers using manual matching")

matchedPoints1_true_inlier = matchedPoints1_true(inlier_true,:);
matchedPoints2_true_inllier = matchedPoints2_true(inlier_true,:);
figure();
showMatchedFeatures(img1,img2,matchedPoints1_true_inlier,matchedPoints2_true_inllier);
title("Unique correspondences of inliers using matchFeatures")

% Use the provided parameters of the focal length and optical centre, along
% with the image size to estimate the camera's intrinsic parameters
imageSize = [size(img1, 1), size(img1, 2)];
intrinsics = cameraIntrinsics([517.3 516.5],[318.6 255.3],[480 640]);
cameraParams = cameraParameters('IntrinsicMatrix', intrinsics.IntrinsicMatrix);

% Find the orientation and location of the camera between the two images
% using the geometric transformation and the camera intrinsics parameters
[relativeOrientation,relativeLocation] = relativeCameraPose(tform,cameraParams,matchedPoints1,matchedPoints2);


toc