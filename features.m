clc
clear
close all
tic

img1 = rgb2gray(imread("file1.png"));
img2 = rgb2gray(imread("file2.png"));

%% Part 1
SIFTfeatures1_1 = detectSIFTFeatures(img1, NumLayersInOctave=2);
SIFTfeatures2_1 = detectSIFTFeatures(img2, NumLayersInOctave=2);

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

%%% Increase number of layers in octave to 5, higher number of layers means
%%% that larger featuresin the image are identified
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

[features1_3, valid_points1] = extractFeatures(img1, SIFTfeatures1_3);
[features2_3, valid_points2] = extractFeatures(img2, SIFTfeatures2_3);


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

uniqueMatch1to2 = [];
uniqueCount = 0;
for i = 1:length(image1match2)
    if image2match1(image1match2(i)) == i
        uniqueCount = uniqueCount + 1;
        uniqueMatch1to2(uniqueCount, 1) = i;
        uniqueMatch1to2(uniqueCount, 2) = image1match2(i);
    end
end

uniqueMatch1to2_true = matchFeatures(processedFeatures1_3, processedFeatures2_3, 'Unique', true, 'MatchThreshold', 100, 'MaxRatio', 1.0, 'Metric', 'SSD');

matchedPoints1 = valid_points1(uniqueMatch1to2(:,1),:);
matchedPoints2 = valid_points2(uniqueMatch1to2(:,2),:);
figure(); 
showMatchedFeatures(img1,img2,matchedPoints1,matchedPoints2,'montage');
title("Unique correspondences using manual matching")


matchedPoints1_true = valid_points1(uniqueMatch1to2_true(:,1),:);
matchedPoints2_true = valid_points2(uniqueMatch1to2_true(:,2),:);
figure(); 
showMatchedFeatures(img1,img2,matchedPoints1_true,matchedPoints2_true,'montage');
title("Unique correspondences using matchFeatures")

tform = estimateGeometricTransform2D(matchedPoints1_true,matchedPoints2_true,'projective')

imageSize = [size(img1, 1), size(img1, 2)];
intrinsics = cameraIntrinsics([517.3 516.5],[318.6 255.3],imageSize)
cameraParams = cameraParameters('IntrinsicMatrix', intrinsics.IntrinsicMatrix)
[relativeOrientation,relativeLocation] = relativeCameraPose(tform,cameraParams,matchedPoints1_true,matchedPoints2_true)


toc