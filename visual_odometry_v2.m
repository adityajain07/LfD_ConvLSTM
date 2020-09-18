clear
clc
close all

imageDir = '/Users/adityaj/Desktop/trajectory/reach1/data/';
images = imageDatastore(imageDir);

%% View the first image as a sample

% Create an empty imageviewset object to manage the data associated with each view.
vSet = imageviewset;

% Read and display the first image.
Irgb = readimage(images, 1);
player = vision.VideoPlayer('Position', [20, 400, 650, 510]);
step(player, Irgb);

%% Build the camera intrinsic matrix

data = load('camera_parameters_robot.mat');
cameraParams = data.cameraParams8;
intrinsics = cameraParams.Intrinsics;

%%

prevI = undistortImage(rgb2gray(Irgb), intrinsics); 

% Detect features. 
prevPoints = detectSURFFeatures(prevI, 'MetricThreshold', 200);

% subplot(1, 2, 1);
% imshow(prevI);
% hold on
% plot(prevPoints);
% hold off
% title('Original points');

% Select a subset of features, uniformly distributed throughout the image.
numPoints = 2000;
prevPoints = selectUniform(prevPoints, numPoints, size(prevI));

% subplot(1, 2, 2);
% imshow(prevI);
% hold on
% plot(prevPoints);
% hold off
% title('Uniform points');

% Extract features. Using 'Upright' features improves matching quality if 
% the camera motion involves little or no in-plane rotation.
prevFeatures = extractFeatures(prevI, prevPoints, 'Upright', true);

% Add the first view. Place the camera associated with the first view
% at the origin, oriented along the Z-axis.
viewId = 1;
vSet = addView(vSet, viewId, rigid3d(eye(3), [0 0 0]), 'Points', prevPoints);



% Setup axes.
figure
axis([-220, 50, -140, 20, -50, 300]);

% Set Y-axis to be vertical pointing down.
view(gca, 3);
set(gca, 'CameraUpVector', [0, -1, 0]);
camorbit(gca, -120, 0, 'data', [0, 1, 0]);

grid on
xlabel('X (cm)');
ylabel('Y (cm)');
zlabel('Z (cm)');
hold on

% Plot estimated camera pose. 
cameraSize = 7;
camPose = poses(vSet);
camEstimated = plotCamera(camPose, 'Size', cameraSize,...
    'Color', 'g', 'Opacity', 0);


% Initialize camera trajectories.
trajectoryEstimated = plot3(0, 0, 0, 'g-');

legend('Estimated Trajectory');
title('Camera Trajectory');


%% Estimate pose of second view
% Read and display the image.
viewId = 2;
player = vision.VideoPlayer('Position', [20, 400, 650, 510]);
Irgb = readimage(images, viewId);
step(player, Irgb);

% Convert to gray scale and undistort.
I = undistortImage(rgb2gray(Irgb), intrinsics);

% Match features between the previous and the current image.
[currPoints, currFeatures, indexPairs] = helperDetectAndMatchFeatures(...
    prevFeatures, I);

% figure
% imshow(prevI);
% hold on
% plot(currPoints);
% hold off
% title('Next Image Points Overlayed');

% Estimate the pose of the current view relative to the previous view.
[orient, loc, inlierIdx] = helperEstimateRelativePose(...
    prevPoints(indexPairs(:,1)), currPoints(indexPairs(:,2)), intrinsics);

% Exclude epipolar outliers.
indexPairs = indexPairs(inlierIdx, :);
    
% Add the current view to the view set.
vSet = addView(vSet, viewId, rigid3d(orient, loc), 'Points', currPoints);

% Store the point matches between the previous and the current views.
vSet = addConnection(vSet, viewId-1, viewId, 'Matches', indexPairs);


vSet = helperNormalizeOwn(vSet);

% Update the camera plots
camEstimated.AbsolutePose = poses(vSet).AbsolutePose(viewId);

% Plot the estimated trajectory.
locations = vertcat(poses(vSet).AbsolutePose.Translation);
set(trajectoryEstimated, 'XData', locations(:,1), 'YData', ...
    locations(:,2), 'ZData', locations(:,3));


prevI = I;
prevFeatures = currFeatures;
prevPoints   = currPoints;

%% Global adjustment for first initial views

for viewId = 3:29
    % Read and display the next image
    Irgb = readimage(images, viewId);
    step(player, Irgb);
    
    % Convert to gray scale and undistort.
    I = undistortImage(rgb2gray(Irgb), intrinsics);
    
    % Match points between the previous and the current image.
    [currPoints, currFeatures, indexPairs] = helperDetectAndMatchFeatures(...
        prevFeatures, I);
      
    % Eliminate outliers from feature matches.
    inlierIdx = helperFindEpipolarInliers(prevPoints(indexPairs(:,1)),...
        currPoints(indexPairs(:, 2)), intrinsics);
    indexPairs = indexPairs(inlierIdx, :);
    
    % Triangulate points from the previous two views, and find the 
    % corresponding points in the current view.
    [worldPoints, imagePoints] = helperFind3Dto2DCorrespondences(vSet,...
        intrinsics, indexPairs, currPoints);
    
    % Since RANSAC involves a stochastic process, it may sometimes not
    % reach the desired confidence level and exceed maximum number of
    % trials. Disable the warning when that happens since the outcomes are
    % still valid.
    warningstate = warning('off','vision:ransac:maxTrialsReached');
    
    % Estimate the world camera pose for the current view.
    [orient, loc] = estimateWorldCameraPose(imagePoints, worldPoints, intrinsics);
    
    % Restore the original warning state
    warning(warningstate)
    
    % Add the current view to the view set.
    vSet = addView(vSet, viewId, rigid3d(orient, loc), 'Points', currPoints);
    
    % Store the point matches between the previous and the current views.
    vSet = addConnection(vSet, viewId-1, viewId, 'Matches', indexPairs);    
    
    tracks = findTracks(vSet); % Find point tracks spanning multiple views.
        
    camPoses = poses(vSet);    % Get camera poses for all views.
    
    % Triangulate initial locations for the 3-D world points.
    xyzPoints = triangulateMultiview(tracks, camPoses, intrinsics);
    
    % Refine camera poses using bundle adjustment.
    [~, camPoses] = bundleAdjustment(xyzPoints, tracks, camPoses, ...
        intrinsics, 'PointsUndistorted', true, 'AbsoluteTolerance', 1e-5,...
        'RelativeTolerance', 1e-5, 'MaxIterations', 200, 'FixedViewID', 1);
        
    vSet = updateView(vSet, camPoses); % Update view set.
    
    % Bundle adjustment can move the entire set of cameras. Normalize the
    % view set to place the first camera at the origin looking along the
    % Z-axes and adjust the scale to match that of the ground truth.
    
    vSet = helperNormalizeOwn(vSet);

    % Update the camera plots
    camEstimated.AbsolutePose = poses(vSet).AbsolutePose(viewId);

    % Plot the estimated trajectory.
    locations = vertcat(poses(vSet).AbsolutePose.Translation);
    set(trajectoryEstimated, 'XData', locations(:,1), 'YData', ...
    locations(:,2), 'ZData', locations(:,3));

    prevI = I;
    prevFeatures = currFeatures;
    prevPoints   = currPoints;  
end