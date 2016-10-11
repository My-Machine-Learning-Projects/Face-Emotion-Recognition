imgSets = imageSet('E:\Face Emotion Recognition\Second Try\JAFFE 4 Emotions\','recursive');
% { imgSets.Description } % display all labels on one line
% [imgSets.Count]         % show the corresponding count of images
minSetCount = min([imgSets.Count]); % determine the smallest amount of images in a category

% Use partition method to trim the set.
imgSets = partition(imgSets, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
%[imgSets.Count]

[trainingSets, validationSets] = partition(imgSets, 0.7, 'randomize');
bag = bagOfFeatures(trainingSets);

%SVM with Linear Kernel
categoryClassifier = trainImageCategoryClassifier(trainingSets, bag);

% Training the classifier with SVM using Gaussian Kernel
% mySVM = templateSVM('KernelFunction', 'gaussian');
% categoryClassifier = trainImageCategoryClassifier(trainingSets, bag, 'LearnerOptions', mySVM);

% Evaluating performance on TrainingSet
confMatrix = evaluate(categoryClassifier, trainingSets);

% Evaluating performance on ValidationSet
confMatrix_val = evaluate(categoryClassifier, validationSets);

% Evaluating performance on test image
img = imread('test.jpg');
[labelIdx, scores] = predict(categoryClassifier, img);

% Display the string label
% categoryClassifier.Labels(labelIdx)