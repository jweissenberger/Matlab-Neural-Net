%% Iris Neural Network

% A Neural Network that classifies irises bases on dimensions of their
% features
% the columns of iris in order are: sepal length, sepal width, petal length
% petal width and then the species

%% Data Preperation
load fisheriris.mat;

% 1 = setosa, 2 = versicolor, 3 = verginica
% creates numeric lables for the network to predict

label = ones(1, 150);

for i = 51 : 150
    if i <=100
        label(1,i) = 2;
    else 
        label(1,i) = 3;
    end
end

% add the labels with the corresponding data
iris = [meas label'];

% a randomized test-train split, that outputs the indicies
[trainind, ~, testind] = dividerand(iris', .8, 0, .2);
trainind = trainind';
testind = testind';

% seperate the inputs from the outputs for the test and train data
X_train = trainind(:,1:4);
Y_train = trainind(:,5);

% need to change these so that the network can read it
X_test = testind(:,1:4);
Y_test = testind(:,5);


%% Define Hyperparameters
% This will be a 4 layer neural network, with two hidden layers

inputLayerSize = 4; % representing the 4 features 
outputLayerSize = 3; % representing the 3 kinds of iris
hiddenLayer1Size = 10;
hiddenLayer2Size = 10;


%% Randomly Initialize Weights and Biases

W12 = rand(inputLayerSize, hiddenLayer1Size);
W23 = rand(hiddenLayer1Size, hiddenLayer2Size);
W34 = rand(hiddenLayer2Size, outputLayerSize);
% figure out biases


%% Forward Propagation
% (all matrix multiply)
% need to put in biases

z2 = X_train*W12;% activity going into the second layer
a2 = sigmoid(z2);
z3 = a2*W23;
a3 = sigmoid(z3);
z4 = a3*W34;
Yout = sigmoid(z4);


