%% Iris Neural Network

% A Neural Network that classifies irises bases on dimensions of their
% features
% the columns of iris in order are: sepal length, sepal width, petal length
% petal width and then the species

% *************************************************************
% the architecture of the network needs to be changed, if you watch the
% weights over the course of training, you'll see that the weights in the
% first layer barely change no matter how many iterations you train on

% this could also be an issue from stochastic gradient descent 
% (could switch to batch gradient descent by taking the average of the
% backprop error over a given batch) but this dataset is simple enough to
% where I shouldn't have to worry about that

% need to go through: forward prop, back prop and the updating of weights
% to switch to a single hidden layer 
%% Data Preperation
load fisheriris.mat;

% set random seed
rng(1)

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
Y_trainVals = trainind(:,5);
% creates Y_train such that ther is an output for each node
Y_train = zeros(120, 3);
for i = 1 : 120
   if Y_trainVals(i) == 1
       Y_train(i, 1) = 1;
   end
   
   if Y_trainVals(i) == 2
       Y_train(i, 2) = 1;
   end
   
   if Y_trainVals(i) == 3
       Y_train(i, 3) = 1;
   end
end

% need to change these so that the network can read it
X_test = testind(:,1:4);
Y_testVals = testind(:,5);
% creates Y_train such that ther is an output for each node
Y_test = zeros(30, 3);
for i = 1 : 30
   if Y_testVals(i) == 1
       Y_test(i, 1) = 1;
   end
   
   if Y_testVals(i) == 2
       Y_test(i, 2) = 1;
   end
   
   if Y_testVals(i) == 3
       Y_test(i, 3) = 1;
   end
end

%change orientation 
X_test = X_test';
X_train = X_train';
Y_test = Y_test';
Y_train = Y_train';



% only look at setosa and versicolor because they are linearly separable 
% and I am using a sigmoid activation function for the last layer and would
% need to use softmax if I wanted to do this for more than two outputs
Y_train = Y_train(1:2, 1:79);
X_train = X_train(:, 1:79);
Y_test = Y_test(1:2, 1:19);
X_test = X_test(:, 1:19);


[num_features, ~] = size(X_train);
%scale the data to between 0 and 1
%note they are scaled separately to prevent leakage
for i = 1 : num_features,
   X_train(i, :) = (X_train(i, :) - min(X_train(i, :))) / (max(X_train(i, :)) - min(X_train(i, :))); 
   X_test(i, :) = (X_test(i, :) - min(X_test(i, :))) / (max(X_test(i, :)) - min(X_test(i, :)));
end


%% Define Hyperparameters
% This will be a 4 layer neural network, with two hidden layers

inputLayerSize = 4; % representing the 4 features 
outputLayerSize = 2; % representing the 2 kinds of iris we are looking at
hiddenLayer1Size = 4;
hiddenLayer2Size = 3;


%% Randomly Initialize Weights and Biases

W12 = rand(hiddenLayer1Size, inputLayerSize);
b12 = rand(hiddenLayer1Size, 1);
W23 = rand(hiddenLayer2Size, hiddenLayer1Size);
b23 = rand(hiddenLayer2Size, 1);
W34 = rand(outputLayerSize, hiddenLayer2Size);
b34 = rand(outputLayerSize, 1);

%% Number of iterations of training 
for i = 1 : 1000,
    
%% Randomly Select Training Example
% Because this neural network is trained using Stochastic gradient descent
% the network is trained using only one example

[~,s] = size(X_train);
n = randi(s);
Xone = X_train(:,n);
Yone = Y_train(:,n);

%% Forward Propagation

[Yout, a3, a2, z4, z3, z2] = ForwardProp( Xone, W12, b12, W23, b23, W34, b34 );

%% Back Propagation

[del4, del3, del2] = Backprop(Yout, Yone, z4, z3, z2, W34, W23);

%% Update Weights and bias

nu = 0.1; %learning rate

W34 = W34 - nu * (del4*a3');
b34 = b34 - nu * del4;

W23 = W23 - nu * (del3*a2');
b23 = b23 - nu * del3;

W12 = W12 - nu * (del2*Xone');
b12 = b12 - nu * del2;


%% Check Accuracy
if rem(i, 2) == 0
    [ Ycheck, ~, ~, ~, ~, ~ ] = ForwardProp( X_test, W12, b12, W23, b23, W34, b34 );
    fprintf('Validation accuracy: %f \n', AccuracyCheck(Ycheck, Y_test))
    
    [ Ycheck, ~, ~, ~, ~, ~ ] = ForwardProp( X_train, W12, b12, W23, b23, W34, b34 );
    fprintf('Train accuracy: %f \n \n', AccuracyCheck(Ycheck, Y_train))
    

end
end

