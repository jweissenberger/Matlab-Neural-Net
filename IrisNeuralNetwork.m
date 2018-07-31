%% Iris Neural Network

% A Neural Network that classifies irises bases on dimensions of their
% features
% the columns of iris in order are: sepal length, sepal width, petal length
% petal width and then the species

%% Data Preperation

%load the data
load fisheriris.mat;

% create the output labels
label = zeros(150,3);
for i = 1 : 150
    if i <=50
        label(i,1) = 1;
    elseif i <=100
        label(i,2) = 1;
    else 
        label(i,3) = 1;
    end
end

%concat the labels to the rest of the data
iris = [meas label];

%scale the data
for g = 1 : 4,
    clear max
    max = max(iris(:, g));
    clear min
    min = min(iris(:, g));
    for j = 1:150,
        iris(j,g) = (iris(j,g)-min)/(max-min);
    end
end

%test train split
[trainind, ~, testind] = dividerand(iris', .8, 0, .2);

%break up the important information from the test traint split
X_test = testind(1:4, :);
Y_test = testind(5:7, :);
X_train = trainind(1:4, :);
Y_train = trainind(5:7, :);
%% Define Hyperparameters
% This will be a 4 layer neural network, with two hidden layers

inputLayerSize = 4; % representing the 4 features 
outputLayerSize = 3; % representing the 3 kinds of iris
hiddenLayer1Size = 6;
hiddenLayer2Size = 5;


%% Randomly Initialize Weights and Biases

W12 = rand(hiddenLayer1Size, inputLayerSize); %try randn
b12 = rand(hiddenLayer1Size, 1);
W23 = rand(hiddenLayer2Size, hiddenLayer1Size);
b23 = rand(hiddenLayer2Size, 1);
W34 = rand(outputLayerSize, hiddenLayer2Size);
b34 = rand(outputLayerSize, 1);

%% Number of iterations of training 
fprintf(' Iteration \t  Accuracy \n')
for i = 1 : 100
    
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
if rem(i, 10) == 0
[ Ycheck, a3, a2, z4, z3, z2 ] = ForwardProp( X_test, W12, b12, W23, b23, W34, b34 );
fprintf('% i \t\t % .4f \n',i, AccuracyCheck(Ycheck, Y_test))
end 

end
