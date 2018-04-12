load fisheriris.mat;

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

iris = [meas label];


for g = 1 : 4,
    clear max
    max = max(iris(:, g));
    clear min
    min = min(iris(:, g));
    for j = 1:150,
        iris(j,g) = (iris(j,g)-min)/(max-min);
    end
end

[trainind, ~, testind] = dividerand(iris', .8, 0, .2);

X_test = testind(1:4, :);
Y_test = testind(5:7, :);
X_train = trainind(1:4, :);
Y_train = trainind(5:7, :);










