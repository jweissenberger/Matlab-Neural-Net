function [ accuracy ] = AccuracyCheck( Yout, Y_correct )
%Calculates the accuracy of a model given the actual answers and the
%predicted answers of a neural network
%   Yout and Y_test must be the same size

[~, col] = size(Yout);

corr = 0;

for i = 1 : col

    [~, a] = max( Yout(:, i));
    [~, b] = max( Y_correct(:, i));
    
    if a == b
        corr = corr + 1;
    end

end 

accuracy = corr/col;

end

