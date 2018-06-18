function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% 向前传递
a1=[ones(m,1) X];
z2=a1*Theta1';
a2=[ones(size(z2,1),1) sigmoid(z2)];
z3=a2*Theta2';
a3=sigmoid(z3);
h=a3;
%多分类问题，需要用向量表示每一类输出结果
Y=zeros(m,num_labels);
for i=1:m
Y(i,y(i))=1;
end
J=-1/m*sum(sum(Y.*log(h)+(1-Y).*log(1-h)));
%加上一些惩罚，减少theta的影响
penalty=lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2))+...
                      sum(sum(Theta2(:,2:end).^2)));
J=J+penalty;

%反向传播
%首先计算y的差距
delta3=a3.-Y;
delta2=(delta3*Theta2).*sigmoidGradient([ones(size(z2,1),1) z2]);
%轴距的列不需要
delta2=delta2(:,2:end);

%计算权重的差距
D1=delta2'*a1;
D2=delta3'*a2;
%第一列权重不需要乘以lambda因为是bias,让第一列是0即可
Theta1_grad=D1./m+lambda/m*[zeros(size(Theta1, 1), 1)...
                              Theta1(:, 2:end)];

Theta2_grad=D2./m+lambda/m*[zeros(size(Theta2, 1), 1)...
                              Theta2(:, 2:end)];


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
