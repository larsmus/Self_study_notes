## Chapter 6: Deep Feedforward Networks

The first chapter (6.1) describes the classic example of linear models failing to learn XOR. If all the hidden layers are linear, the function itself would still be linear in the input data, and would not be able to capture the underlying XOR model. They proceed to show that a simple **rectified linear unit** or ReLU activation layer , $g(x) = \max\{0,x\}$ solves the problem. This yields a piecewise linear function, and therefore preserve many of the properties that make linear models easy to optimize. ReLU is the default activation function recommended for most feedforward neural networks. 

The non-linearity of a neural network, will often cause the loss function to be non-convex and can therefore not guarantee optimality conditions. Usually use iterative gradient-based methods. 

Most modern neural networks are trained using maximum likelihood, i.e. the cost function is the negative log likelihood or equivalently described as the cross-entropy between the training data and the model distribution. 

$$J(\boldsymbol{\theta}) = - \mathbb{E}_{\boldsymbol{x},\boldsymbol{y}}\log p_{model}(\boldsymbol{y} | \boldsymbol{x}).$$











