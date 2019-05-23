## Chapter 6: Deep Feedforward Networks

The first chapter (6.1) describes the classic example of linear models failing to learn XOR. If all the hidden layers are linear, the function itself would still be linear in the input data, and would not be able to capture the underlying XOR model. They proceed to show that a simple **rectified linear unit** or ReLU activation layer , $g(x) = \max\{0,x\}$ solves the problem. This yields a piecewise linear function, and therefore preserve many of the properties that make linear models easy to optimize. ReLU is the default activation function recommended for most feedforward neural networks. 

The non-linearity of a neural network, will often cause the loss function to be non-convex and can therefore not guarantee optimality conditions. Usually use iterative gradient-based methods. 

Most modern neural networks are trained using maximum likelihood, i.e. the cost function is the negative log likelihood or equivalently described as the cross-entropy between the training data and the model distribution (Empirical mean (M) estimation).

$$J(\boldsymbol{\theta}) = - \mathbb{E}_{\boldsymbol{x},\boldsymbol{y}}\log p_{model}(\boldsymbol{y} | \boldsymbol{x}).$$

 Advantages:

*  Specifying the model (distribution) implies a cost function. 

* Avoid saturation, i.e that the gradient becomes very close to zero. Ofte caused because the activation functions becomes very flat (saturate), e.g. the exponential function. 

Disadvantage:

* Can approach negative infinity by assigning high density to the correct training set outputs.

Since a neural network in minimising in a function space, the cost function can be viewed as a **functional**, i.e. a mapping from the function space to the real numbers. Under squared error loss, we want to find the posterior mean of the function given our distribution. Note that MSE is not very commonly used in neural networks, since some output units saturate quickly when combined with this cost function. Hence the cross-entropy is more popular.

### Output units

The choice of output unit is tightly coupled with the cost function. Note that any function as output unit can also be used as a hidden unit. Some output units are:

* Linear units, for Gaussian output distributions. Doesn't not saturate. 
* Sigmoid units, for Bernoulli output distributions (Sigmoid function is the soft max function applied to a binary problem).
* Softmax for multinoulli output distributions. Often used as output of classifier, rarely inside the model itself. Squared error loss is not a good match for softmax. Softmax is invariant under addition, can normalise be subtracting the max to avoid saturation. Named softmax since it is a smoothed version of argmax. 
* Neural networks with gaussian mixtures as their output is often called mixture density models. 

Possibly solution to numerically unstable gradient-based optimisation are clipping gradients (near zero) and scleras gradients heuristically. 

### Hidden Units

How to choose the hidden units in your network? Specific for feedforward networks. 

Can usually disregard nondifferentiability at local points, such that in ReLu. Can be justified by numerical accuracy and can just use a sub gradient. 

* **ReLU**: $h(x) = \max\{0, \bf{W}^T \bf{x} + \bf{b}\}$. Often used as a default unit. Gradient is binary, can represent if a unit is active. Good practice to initialise $b$ as a vector of small values (0.1) such that the unit will be initially active. One drawback is that one can't learn on a domain where the gradient is zero. Some generalisations uses a slope for when the the unit is negative. When $z_i < 0: h_i = g(z, \alpha)_i = \max\{0, z_i\} + \alpha_i \min\{0, z_i\} $ If the slope is $-1$, then one have the absolute function $g(x) = |x|$. This is used in object recognition where the features are invariant to polarisation. A **leaky ReLU** fixes $\alpha_i$ to be a small value, while **parametric ReLU** treats $\alpha_i$ as a learnable parameter. 
* **Maxout units**:  Each maxout layer outputs the maximum of one of $k$ groups. $g(z)_i = \max_{j \in \mathbb{G}^{(i)}} \{z_i\}$, where $\mathbb{G^{(i)}} = \{(i-1)k, \dots, ik\}$. Generalise ReLU. Provides a way of learning a piecewise linear function that response to multiple directions in the input space.  The maxout layer can learning's a piecewise linear, convex function with up to $k$ pieces, and can thus be seen as learning the activation function itself, rather just the relationship between the units. Typically needs more regularisation then rectified linear units. Also, there are fewer parameters which can be an advantage computationally. Can also resist catastrophic forgetteng.  
* **Logistic sigmoid and hyperbolic tangent:** Closely related, $\tanh(z) = 2\sigma(2z) -1$. Sigmoid function is useful when we need to predict a probability. Saturate at both high and low values, so the use in hidden layers are discouraged. Can use in output layers when compatible with an appropriate loss function that undos the saturation. When a logistic sigmoid function must be used, the hyperbolic tangent usually works better. Is resembles the identity function more closely, hence training resembles a linear function as long as the activation of the network can be kept small. Also used in other architectures other than feedforward networks. 
* These former are the most common one, but exists a lot of others. Not unusual to get the same performance on other unit functions, but are mostly uninteresting. Sometimes it is acceptable to have a **purely linear unit** in the network, which offer an effective way of reducing the number of parameters. **Softmax** may also be used as an hidden unit. **Radial basis function (RBF)** can be used, but saturates often and are difficult to optimise. **Softplus** $g(a) = \log(1+e^a)$. A smooth version of the rectifier. Empirically it is not often better, and are therefore discouraged. **Hard tanh**: $g(a)= \max\{-1, \min\{1, a\}\}$. 

### Architecture design













