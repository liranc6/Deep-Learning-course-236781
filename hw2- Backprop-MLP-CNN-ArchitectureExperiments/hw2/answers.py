r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**
1.) A) The Jacobian tensor will contain the derivative of every member of the output matrix with respect to every member of the input matrix.
    The output matrix is of shape (64,1024) and the input matrix is of shape (64,512).
    Hence the Jacobian will be a 4d tensor of shape (64,1024)x(64,512).
    B)$y_{i,j} = \sigma(w_{i,1}x_{1,j} + ... + w_{i,n}x_{n,j})$
    Hence we get that $\frac{\partial y_{i,j}}{\partial x_{a,b}} = w_{b,j}$ if a = i and 0 otherwise.
    This means that only around $\frac{1}{n}$ of the entries of the jacobian are non-zero which makes it sparse.
    C) No we do not have to materialize the above Jacobian:
    By the chain rule we get that $\delta\mat{X} = \pderiv{L}{\mat{Y}}*\pderiv{\mat{Y}}{\mat{X}} = \pderiv{L}{\mat{Y}}*\pderiv{\mat{X}\mat{W}^{T}}{\mat{X}} = \pderiv{L}{\mat{Y}}*\mat{W}^{T}$.
    This only requires matrix multiplication and so we never have to calculate the Jacobian.
    
2.) A) The Jacobian tensor will contain the derivative of every member of the output matrix with respect to every member of the input matrix.
    The output matrix is of shape (512,1024) and the input matrix is of shape (64,512).
    Hence the Jacobian will be a 4d tensor of shape (512,1024)x(64,512).
    B)$y_{i,j} = \sigma(w_{i,1}x_{1,j} + ... + w_{i,n}x_{n,j})$
    Hence we get that $\frac{\partial y_{i,j}}{\partial w_{a,b}} = x_{i,a}$ if b = j and 0 otherwise.
    This means that only around $\frac{1}{n}$ of the entries of the jacobian are non-zero which makes it sparse.
    C) No we do not have to materialize the above Jacobian:
    By the chain rule we get that $\delta\mat{W} = \pderiv{L}{\mat{Y}}*\pderiv{\mat{Y}}{\mat{W}} = \pderiv{L}{\mat{Y}}*\pderiv{\mat{X}\mat{W}^{T}}{\mat{W}} = \pderiv{L}{\mat{Y}}*\mat{X}^{T}$.
    This only requires matrix multiplication and so we never have to calculate the Jacobian.




"""

part1_q2 = r"""
**Your answer:**
No, it is not required.
Back propogation is a method that allows to accelerate the computation of gradients by computing them efficiently.
As we discussed in the previous question the intermediate gradients of the chain rule are 4D tensors and so we can just 
multiply these tensors according to the chain rule and obtain the required gradients without backpropogation.

Additionally there is a subfield of learning which focuses on descent based approaches without derivatives at all.
A great example of this is genetic algorithms, which manage optimize functions in a descent-like way without those functions being differentiable.



"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 1, 0.05, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    lr_vanilla = 0.03
    lr_momentum = 0.003
    lr_rmsprop = 0.00021544346900318823  #There is a truly marvelous way to derive this value but this comment is too small to contain it.
    wstd = 0.2
    reg = 0.001584893192461114
    '''lr_vanilla = 0.02
    lr_momentum = 0.002
    lr_rmsprop = 0.0002
    wstd = 0.2
    reg = 0.002'''
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.2
    lr = 0.002
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1.) We think of the dropout mechanic as a form of regularization and so we would expect a low value to yield a case of overfitting and a high value to yield a case of underfitting.
    The graphs match our expectations since we see that for low values like 0 we get an extreme case of overfitting and for high values like 0.8 we get an extreme case of underfitting.
2.) The low dropout setting achieves the better train results but worse test results than the high setting.
    We expected this to happen since low values mean overfitting and high values mean underfitting.

"""

part2_q2 = r"""
**Your answer:**
Yes, it is possible for both loss and accuracy to increase at the same time since accuracy only depends on the sample 
with the highest score while the loss depends on the entire distribution.
Because of this, if we assume that the dataset has 2 samples: one with label 0 and one with label 1 and that in one epoch, the model
returns the scores 0: 0.49 1: 0.51 and 0:0 1:1 for inputs 0,1 respectively.
We get that the avg loss of this epoch is ~ 0.51 and the accuracy is 50%.
Now if the model returns the scores 0: 0.51 1: 0.49 and 0: 0.49 1: 0.51 for inputs 0,1 respectively.
We get that the avg loss of this epoch is ~ 0.97 and the accuracy is 100%.
We see that both loss and accuracy increased as desired.



"""

part2_q3 = r"""
**Your answer:**
1.) Gradient descent is a method of optimization which minimizes the value of a function by taking small steps in the opposite direction of its gradient.
    As we saw before back-propagation is not required for this calculation and so this method would still work independently from it.
    Back-propagation is an efficient way to calculate the gradients of a network which relies on the chain rule.
    This method can be used with any descent based optimizers and not just GD.
    
2.) The main difference between GD and SGD is the way the gradients are calculated:
    In GD the gradients are calculated on the entire dataset while in SGD the gradient is only calculated on a single sample or a small batch of samples.
    This difference heavily affects the training process in the following ways:
    a) Loading the entire dataset into memory is often infeasible and so GD is often impossible to use while SGD is still pracical on the same problem.
    b) Using the entire dataset to calculate the gradients can lead to cases of over fitting, using only a small batch every time produces slightly different kinds of gradients each time which helps escape local minima.
    c) GD tends to converge quicker than SGD since one step of GD can take SGD many steps to catch up to.
    
3.) One of the main challenges of Deep learning is that loss functions tend have many local minimas and models tend to have millions of parameters.
    SGD is more fit to deal with those challenges for the following reasons:
    
    a) Datasets in deep learning are huge and loading them into memory for computation is infeasible.
    b) Calculating a step of SGD is much faster than GD because of the small batch size.
    c) SGD is better at escaping local minima which is crucial due to the landscape of losses in Deep learning.
    
4.) A.) Yes, this would yield a method equivalent to GD.
        The reason for these 2 methods being equivalent is that the loss computed in each forward pass is a sum of the losses of each seperate sample and so we get that
        from the linearity of the gradient, the gradient of the sum of the losses is the sum of the gradients of the losses which again can be further broken down into each sample.
        Once broken down into each sample we get precisely the step of GD as required.

    B.) The reason we got a memory error is that even though we are not saving the entire dataset in memory, we need to save the input for each layer in order to use it in the backward pass to compute gradients.
        This means that the dataset is being saved implicitly in memory even if it wasn't our intention.
    



"""


# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 4
    hidden_dims = 8
    activation = 'relu'
    out_activation = 'none'
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.003
    weight_decay = 0.01
    momentum = 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**
1.) We see in the graphs that the test loss was quite low and then it increased, stopping at the point at which it was smaller would yield a lower optimization error.
2.) We see that the final distance between the test accuracy and the train accuracy is quite high which yields a high generalization error.
3.) The original dataset includes noise which is an error that is independent of the choice of model as well a difference in rotation between the 3 datasets which is something that mlps cannot properly model.
    This means that MLPS will have trouble learning the optimal classifier which leads to a high approximation error.



"""

part3_q2 = r"""
**Your answer:**
According to the way data is generated, we generate 4000 samples with degree 10 and then 4000 sample with degree 50.
Because we do not shuffle, we get that the train set gets all the degree 10 samples as well as some of the degree 50 samples.
Specifically all degree 50 samples in the train set are of the same kind which leads to an imbalance in the validation set.
As we see in the confusion matrix, there is a much bigger chance for a true negative than a true positive.
This leads to the model focusing more on negative samples and less on positive samples whioch would lead to a high FNR as we see in the matrix.




"""

part3_q3 = r"""
**Your answer:**
1.) In this case a false positive of our model leads to a healthy person getting an expansive and unnecessary test.
    A false negative would lead to a person getting non-lethal symptoms and then a low cost cure.
    In this case a FP is worse than a FN and so we would choose the optimal point such that it prioritizes a low FPR over a low FNR.

2.) In this case a FP would still lead to an unnecessary expense but a FN could lead to death.
    Because of that, we want to ensure a low FN rate and so we would choose a point on the ROC curve that ensures a low FNR while trying to minimize the FPR.

"""


part3_q4 = r"""
**Your answer:**
1.) We see that for all of the models, the decision boundaries get sharper and fit the data better.
    We also see that the performance on both datasets improves but that the model overfits more as the width grows.

2.) With the depth we see similar results as to those with the width but with one significant difference:
    changing the depth leads to a much more expressive model than changing the width and as such the changes here are more profound than before.
    
3.) Even though deeper models are much more expressive than wider models with the same number of neurons, they are much harder  and so we see that the deeper model gets slightly worse results.

4.) As we explained before, the validation set is not balanced between the samples while the test set is.
    This means that using a default threshold would lead to a high error rate from the shift in distribution.
    Changing the threshold makes the model invariant to this imbalance which greatly increases generaliztion.
    


"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr, weight_decay, momentum = 0.01, 0.001, 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**

**1-**
**Vanilla resnet block**
after each convolution we have (channels-out * ((channels_in * width * height) + 1) parameters.
$K \cdot (C_in \cdot F^2 + 1)$

Thus
First convolution:
Parameters = $256 \cdot ((256 \cdot 3 \cdot 3) + 1) = 590,080 $
   
Second convolution:
Parameters = $256 \cdot ((256 \cdot 3 \cdot 3) + 1) = 590,080 $
   
Total parameters = 590,080 $\cdot$ 2 = 1,180,160
   
*BottleNeck block*
First convolution:
parameters = $64 \cdot ((256 \cdot 1 \cdot 1) + 1) = 16,448 $
   
Second convolution:
parameters = $64 \cdot ((64 \cdot 3 \cdot 3) + 1) = 36,928 $
   
Third convolution:
parameters = $256 \cdot ((64 \cdot 1 \cdot 1) + 1) = 16,640 $
   
Total parameters = 16,448 + 36,928 + 16,640 = 70,016 
   
We can see that in bottleneck there are fewer parameters

**2-**
For each filter, for each stride we have $C_{in}\times k^2$ multiplication operations $C_{in}\times (k^2 -1)$ addition operations and 1 bias adding operation.
Therefore we have $2\times C_{in}\times k^2$ operations per filter $\cdot$ stride.
We have $H\times W$ strides, so we have  $2\times C_{in}\times k^2\times C_{out}\times H\times W$  operations per layer. 
For each layer we denote the number of params $P_l = C_{in}\times k^2\times C_{out}$  
Thus, the number of floating point operations for each layer is  $2\times P_l\times H\times W$

shortcut connections require $C_{out}\times H \times W$ Addition operations


To conclude the final equation of floating point operations would be:
floating point operations = $C_{out}\times H \times W +\sum_{l\in layers} 2\times P_l\times H \times W$
Using dimension preserving padding, we get constant HxW thus we can simplify:
floating point operations = $C_{out}\times H \times W +2\times H \times W\sum_{l\in layers}P_l$
floating point operations = $H \times W(C_{out} + 2\times\sum_{l\in layers}P_l)$
denoting $P_b$ as number of parameters in block we get
floating point operations = $H \times W(C_{out} + 2\times P_b)$

In vanilla block we have floating point operations = $H \times W(256+2\times 1,180,160) = 2,360,576\times H\times W$
In the bottleneck block we have floating point operations = $H\times W (256+2\times 70,016) =  H\times W \times 140,288 $

We can see that bottleneck requires much less floating point operations to compute an output

**3-** 
Spatial - 
    regular block - we use two convolution layers of 3x3 thus we get respective field of 5x5.
    bottleneck block -  we use two convolution layers of 1x1 and one convolution layer of 3x3 thus we get
        respective field of 3x3.
   
We see that vanilla block combines the input better in terms of spatial.
   
   
Across feature map-   
   In bottleneck block not all inputs has the same influence across feature map, that because we project the first layer to a smaller dimension
   In vanilla block we don't project the input (therefore we have the same influence)
"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**

1. We got the best accuracy by using L=4.
We see that when using L=2 and L=4 the results are almost the same.
As we learned in class - deeper network (with more layers) are more complex and can fit better to general data (up to a certen level, where we get overfit)

2. The network was untrainable for depths 8 and 16 - the largest depths. 
We think that happened because of vanishing gradients - the info flowing in the network passed a lot of loss layers, and it makes the gradient zero.
"""

part5_q2 = r"""
**Your answer:**
When looking at both 1.1 and 1.2 experiments - we can see that both networks aren't trainable with L=8 due to vanishing gradients.
We also see that with L=4 we get better test accuracy for every K tested. Which suitable for what we got in experiment 1.1

We see that for L=4 for more filters per layers we got better test accuracy whereas for L=2 the fewer filters per layer (on number tested) get better test accuracy.




"""

part5_q3 = r"""
**Your answer:**
We can see that for L=4 - we got vanishing gradients and the network was untrainable.

As we saw in the previous experiments for a fixed k we get better test accuracy for higher amount of layers,
but we can also see that after adding too much architecture complexity, the networks can be unstable and become untrainable.
"""

part5_q4 = r"""
We can see that the model was not trainable with K=32, our guess is that it might be because of the momentum and that
we jump between local minimas.
We can see that for L=2 and k=[64, 128, 256] the model was trainable and for L=4 and L=8, it wasnt. we balieve it for
the same reasons as the previous experiments.

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**

For the first picture:
1. The model didn't recognize the dolphins present in the image. Instead of it, it detects a bird and a person.
2. The reason for this is that "dolphin" is not a possible class for an object in YoloV3 model. It probably detects a 
bird because of the sky in the background and the black color of the dolphin. The method I would suggest to recognize 
dolphin in the image would be to add some dolphin images to the dataset, a new dolphin class and retrain the model with
it. I would also suggest having some shadow images of dolphins in the new dataset to be able to recognize them.

For the second picture:
1. The model detected the objects almost right. In right side, it detects two dogs and draw the relevant bounding boxes
around them. In left side, it detects a cat, which is present in the image, but draw the bounding box mainly on the
third dog present left and not around the cat. So it missed the right bounding box around the cat and the detection of
the third dog.
2. I would say that the reason why the models didn't detects well the object on right is because the image is cluttered,
there is a lot of objects and it's difficult for the model to detect each one of them.
(TODO: check this with team)
"""


part6_q2 = r"""
**Your answer:**




"""


part6_q3 = r"""
**Your answer:**

For the first picture - Illumination conditions:
The model detects the bottle present in the image as a cup.
Although 'bottle' is in the classes names, the model detects it as a cup.
It's because of the lighting condition, we have a low 
light (I needed to shut down the light in the room :)) and maybe also because of the clutter in the background
(multiple objects to detect in a small area).
Yet,it identifies the tv monitor, keyboard and mouse correctly. 

For the second picture - Occlusion:
The model didn't detect the 'hair drier' in the picture even though it is in the classes names.
We believe that the reason is because it's cut in the image, and partially occlude, and thus missing important features.

For the third picture - Cluttering:
Crowded or Cluttered Scenario: Too many objects in the image make it extremely crowded.
We can see that the model can detect apples but it didnt detect all the apples.
It's not obvious there should be apple in the air. It detects some apple clustered together on the table and in the
floor box but there is a lot of missing. We think the reason for this is that there is too much object to detect in this
image. Also, it was able to detect other objects. 



"""

part6_bonus = r"""
**Your answer:**




"""