r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1)  No, the test set allows us to estimate our loss Ld over the entire distribution.
2)  No, for example there could be a case where there are 10 classes and 9 make up the train set
    and the last makes up the test set. In this case the training is completely irrelevant to the test set 
    and learning will fail.
3)  Yes, validation exists to try to estimate the test accuracy without data leaking from the test set into the
    learning process. Incorporating the test set would go against that goal and make the final test accuracy
    not representative of the generalized loss.
4)  Not exactly. While the performance of each fold can be thought of as a proxy for the generalized error, 
    The point of k-fold cv is to use the average of all the folds as said proxy.

"""

part1_q2 = r"""
**Your answer:**

No, this approach is not justified and would lead to bad generalization.
In general using test data during training is a bad idea since it can lead to the model learning
the test set and as such overfitting to both it and the train set.
Because of that, the test set stops being representative of the generalized distribution and the 
performance on actual unseen data will be bad.
    

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
k serves as a way to control the regularization of the model and by doing so we move
between overfitting and underfitting.
With k=1 we memorize the entire dataset and achieve a train accuracy of 100% which is a case
of extreme overfitting. On the other hand for large values like k = num_samples we just always
guess the most common sample in the dataset and dont learn anything which is an extreme case of underfitting.
Inbetween those two extremes there is an optimal k which helps avoid both overfitting and underfitting.

"""

part2_q2 = r"""
**Your answer:**

1)  Training on just the training set and evaluating the model based on the same train set will
    encourage models that overfit heavily to the train set and achieve poor generalization.
    For example, if we were to try this method with knn we would always get k = 1
    since it achieves 100% train accuracy but terrible generalization.
    k-fold cv avoids this by evaluating the model on unseen folds and by doing so simulate normal 
    training conditions which can generalize.
2)  Constantly evaluating the model on the train set will make the learning process too specific to the
    chosen test set and cause test data to implicitly leak into the training process.
    In essence the more we know about the test set, the less it represents the distribution and
    so this method will create a model that achieves good test accuracy but generalizes poorly.

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
We will show that $\Delta > 0$  is arbitrary by showing that changing it to $\alpha\Delta$ results in an equivalent problem.
Let $\alpha > 0$ and define $\mathbf{W'} = \alpha*\mathbf{W}$ Hence the new loss is now 
    $L(\mathbf{W'}) =
    \frac{1}{N} \sum_{i=1}^{N} 
    \max\left(0, \alpha\Delta+ \vec{w'_j} \vec{x_i} 
    - \vec{w'_{y_i}} \vec{x_i}\right) 
    +
    \frac{\lambda}{2} \|{\mathbf{W'}}\|^2 =\\=
    \frac{1}{N} \sum_{i=1}^{N} 
    \max\left(0, \alpha\Delta+ \vec{w'_j} \vec{x_i} 
    - \vec{w'_{y_i}} \vec{x_i}\right) 
    +
    \frac{\lambda}{2} \|{\mathbf{W'}}\|^2 =\\=
    \frac{1}{N} \sum_{i=1}^{N} 
    \max\left(0, \alpha\Delta+ \alpha*\vec{w_j} \vec{x_i} 
    - \alpha*\vec{w_{y_i}} \vec{x_i}\right) 
    +
    \frac{\lambda}{2}*\alpha^2 \|{\mathbf{W}}\|^2=\\=
    \frac{1}{N} \sum_{i=1}^{N} 
    \max\left(0, \alpha(\Delta+ \vec{w_j} \vec{x_i} 
    - \vec{w_{y_i}} \vec{x_i}\right)) 
    +
    \frac{\lambda}{2}*\alpha^2 \|{\mathbf{W}}\|^2=\\=
    \frac{1}{N} \sum_{i=1}^{N} 
    \max\left(0, \Delta+ \vec{w_j} \vec{x_i} 
    - \vec{w_{y_i}} \vec{x_i}\right) 
    +
    \frac{\lambda}{2}*\alpha^2 \|{\mathbf{W}}\|^2
    $
    The final formulation is equivalent to the original problem differing in the choice of $\lambda$ and so because for every $\lambda$ that is optimal for the original, $\lambda*\alpha^2$ is optimal for the new problem, we get that the two are equivalent.
"""

part3_q2 = r"""
**Your answer:**


1) What the linear model is actually learning is a "prototype" of each digit,
Meaning an average representation of each digit that is similar to all examples of that digit.
Because of that, the final weights look like the digits and are interpretable.
The samples that we got wrong tend to be rotated or deformed in some way and so because we only learn an 
average digit and we dont try to characterize or define each digit, we get those ones wrong.

2) Both of these interpretations try comparing the sample to other things and seeing what is closest.
the difference is that knn compares the sample to the dataset and does a "vote" to decide the outcome 
while our model learns a representation of each digit and compares the new sample to those representations.

"""

part3_q3 = r"""
**Your answer:**

1) Too low.
In the case of a good learning rate, the graph would be far less steep but would be relatively monotonous, 
decreasing towards the solution at a good rate.
A high learning rate would not decrease continuously like ours but keep bouncing around values without converging or even start diverging.

2) Our model is highly overfitted to the training set.
We can see that because the training loss is quite a bit higher that both the validation loss and the test loss.
This can be fixed by increasing the weight decay value.

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**
The ideal pattern we wish to see in residual plots is residuals with roughly the same size and that size being quite small around the zero line.
We can see on the first plot that the points forms a curved pattern,
whereas on the final plot after the CV we see that the variance is roughly the same and the points are forming approximately constant width band around the identity line. i.e.:we predict the data better.
We can learn that feature engineering and tunning can improve predictions

"""

part4_q2 = r"""
**Your answer:**

1. Yes, this is still a linear regression but not in the original parameters.
while we are still fitting a line to our data while minimizing mse, this data is now a non-linear representation of the original data and as such can fit a non-linear function to the data which impossible to achieve by linear regression.
Note: we address the linearity of the parameters and not of the features.

2. Yes, we can fit any non-linear function, because any non-linear function has a matching transformation on the features which makes the function linear on the new featurs 
BUT we might "overfit" the training data and get worse test results.

3. Yes it would still be a hyperplane but not in the original space of features.
This new hyperplane would still be linearly separating data but the data is now the original data after a non-linear transformation.
Because of that data that could not be separated previously may be linearly separable now

"""

part4_q3 = r"""
**Your answer:**
1.) The reason we look at a logspace instead of a linspace is because Order of magnitude
    tends to matter more than the distance between values and usually values seperated by a constant will cause a lot of different values to give the same results.
    On the other hand values with different orders of magnitude tend to give different results while not skipping over many good values of the parameter.

2.) First, we have 3 possible values for lambda and 20 possible values for degree which means 60 total combinations.
    For every combination we train the model on 3 seperate folds of the cv process and so the model is trained 180 times in total.
"""

# ==============
