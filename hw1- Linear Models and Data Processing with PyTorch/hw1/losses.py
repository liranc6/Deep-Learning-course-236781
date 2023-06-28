import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        t1 = x_scores+self.delta
        indices = torch.unsqueeze(y,0)
        indices = torch.transpose(torch.cat((indices,indices) , 0),1,0)
        t2 = torch.gather(x_scores, 1, indices)[..., 0]
        t1 = torch.transpose(t1, 1, 0)-t2
        t1 = torch.transpose(t1,1,0)
        hinge = torch.maximum(torch.zeros_like(t1),t1)
        in_loss = torch.sum(hinge, 1)
        in_loss = in_loss-torch.ones_like(in_loss)*self.delta
        loss = torch.sum(in_loss)/len(in_loss)

        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx = {'M': t1 , 'x_scores': x , 'y': y}
        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        indicator = (self.grad_ctx['M']>0).float().to(dtype = self.grad_ctx['M'].dtype)
        diag_sum = torch.tensor([list(torch.sum(indicator, 1))])
        diag_sum = torch.transpose(diag_sum, -1, 0)
        index = torch.tensor([list(self.grad_ctx['y'])])
        index = torch.transpose(index, -1, 0)
        zeros = torch.zeros_like(indicator , dtype = self.grad_ctx['M'].dtype )
        g = indicator-zeros.scatter_(-1 , index , diag_sum , reduce = 'add')
        grad = torch.transpose(self.grad_ctx['x_scores'],1,0)@g/self.grad_ctx['x_scores'].shape[0]
        # ========================

        return grad
