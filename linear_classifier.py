"""
Implements linear classifeirs in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import os
import torch
import torchvision
import random
import statistics
import time
import math
from abc import abstractmethod
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Optional

MODEL_PATH = './saved_models'

if torch.cuda.is_available:
    print('Good to go!')
else:
    print('No available GPU!')

# Template class modules that we will use later: Do not edit/modify this class
class LinearClassifier:
    """An abstarct class for the linear classifiers"""

    # Note: We will re-use `LinearClassifier' in both SVM and Softmax
    def __init__(self):
        random.seed(0)
        torch.manual_seed(0)
        self.W = None

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        learning_rate: float = 1e-3,
        reg: float = 1e-5,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ):
        train_args = (
            self.loss,
            self.W,
            X_train,
            y_train,
            learning_rate,
            reg,
            num_iters,
            batch_size,
            verbose,
        )
        self.W, loss_history = train_linear_classifier(*train_args)
        return loss_history

    def predict(self, X: torch.Tensor):
        return predict_linear_classifier(self.W, X)

    @abstractmethod
    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - W: A PyTorch tensor of shape (D, C) containing (trained) weight of a model.
        - X_batch: A PyTorch tensor of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A PyTorch tensor of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an tensor of the same shape as W
        """
        raise NotImplementedError

    def _loss(self, X_batch: torch.Tensor, y_batch: torch.Tensor, reg: float):
        self.loss(self.W, X_batch, y_batch, reg)

    def save(self, path: str):
        torch.save({"W": self.W}, path)
        print("Saved in {}".format(path))

    def load(self, path: str):
        W_dict = torch.load(path, map_location="cpu")
        self.W = W_dict["W"]
        if self.W is None:
            raise Exception("Failed to load your checkpoint")
        # print("load checkpoint file: {}".format(path))


class LinearSVM(LinearClassifier):
    """A subclass that uses the Multiclass SVM loss function"""

    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        return svm_loss_vectorized(W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """A subclass that uses the Softmax + Cross-entropy loss function"""

    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        return softmax_loss_vectorized(W, X_batch, y_batch, reg)


# **************************************************#
################## Section 1: SVM ##################
# **************************************************#


def svm_loss_naive(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples. When you implment the regularization over W, please DO NOT
    multiply the regularization term by 1/2 (no coefficient).

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """
    dW = torch.zeros_like(W)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = W.t().mv(X[i])
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                # Compute the gradient of the SVM term of the loss function and store #
                # it on dW.
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * torch.sum(W * W)

    # Compute the gradient of the loss function w.r.t. the regularization term  #
    # and add it to dW. (part 2)                                                #
    dW /= num_train
    dW += 2 * reg * W

    return loss, dW


def svm_loss_vectorized(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Structured SVM loss function, vectorized implementation. When you implment
    the regularization over W, please DO NOT multiply the regularization term by
    1/2 (no coefficient). The inputs and outputs are the same as svm_loss_naive.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """
    loss = 0.0
    dW = torch.zeros_like(W)  # initialize the gradient as zero

    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    scores = torch.mm(X, W)
    idx = torch.arange(0, X.shape[0])
    correct_class_score = scores[idx, y].unsqueeze(-1)
    margin = scores - correct_class_score + 1
    margin[idx, y] = 0
    loss = torch.sum((margin > 0)*margin)
    loss /= X.shape[0]
    loss += reg * torch.sum(W * W)


    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    margin_mask = margin
    margin_mask[margin > 0] = 1
    margin_sum = torch.sum(margin_mask, axis=1)
    margin_mask[idx, y] = -1*margin_sum
    dW = torch.mm(X.T, margin_mask)
    dW /= X.shape[0]
    dW += 2 * reg * W

    return loss, dW


def sample_batch(
    X: torch.Tensor, y: torch.Tensor, num_train: int, batch_size: int
):
    """
    Sample batch_size elements from the training data and their
    corresponding labels to use in this round of gradient descent.
    """
    X_batch = None
    y_batch = None

    # Store the data in X_batch and their corresponding labels in           #
    # y_batch; after sampling, X_batch should have shape (batch_size, dim)  #
    # and y_batch should have shape (batch_size,)                           #
    idx = torch.randint(0, y.shape[0], (batch_size,))
    X_batch = X[idx]
    y_batch = y[idx]
    return X_batch, y_batch


def train_linear_classifier(
    loss_func: Callable,
    W: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 1e-3,
    reg: float = 1e-5,
    num_iters: int = 100,
    batch_size: int = 200,
    verbose: bool = False,
):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - loss_func: loss function to use when training. It should take W, X, y
      and reg as input, and output a tuple of (loss, dW)
    - W: A PyTorch tensor of shape (D, C) giving the initial weights of the
      classifier. If W is None then it will be initialized here.
    - X: A PyTorch tensor of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Returns: A tuple of:
    - W: The final value of the weight matrix and the end of optimization
    - loss_history: A list of Python scalars giving the values of the loss at each
      training iteration.
    """
    # assume y takes values 0...K-1 where K is number of classes
    num_train, dim = X.shape
    if W is None:
        # lazily initialize W
        num_classes = torch.max(y) + 1
        W = 0.000001 * torch.randn(
            dim, num_classes, device=X.device, dtype=X.dtype
        )
    else:
        num_classes = W.shape[1]

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
        # TODO: implement sample_batch function
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

        # evaluate loss and gradient
        loss, grad = loss_func(W, X_batch, y_batch, reg)
        loss_history.append(loss.item())

        # perform parameter update
        W -= grad*learning_rate

        if verbose and it % 100 == 0:
            print("iteration %d / %d: loss %f" % (it, num_iters, loss))

    return W, loss_history


def predict_linear_classifier(W: torch.Tensor, X: torch.Tensor):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - W: A PyTorch tensor of shape (D, C), containing weights of a model
    - X: A PyTorch tensor of shape (N, D) containing training data; there are N
      training samples each of dimension D.

    Returns:
    - y_pred: PyTorch int64 tensor of shape (N,) giving predicted labels for each
      elemment of X. Each element of y_pred should be between 0 and C - 1.
    """
    y_pred = torch.zeros(X.shape[0], dtype=torch.int64)
    y_pred = torch.mm(X, W)
    y_pred = torch.argmax(y_pred, axis=1)
    return y_pred


def svm_get_search_params():
    """
    Return candidate hyperparameters for the SVM model. You should provide
    at least two param for each, and total grid search combinations
    should be less than 25.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    """

    learning_rates = []
    regularization_strengths = []

    learning_rates = [2e-3, 3e-3, 4e-4, 5e-4, 6e-4]
    regularization_strengths = [1e-3, 1e-2, 5e-3]

    return learning_rates, regularization_strengths


def test_one_param_set(
    cls: LinearClassifier,
    data_dict: Dict[str, torch.Tensor],
    lr: float,
    reg: float,
    num_iters: int = 2000,
):
    """
    Train a single LinearClassifier instance and return the learned instance
    with train/val accuracy.

    Inputs:
    - cls (LinearClassifier): a newly-created LinearClassifier instance.
                              Train/Validation should perform over this instance
    - data_dict (dict): a dictionary that includes
                        ['X_train', 'y_train', 'X_val', 'y_val']
                        as the keys for training a classifier
    - lr (float): learning rate parameter for training a SVM instance.
    - reg (float): a regularization weight for training a SVM instance.
    - num_iters (int, optional): a number of iterations to train

    Returns:
    - cls (LinearClassifier): a trained LinearClassifier instances with
                              (['X_train', 'y_train'], lr, reg)
                              for num_iter times.
    - train_acc (float): training accuracy of the svm_model
    - val_acc (float): validation accuracy of the svm_model
    """
    train_acc = 0.0  # The accuracy is simply the fraction of data points
    val_acc = 0.0  # that are correctly classified.

    # Write code that, train a linear SVM on the training set, compute its    #
    # accuracy on the training and validation sets                            #

    loss_history = cls.train(X_train=data_dict['X_train'], y_train=data_dict['y_train'], learning_rate=lr, reg=reg, num_iters=num_iters)
    y_train_pred = cls.predict(data_dict['X_train'])
    train_acc = 100.0 * (data_dict['y_train'] == y_train_pred).double().mean().item()
    y_val_pred = cls.predict(data_dict['X_val'])
    val_acc = 100.0 * (data_dict['y_val'] == y_val_pred).double().mean().item()

    return cls, train_acc, val_acc


# **************************************************#
################ Section 2: Softmax ################
# **************************************************#


def softmax_loss_naive(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Softmax loss function, naive implementation (with loops).  When you implment
    the regularization over W, please DO NOT multiply the regularization term by
    1/2 (no coefficient).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an tensor of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W)

    # Compute the softmax loss and its gradient using explicit loops.           #
    # Store the loss in loss and the gradient in dW.                            #

    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        scores = W.t().mv(X[i])
        correct_class_score = scores[y[i]]
        log_C = (-1)*torch.max(scores)
        scores_exp_sum = torch.sum(torch.exp(scores + log_C))
        
        for j in range(num_classes):
            dW[:, j] += X[i] * (torch.exp(scores[j] + log_C) / scores_exp_sum - int(j == y[i]))
        
        loss += -1 * torch.log(torch.exp(correct_class_score + log_C) / scores_exp_sum)

    loss /= num_train
    loss += reg * torch.sum(W * W)

    dW /= num_train
    dW += 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Softmax loss function, vectorized version.  When you implment the
    regularization over W, please DO NOT multiply the regularization term by 1/2
    (no coefficient).

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W)

    # Compute the softmax loss and its gradient using no explicit loops.        #
    # Store the loss in loss and the gradient in dW.                            #

    scores = torch.mm(X, W)
    idx = torch.arange(0, X.shape[0])
    correct_class_score = scores[idx, y]
    log_C, _ = torch.max(scores, axis=1)
    log_C *= -1

    scores_exp_sum = torch.sum(torch.exp(scores + log_C.unsqueeze(-1)), axis=1)

    loss = -1 * torch.sum(torch.log(torch.exp(correct_class_score + log_C) / scores_exp_sum))
    
    scores_exp = torch.exp(scores + log_C.unsqueeze(-1)) / scores_exp_sum.unsqueeze(-1)
    scores_exp[idx, y] -= 1
    dW = torch.mm(X.T, scores_exp)
    
    loss /= X.shape[0]
    loss += reg * torch.sum(W * W)

    dW /= X.shape[0]
    dW += 2 * reg * W

    return loss, dW


def softmax_get_search_params():
    """
    Return candidate hyperparameters for the Softmax model. You should provide
    at least two param for each, and total grid search combinations
    should be less than 25.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    """
    learning_rates = []
    regularization_strengths = []

    learning_rates = [1e-1, 2e-2, 4e-2, 6e-2, 8e-2,1e-2]
    regularization_strengths = [1e-3, 1e-4]

    return learning_rates, regularization_strengths


def svm_main():
    from utils.cifar import preprocess_cifar10
    import utils.utils
    
    # load cifar10 data
    utils.reset_seed(0)
    data_dict = preprocess_cifar10(bias_trick=True, cuda=True, dtype=torch.float64)

    # find the best learning_rates and regularization_strengths combination
    learning_rates, regularization_strengths = svm_get_search_params()
    num_models = len(learning_rates) * len(regularization_strengths)

    i = 0
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (train_acc, val_acc). 
    results = {}
    best_val = -1.0   # The highest validation accuracy that we have seen so far.
    best_svm_model = None # The LinearSVM object that achieved the highest validation rate.
    num_iters = 2000 # number of iterations

    for lr in learning_rates:
        for reg in regularization_strengths:
            i += 1
            print('Training SVM %d / %d with learning_rate=%e and reg=%e'
                    % (i, num_models, lr, reg))
    
            utils.reset_seed(0)

            cand_svm_model, cand_train_acc, cand_val_acc = test_one_param_set(LinearSVM(), data_dict, lr, reg, num_iters)

            if cand_val_acc > best_val:
                best_val = cand_val_acc
                best_svm_model = cand_svm_model # save the svm
            results[(lr, reg)] = (cand_train_acc, cand_val_acc)


    for lr, reg in sorted(results):
        train_acc, val_acc = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_acc, val_acc))
        
    print('best validation accuracy achieved during cross-validation: %f' % best_val)

    # save the best model
    path = os.path.join(MODEL_PATH, 'svm_best_model.pt')
    best_svm_model.save(path)

    # evaluation
    utils.reset_seed(0)
    y_test_pred = best_svm_model.predict(data_dict['X_test'])
    test_accuracy = torch.mean((data_dict['y_test'] == y_test_pred).double())
    print('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy)

def softmax_main():
    from utils.cifar import preprocess_cifar10
    import utils.utils
    
    # load cifar10 data
    utils.reset_seed(0)
    data_dict = preprocess_cifar10(bias_trick=True, cuda=True, dtype=torch.float64)
    
    # find the best learning_rates and regularization_strengths combination
    learning_rates, regularization_strengths = softmax_get_search_params()
    num_models = len(learning_rates) * len(regularization_strengths)

    i = 0
    # The keys should be tuples of (learning_rate, regularization_strength) and
    # the values should be tuples (train_acc, val_acc)
    results = {}
    best_val = -1.0   # The highest validation accuracy that we have seen so far.
    best_softmax_model = None # The Softmax object that achieved the highest validation rate.
    num_iters = 2000 # number of iterations

    for lr in learning_rates:
        for reg in regularization_strengths:
            i += 1
            print('Training Softmax %d / %d with learning_rate=%e and reg=%e'
                    % (i, num_models, lr, reg))
    
            utils.reset_seed(0)
            cand_softmax_model, cand_train_acc, cand_val_acc = test_one_param_set(Softmax(), data_dict, lr, reg, num_iters)

            if cand_val_acc > best_val:
                best_val = cand_val_acc
                best_softmax_model = cand_softmax_model # save the classifier
            results[(lr, reg)] = (cand_train_acc, cand_val_acc)


    # Print out results.
    for lr, reg in sorted(results):
        train_acc, val_acc = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_acc, val_acc))
    
    print('best validation accuracy achieved during cross-validation: %f' % best_val)

    # save the best model
    path = os.path.join(MODEL_PATH, 'softmax_best_model.pt')
    best_softmax_model.save(path)

    # evaluation
    y_test_pred = best_softmax_model.predict(data_dict['X_test'])
    test_accuracy = torch.mean((data_dict['y_test'] == y_test_pred).double())
    print('softmax on raw pixels final test set accuracy: %f' % (test_accuracy, ))


if __name__ == '__main__':
    exp = input('Choose: SVM or Softmax?')
    if exp == 'SVM':
        svm_main()
    elif exp == 'Softmax':
        softmax_main()
    else:
        raise NotImplementedError