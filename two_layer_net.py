"""
Implements a two-layer Neural Network classifier in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import os
import torch
import random
import statistics
from linear_classifier import sample_batch
from typing import Dict, List, Callable, Optional

MODEL_PATH = './saved_models'

if torch.cuda.is_available:
    print('Good to go!')
else:
    print('No available GPU!')

class TwoLayerNet(object):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        std: float = 1e-4,
    ):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        - dtype: Optional, data type of each initial weight params
        - device: Optional, whether the weight params is on GPU or CPU
        - std: Optional, initial weight scaler.
        """
        # reset seed before start
        random.seed(0)
        torch.manual_seed(0)

        self.params = {}
        self.params["W1"] = std * torch.randn(
            input_size, hidden_size, dtype=dtype, device=device
        )
        self.params["b1"] = torch.zeros(hidden_size, dtype=dtype, device=device)
        self.params["W2"] = std * torch.randn(
            hidden_size, output_size, dtype=dtype, device=device
        )
        self.params["b2"] = torch.zeros(output_size, dtype=dtype, device=device)

    def loss(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        reg: float = 0.0,
    ):
        return nn_forward_backward(self.params, X, y, reg)

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        learning_rate: float = 1e-3,
        learning_rate_decay: float = 0.95,
        reg: float = 5e-6,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ):
        # fmt: off
        return nn_train(
            self.params, nn_forward_backward, nn_predict, X, y,
            X_val, y_val, learning_rate, learning_rate_decay,
            reg, num_iters, batch_size, verbose,
        )
        # fmt: on

    def predict(self, X: torch.Tensor):
        return nn_predict(self.params, nn_forward_backward, X)

    def save(self, path: str):
        torch.save(self.params, path)
        print("Saved in {}".format(path))

    def load(self, path: str):
        checkpoint = torch.load(path, map_location="cpu")
        self.params = checkpoint
        if len(self.params) != 4:
            raise Exception("Failed to load your checkpoint")

        for param in ["W1", "b1", "W2", "b2"]:
            if param not in self.params:
                raise Exception("Failed to load your checkpoint")
        # print("load checkpoint file: {}".format(path))


def nn_forward_pass(params: Dict[str, torch.Tensor], X: torch.Tensor):
    """
    The first stage of our neural network implementation: Run the forward pass
    of the network to compute the hidden layer features and classification
    scores. The network architecture should be:

    FC layer -> ReLU (hidden) -> FC layer (scores)

    As a practice, we will NOT allow to use torch.relu and torch.nn ops
    just for this time (you can use it from A3).

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - X: Input data of shape (N, D). Each X[i] is a training sample.

    Returns a tuple of:
    - scores: Tensor of shape (N, C) giving the classification scores for X
    - hidden: Tensor of shape (N, H) giving the hidden layer representation
      for each input value (after the ReLU).
    """
    # Unpack variables from the params dictionary
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    N, D = X.shape

    # Compute the forward pass
    hidden = None
    scores = None

    # Perform the forward pass, computing the class scores for the input.      #
    # Store the result in the scores variable, which should be an tensor of    #
    # shape (N, C).                                                            #

    fc_result = torch.mm(X, W1) + b1
    hidden = fc_result
    hidden[hidden < 0] = 0
    scores = torch.mm(hidden, W2) + b2

    return scores, hidden


def nn_forward_backward(
    params: Dict[str, torch.Tensor],
    X: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    reg: float = 0.0
):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network. When you implement loss and gradient, please don't forget to
    scale the losses/gradients by the batch size.

    Inputs: First two parameters (params, X) are same as nn_forward_pass
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a tensor scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    N, D = X.shape

    scores, h1 = nn_forward_pass(params, X)
    # If the targets are not given then jump out, we're done
    if y is None:
        return scores

    # Compute the loss
    loss = None

    # Compute the loss, based on the results from nn_forward_pass.             #
    # This should include both the data loss and L2 regularization for W1 and  #
    # W2.                                                                      #

    idx = torch.arange(0, X.shape[0])
    correct_class_score = scores[idx, y]
    log_C, _ = torch.max(scores, axis=1)
    log_C *= -1

    scores_exp_sum = torch.sum(torch.exp(scores + log_C.unsqueeze(-1)), axis=1)

    loss = -1 * torch.sum(torch.log(torch.exp(correct_class_score + log_C) / scores_exp_sum))
    loss /= N
    loss += reg*torch.sum(W1*W1) + reg*torch.sum(W2*W2)


    # Backward pass: compute gradients
    grads = {}
    # Compute the backward pass, computing the derivatives of the weights and #
    # biases. Store the results in the grads dictionary.                      #
    scores_exp = torch.exp(scores + log_C.unsqueeze(-1)) / scores_exp_sum.unsqueeze(-1)
    scores_exp[idx, y] -= 1
    dW2 = torch.mm(h1.T, scores_exp)
    dW2 /= N
    dW2 += 2 * reg * W2
    
    db2 = torch.mean(scores_exp, axis=0)
    
    dh = torch.mm(scores_exp, W2.T)
    dh[h1 == 0] = 0

    dW1 = torch.mm(X.T, dh)
    dW1 /= N
    dW1 += 2 * reg * W1
    
    db1 = torch.mean(dh, axis=0)
    
    grads['W2'] = dW2
    grads['b2'] = db2
    grads['W1'] = dW1
    grads['b1'] = db1

    return loss, grads


def nn_train(
    params: Dict[str, torch.Tensor],
    loss_func: Callable,
    pred_func: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    learning_rate: float = 1e-3,
    learning_rate_decay: float = 0.95,
    reg: float = 5e-6,
    num_iters: int = 100,
    batch_size: int = 200,
    verbose: bool = False,
):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - loss_func: a loss function that computes the loss and the gradients.
      It takes as input:
      - params: Same as input to nn_train
      - X_batch: A minibatch of inputs of shape (B, D)
      - y_batch: Ground-truth labels for X_batch
      - reg: Same as input to nn_train
      And it returns a tuple of:
        - loss: Scalar giving the loss on the minibatch
        - grads: Dictionary mapping parameter names to gradients of the loss with
          respect to the corresponding parameter.
    - pred_func: prediction function that im
    - X: A PyTorch tensor of shape (N, D) giving training data.
    - y: A PyTorch tensor of shape (N,) giving training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - X_val: A PyTorch tensor of shape (N_val, D) giving validation data.
    - y_val: A PyTorch tensor of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.

    Returns: A dictionary giving statistics about the training process
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

        # Compute loss and gradients using the current minibatch
        loss, grads = loss_func(params, X_batch, y=y_batch, reg=reg)
        loss_history.append(loss.item())

        params['W1'] -= learning_rate*grads['W1']
        params['W2'] -= learning_rate*grads['W2']
        params['b1'] -= learning_rate*grads['b1']
        params['b2'] -= learning_rate*grads['b2']

        if verbose and it % 100 == 0:
            print("iteration %d / %d: loss %f" % (it, num_iters, loss.item()))

        # Every epoch, check train and val accuracy and decay learning rate.
        if it % iterations_per_epoch == 0:
            # Check accuracy
            y_train_pred = pred_func(params, loss_func, X_batch)
            train_acc = (y_train_pred == y_batch).float().mean().item()
            y_val_pred = pred_func(params, loss_func, X_val)
            val_acc = (y_val_pred == y_val).float().mean().item()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            # Decay learning rate
            learning_rate *= learning_rate_decay

    return {
        "loss_history": loss_history,
        "train_acc_history": train_acc_history,
        "val_acc_history": val_acc_history,
    }


def nn_predict(
    params: Dict[str, torch.Tensor], loss_func: Callable, X: torch.Tensor
):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - loss_func: a loss function that computes the loss and the gradients
    - X: A PyTorch tensor of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A PyTorch tensor of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    scores, h1 = nn_forward_pass(params, X)
    y_pred = torch.argmax(scores, axis=1)

    return y_pred


def nn_get_search_params():
    """
    Return candidate hyperparameters for a TwoLayerNet model.
    You should provide at least two param for each, and total grid search
    combinations should be less than 256. If not, it will take
    too much time to train on such hyperparameter combinations.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - hidden_sizes: hidden value sizes, e.g. [8, 16, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    - learning_rate_decays: learning rate decay candidates
                                e.g. [1.0, 0.95, ...]
    """
    learning_rates = []
    hidden_sizes = []
    regularization_strengths = []
    learning_rate_decays = []

    learning_rates = [1e-3, 1e-2, 1e-1, 1e0]
    hidden_sizes = [2, 8, 32, 128]
    regularization_strengths = [0, 1e-5, 1e-3, 1e-1]
    learning_rate_decays = [1, 0.95, 0.9, 0.85]

    return (
        learning_rates,
        hidden_sizes,
        regularization_strengths,
        learning_rate_decays,
    )


def find_best_net(
    data_dict: Dict[str, torch.Tensor], get_param_set_fn: Callable
):
    """
    Tune hyperparameters using the validation set.
    Store your best trained TwoLayerNet model in best_net, with the return value
    of ".train()" operation in best_stat and the validation accuracy of the
    trained best model in best_val_acc. Your hyperparameters should be received
    from in nn_get_search_params

    Inputs:
    - data_dict (dict): a dictionary that includes
                        ['X_train', 'y_train', 'X_val', 'y_val']
                        as the keys for training a classifier
    - get_param_set_fn (function): A function that provides the hyperparameters
                                   (e.g., nn_get_search_params)
                                   that gives (learning_rates, hidden_sizes,
                                   regularization_strengths, learning_rate_decays)
                                   You should get hyperparameters from
                                   get_param_set_fn.

    Returns:
    - best_net (instance): a trained TwoLayerNet instances with
                           (['X_train', 'y_train'], batch_size, learning_rate,
                           learning_rate_decay, reg)
                           for num_iter times.
    - best_stat (dict): return value of "best_net.train()" operation
    - best_val_acc (float): validation accuracy of the best_net
    """

    best_net = None
    best_stat = None
    best_val_acc = 0.0
    
    learning_rates, hidden_sizes, regularization_strengths, learning_rate_decays = get_param_set_fn()
    for lr in learning_rates:
      for hs in hidden_sizes:
        for reg in regularization_strengths:
          for lr_d in learning_rate_decays:
            net = TwoLayerNet(input_size = 3 * 32 * 32, hidden_size = hs, output_size = 10, dtype=data_dict['X_train'].dtype, device=data_dict['X_train'].device)

            stats = net.train(data_dict['X_train'], data_dict['y_train'],
                              data_dict['X_val'], data_dict['y_val'],
                              num_iters=3000, batch_size=1000,
                              learning_rate=lr, learning_rate_decay=lr_d,
                              reg=reg, verbose=False)

            y_val_pred = net.predict(data_dict['X_val'])
            val_acc = 100.0 * (y_val_pred == data_dict['y_val']).double().mean().item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_net = net
                best_stat = stats
            
            print('train with learning rate: {}, hidden size: {}, regularization: {}, learning rate decays: {}, val accuracy: {}'.format(lr, hs, reg, lr_d, val_acc))

    return best_net, best_stat, best_val_acc


def main():
    from utils.cifar import preprocess_cifar10
    import utils.utils

    utils.reset_seed(0)
    data_dict = preprocess_cifar10(dtype=torch.float64)
    
    # find best parameters and net
    best_net, best_stat, best_val_acc = find_best_net(data_dict, nn_get_search_params)
    print(best_val_acc)

    # save model
    path = os.path.join(MODEL_PATH, 'nn_best_model.pt')
    best_net.save(path)
    
    # evaluation
    y_val_preds = best_net.predict(data_dict['X_val'])
    val_acc = 100 * (y_val_preds == data_dict['y_val']).double().mean().item()
    print('Best val-set accuracy: %.2f%%' % val_acc)

    # test
    y_test_preds = best_net.predict(data_dict['X_test'])
    test_acc = 100 * (y_test_preds == data_dict['y_test']).double().mean().item()
    print('Test accuracy: %.2f%%' % test_acc)


if __name__ == '__main__':
    main()