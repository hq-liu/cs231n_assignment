import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  C = W.shape[1]
  for i in range(N):
    f = np.dot(X[i], W)
    f -= np.max(f)
    fy = f[y[i]]
    exp_sum = np.sum(np.exp(f))
    loss += -fy + np.log(exp_sum)

    dW[:,y[i]] -= X[i]
    for j in range(C):
       dW[:, j] += (np.exp(f[j])/exp_sum) * X[i]
  loss /= N
  dW /= N

  loss += reg * np.sum(W * W)
  dW += 2 * reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]

  f = X.dot(W)
  f -= np.max(f, axis=1).reshape(N, 1)
  exp_sum = np.sum(np.exp(f),axis=1)
  loss += np.sum(np.log(exp_sum)) - np.sum(f[range(N), y])

  counts = np.exp(f) / exp_sum.reshape(N, 1)
  counts[range(N), y] -= 1
  dW = np.dot(X.T, counts)

  loss /= N
  dW /= N

  loss += reg * np.sum(W * W)
  dW += 2 * reg * W



  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

