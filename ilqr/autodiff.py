# Copyright (C) 2018, Anass Al
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
"""Autodifferentiation helper methods."""
import torch
from torch.autograd.functional import jacobian
from torch.autograd.functional import hessian
from torch import autograd


def jacobian_scalar(func, inputs):
    """Computes the Jacobian of a scalar function with respect to inputs.
    Args:
        func: Python function that takes torch tensors as input and returns a torch scalar.
            NOTE: All computations in function must be tracked.
        input: tuple of torch tensor(s), each of shape (input[i].dim).
    Returns:
        tuple of torch tensor(s), each of shape (input[i].dim).
    """
    return jacobian(func, inputs)

def jacobian_scalar_test(func, inputs):
    outputs = func(*inputs)
    return autograd.grad(outputs, inputs)

def jacobian_vector(func, inputs):
    """Computes the Jacobian of a vector function with respect to inputs.
    Args:
        func: Python function that takes torch tensors as input and returns a torch vector.
            NOTE: All computations in function must be tracked (use torch.stack to concatenate).
        input: tuple of torch tensor(s), each of shape (input[i].dim).
    Returns:
        tuple of torch tensor(s), each of shape (output.dim, input[i].dim).
    """
    return jacobian(func, inputs)

def jacobian_vector_once(func, inputs):
    inputs = tuple(i.requires_grad_() for i in inputs)
    outputs = func(*inputs)
    grads = torch.stack(tuple(torch.cat(autograd.grad(o, inputs, allow_unused = True, create_graph = True, retain_graph = True)) for o in outputs))
    return grads

def batch_jacobian(func, inputs):
    """Computes the jacobian of function w.r.t. a batch of inputs.
    Args:
        func: Python function that takes torch tensors as input and returns a torch tensor.
            NOTE: All computations in function must be tracked (use torch.stack to concatenate outputs).
        inputs: tuple of tuples of torch tensors.
    Returns:
        tuple of tuples of torch tensors.
    """
    return tuple(jacobian(func, i) for i in inputs)


def hessian_scalar(func, inputs):
    """Computes the Hessian of a scalar function with respect to inputs.
    Args:
        func: Python function that takes torch tensors as input and returns a torch scalar.
            NOTE: All computations in function must be tracked.
        input: tuple of torch tensor(s), each of shape (input[i].dim).
    Returns:
        tuple of tuples of torch tensor(s), where Hessian[i][j] is a tensor of shape(input[i].dim, input[j].dim)
            containing the hessian of the i-th and j-th input
    """
    return hessian(func, inputs)


def hessian_vector(func, inputs, size):
    """Computes the Hessian of a vector function with respect to inputs.
    Args:
        func: Python function that takes torch tensors as input and returns a torch vector.
            NOTE: All computations in function must be tracked.
        input: tuple of torch tensor(s), each of shape (input[i].dim).
        size: number of dimensions of output of function
    Returns:
        tuple of tuples of torch tensor(s), where each tensor is of shape (output.dim, input[i], input[j].dim).
    """
    hessians = tuple(hessian(lambda *x : func(*x)[i], inputs) for i in range(size))
    hessians = tuple(tuple(torch.stack(tuple(hessians[k][i][j] for k in range(len(hessians)))) for j in range(len(inputs))) for i in range(len(inputs)))
    return hessians

def all_derivs_scalar(func, inputs):
    """
    Returns first and second derivatives of scalar func wrt inputs.
    """
    inputs = tuple(i.requires_grad_() for i in inputs)
    output = func(*inputs)
    first_derivs = torch.cat(autograd.grad(output, inputs, allow_unused = True, create_graph = True, retain_graph = True))
    second_derivs = torch.stack(tuple(torch.cat(autograd.grad(o, inputs, allow_unused = True, create_graph = True)) for o in first_derivs))
    return (first_derivs, second_derivs)

def as_function(expr, inputs, **kwargs):
    """Converts and optimizes a Theano expression into a function.
    Args:
        expr: Theano tensor expression.
        inputs: List of Theano variables to use as inputs.
        **kwargs: Additional key-word arguments to pass to `theano.function()`.
    Returns:
        A function.
    """
    return None
