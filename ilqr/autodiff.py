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
from torch.autograd.functional import (jacobian, hessian)


def jacobian_scalar(func, inputs):
    """Computes the Jacobian of a scalar function with respect to inputs.

    Args:
        func: Python function that takes torch tensors as input and returns a torch scalar.
            NOTE: All computations in function must be tracked.
        input: tuple of torch tensor(s), each of shape (input[i].dim).

    Returns:
        tuple of torch tensor(s), each of shape (input[i].dim).
    """
    return jacobian(func, input)


def jacobian_vector(func, inputs):
    """Computes the Jacobian of a vector function with respect to inputs.

    Args:
        func: Python function that takes torch tensors as input and returns a torch vector.
            NOTE: All computations in function must be tracked (use torch.stack to concatenate).
        input: tuple of torch tensor(s), each of shape (input[i].dim).

    Returns:
        tuple of torch tensor(s), each of shape (output.dim, input[i].dim).
    """
    return jacobian(func, input)



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
