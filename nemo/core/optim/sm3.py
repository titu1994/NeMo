# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.optim.optimizer import Optimizer

__all__ = ['SM3']


def _check_valid_opt_params(lr, momentum, eps, beta):
    if lr < 0:
        raise ValueError(f"Invalid learning rate: {lr}")
    if eps < 0:
        raise ValueError(f"Invalid epsilon value: {eps}")
    if not 0.0 <= momentum < 1.0:
        raise ValueError("Invalid momentum: {0}".format(momentum))
    if not (0.0 <= beta < 1.0):
        raise ValueError(f"Betas have to be between 0 and 1: {beta}")


class SM3(Optimizer):
    """ Implementation of the SM3 optimization algorithm.
    Refer to the paper [Memory Efficient Adaptive Optimization](https://papers.nips.cc/paper/2019/file/8f1fa0193ca2b5d2fa0695827d8270e9-Paper.pdf)

    This codebase ports the implementation from
        -   https://github.com/Enealor/PyTorch-SM3
        -   https://github.com/google-research/google-research/tree/master/sm3

    # NOTE:
        Please refer to the notes by the original implementation when utilizing
        `momentum` > 0 or `beta > 0`.

        - https://github.com/google-research/google-research/tree/master/sm3#advice-on-using-sm3-on-your-model

    # SM3 upper bounds the gradient square sums:
    #
    # To illustrate:
    #
    # For a Tensor `T` of shape [M, N, K].
    #
    # `G` be its gradient of shape [M, N, K]
    #
    # SM3 keeps around three accumulators A1, A2, A3 of size M, N, K
    # respectively.
    #
    # `A` be the accumulator of shape [M, N, K]. `A` is not materialized until
    #   its needed for every step, and is approximated by A1, A2, A3.
    #
    # At every gradient update step the accumulators satisify:
    #   A1_t[i] >= Sum_{s <= t} G_t[i, j, k]^2 for all j, k.
    #   A2_t[j] >= Sum_{s <= t} G_t[i, j, k]^2 for all i, k.
    #   A3_t[k] >= Sum_{s <= t} G_t[i, j, k]^2 for all i, j.
    #
    # The RHS is the gradient sum squares.
    #
    # For every step we materialize the tensor `A` based on accumulated tensors
    # A1, A2 and A3.
    #
    #  A = min(A1[i], A2[j], A3[j]) + G[i, j, k]^2
    #
    # SM3 preconditioned gradient is
    #
    #  preconditioned G = A^{-0.5} * G
    #
    # We then update the individual accumulator factors as:
    #
    #  A1[i] = max_{j, k} A[i, j, k]
    #  A2[j] = max_{i, k} A[i, j, k]
    #  A3[k] = max_{i, j} A[i, j, k]
    """

    def __init__(self, params, lr: float = 0.1, momentum: float = 0.0, beta: float = 0.0, eps: float = 1e-30):
        _check_valid_opt_params(lr=lr, momentum=momentum, eps=eps, beta=beta)
        defaults = {'lr': lr, 'momentum': momentum, 'beta': beta, 'eps': eps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            beta = group['beta']
            eps = group['eps']

            for p in group['params']:
                if p is None:
                    continue
                grad = p.grad

                state = self.state[p]
                shape = grad.shape
                rank = len(shape)

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p, device=p.device, dtype=p.dtype)
                    _add_initial_accumulators(state, grad)

                if grad.is_sparse:
                    update = self._sparse_update(grad, state, eps, beta)

                else:
                    # Get previous accumulators mu_{t-1}
                    if rank > 1:
                        acc_list = [state[_accumulator_key(i)] for i in range(rank)]
                    else:
                        acc_list = [state[_accumulator_key(0)]]

                    # Get update from accumulators and gradients
                    update = _compute_dense_update(beta, acc_list, grad)

                    # Update accumulators.
                    self._update_accumulator(beta, acc_list, update)

                    # Add small amount for numerical stability
                    update.add_(eps).rsqrt_().mul_(grad)

                    if momentum > 0.0:
                        g_bar = state['momentum_buffer']
                        # Original SM3 code uses :
                        # new_g := g_bar + g_bar * (momentum - 1.0) + scaled_g
                        # which resolves to the following :
                        # new_g := g_bar + g_bar * momentum - g_bar + scaled_g
                        # new_g := g_bar * momentum + scaled_g
                        # Therefore the collapsed update has been implemented below
                        update.mul_(1.0 - momentum).add_(g_bar, alpha=momentum)
                        state['momentum_buffer'] = update.detach()

                p.sub_(update, alpha=group['lr'])
                state['step'] += 1
        return loss

    def _sparse_update(self, grad, state, eps, beta):
        # the update is non-linear so indices must be unique
        grad.coalesce()
        grad_indices = grad._indices()
        grad_values = grad._values()

        # Transform update_values into sparse tensor
        def make_sparse(values):
            constructor = grad.new
            if grad_indices.dim() == 0 or values.dim() == 0:
                return constructor().resize_as_(grad)
            return constructor(grad_indices, values, grad.size())

        acc = state[_accumulator_key(0)]
        update_values = _compute_sparse_update(beta, acc, grad_values, grad_indices)
        self._update_sparse_accumulator(beta, acc, make_sparse(update_values))
        # Add small amount for numerical stability
        update_values.add_(eps).rsqrt_().mul_(grad_values)
        update = make_sparse(update_values)

        return update

    def _update_accumulator(self, beta, acc_list, update):
        # Updates individual accumulator factors as:
        #  A1[i] = max_{j, k} A[i, j, k]
        #  A2[j] = max_{i, k} A[i, j, k]
        #  A3[k] = max_{i, j} A[i, j, k]
        for i, acc in enumerate(acc_list):
            nu_max = _max_reduce_except_dim(update, i)
            if beta > 0.:
                torch.max(acc, nu_max, out=acc)
            else:
                # No need to compare - nu_max is bigger because of grad ** 2
                acc.copy_(nu_max)

    def _update_sparse_accumulator(self, beta, acc, update):
        nu_max = _max_reduce_except_dim(update.to_dense(), 0).squeeze()
        if beta > 0.:
            torch.max(acc, nu_max, out=acc)
        else:
            # No need to compare - nu_max is bigger because of grad ** 2
            acc.copy_(nu_max)


def _compute_sparse_update(beta, acc, grad_values, grad_indices):
    # In the sparse case, a single accumulator is used.
    # For sparse case, we only update the accumulator representing the sparse
    # dimension. In this case SM3 is similar to isotropic adagrad but with
    # better bound (due to the max operator).
    #
    # We do not use the column accumulator because it will updated for
    # every gradient step and will significantly overestimate the gradient
    # square. While, the row accumulator can take advantage of the sparsity
    # in the gradients. Even if one implements the column accumulator - it
    # will result in a no-op because the row accumulators will have lower
    # values.
    update_values = torch.gather(acc, 0, grad_indices[0])
    if beta > 0.0:
        update_values.mul_(beta)
    update_values.addcmul_(grad_values, grad_values, value=1.0 - beta)
    return update_values


def _compute_dense_update(beta, acc_list, grad):
    rank = len(acc_list)
    update = acc_list[0].clone()

    # Computes the accumulator before adding the current gradient.
    # A[i, j, k] = min(A1[i], A2[j], A3[j])
    for i in range(1, rank):
        # Compute the minimum accumulator value which is a tighter bound to the
        # gradient sum squares.
        #
        # Note: Here we are broadcasting to compute the minimum.
        # We rely on broadcasting to get the proper end shape.
        update = torch.min(update, acc_list[i])
    if beta > 0.0:
        update.mul_(beta)
    update.addcmul_(grad, grad, value=1.0 - beta)

    return update


def _accumulator_key(i):
    # Returns key used for accessing accumulators
    return 'accumulator_' + str(i)


def _add_initial_accumulators(state, grad):
    # Creates initial accumulators. For a dense tensor of shape (n1, n2, n3),
    # then our initial accumulators are of shape (n1, 1, 1), (1, n2, 1) and
    # (1, 1, n3). For a sparse tensor of shape (n, *), we use a single
    # accumulator of shape (n,).
    shape = grad.shape
    rank = len(shape)
    defaults = {'device': grad.device, 'dtype': grad.dtype}
    acc = {}

    if grad.is_sparse:
        acc[_accumulator_key(0)] = torch.zeros(shape[0], **defaults)
    elif rank == 0:
        # The scalar case is handled separately
        acc[_accumulator_key(0)] = torch.zeros(shape, **defaults)
    else:
        # Create accumulator slots for each dimension
        # Replaces a `shape` of [M, N, K] with 1 in all dimensions except for i.
        # For eg: i = 1 returns [1, N, 1].
        for i in range(rank):
            acc_shape = [1] * i + [shape[i]] + [1] * (rank - 1 - i)
            acc[_accumulator_key(i)] = torch.zeros(acc_shape, **defaults)

    state.update(acc)


def _max_reduce_except_dim(tensor, dim):
    # Computes max along all dimensions except the given dim.
    # If tensor is a scalar, it returns tensor.
    rank = len(tensor.shape)
    result = tensor
    if rank > 0:
        assert dim < rank
        for d in range(rank):
            if d != dim:
                result = result.max(dim=d, keepdim=True).values
    return result
