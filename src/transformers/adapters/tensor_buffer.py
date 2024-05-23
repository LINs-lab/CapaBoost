# -*- coding: utf-8 -*-
import torch 
import torch.distributed as dist

def flatten(tensors, shapes=None, use_cuda=True):
    # init and recover the shapes vec.
    pointers = [0]
    if shapes is not None:
        for shape in shapes:
            pointers.append(pointers[-1] + shape[1])
    else:
        for tensor in tensors:
            pointers.append(pointers[-1] + tensor.nelement())

    # flattening.
    current_device = tensors[0].device
    target_device = tensors[0].device if tensors[0].is_cuda and use_cuda else "cpu"
    vec = torch.empty(pointers[-1], device=target_device)

    for tensor, start_idx, end_idx in zip(tensors, pointers[:-1], pointers[1:]):
        vec[start_idx:end_idx] = (
            tensor.data.view(-1).to(device=target_device)
            if current_device != target_device
            else tensor.data.view(-1)
        )
    return vec

class TensorBuffer:
    """
    Packs multiple tensors into one flat buffer for efficient
    intra-worker communication.
    """

    def __init__(self, tensors, use_cuda=True):
        indices = [0]
        for tensor in tensors:
            new_end = indices[-1] + tensor.nelement()
            indices.append(new_end)

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._tensors_len = len(tensors)
        self._tensors_sizes = [x.size() for x in tensors]

        self.buffer = flatten(tensors, use_cuda=use_cuda)  # copies

    def __getitem__(self, index):
        return self.buffer[self._start_idx[index] : self._end_idx[index]].view(
            self._tensors_sizes[index]
        )

    def __len__(self):
        return self._tensors_len

    def is_cuda(self):
        return self.buffer.is_cuda

    def nelement(self):
        return self.buffer.nelement()

    def unpack(self, tensors, param_names=None, param_to_be_ignored=None):
        param_names = (
            [(None, None)] * len(tensors) if param_names is None else param_names
        )
        for (_, param_name), tensor, entry in zip(param_names, tensors, self):
            if (
                param_name is None
                or param_to_be_ignored is None
                or (
                    param_to_be_ignored is not None
                    and param_to_be_ignored not in param_name
                )
            ):
                tensor.data[:] = entry