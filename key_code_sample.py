# Code adapted from https://github.com/microsoft/LoRA/blob/main/loralib/layers.py.
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import numpy as np
import torch
from src.transformers.adapters.configuration import LoRAConfig
from src.transformers.adapters.tensor_buffer import TensorBuffer

#  ------------------------------------------------------------------------------------------
#  The original CapaBoost idea should use one weight and several different masks. For convenience,
#  here we create an equivalent network with several tied-layers with same weights and different
#  masks. 
#
#  After initialization, all tied-layers have same weights and execute forward and backward pass
#  normally. After that, aggregate_grads_for_tied_weights will gather and sum up gradient of all
#  tied-layers and assign sum-up values back. Then, all tied-layers will execute gradient descent
#  and update values. All tied-layers will maintain same values all the time. 
#  ------------------------------------------------------------------------------------------

# only used in initialization
def init_masks_for_tied_weights(
    self, adapter_name, mask_density, rng_state=np.random.RandomState(0), sparsity_type='equal_per_layer', 
):  # net is a weight-tied block
    def _get_masks(layer):
        if sparsity_type == "equal_per_layer":
            mask = torch.zeros_like(layer.weight.view(-1))
            mask_length = len(mask)
            num_params_to_keep_per_layer = int(mask_length * mask_density)
            selected_keep_pos = rng_state.choice(
                np.arange(mask_length), num_params_to_keep_per_layer, replace=False
            )
            mask[selected_keep_pos] = 1
            return mask.view(layer.weight.size())
        # support NVIDIA sparse tensor core: N:M sparsity
        elif sparsity_type == "NM_structured": 
            N, M = 2, 4
            grads_abs = torch.randn(layer.weight.shape)

            group = int(grads_abs.numel()/M)
            weight_temp = grads_abs.detach().reshape(group, M)
            index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]
            w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
            w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(grads_abs.shape)

            mask = (
                w_b >= 1e-10
            ).float()
            return mask.view(layer.weight.size())
        else:
            raise NotImplementedError
    
    lora_config = self.config.adapters.match(
        adapter_name,
        config_type=LoRAConfig,
        layer_idx=self.layer_idx,
        location_key=self.location_key,
    )
    if adapter_name in self.loras:
        self.lora_A_masks[adapter_name] = []
        self.lora_B_masks[adapter_name] = []
        for i in range(lora_config.num_layer):
            self.lora_A_masks[adapter_name].append(_get_masks(self.loras[adapter_name].lora_A[i]))
            self.lora_B_masks[adapter_name].append(_get_masks(self.loras[adapter_name].lora_B[i]))


def get_latest_dense_tied_weights(self, adapter_name):
    if adapter_name in self.loras:
        self.lora_A_dense_tied_weights[adapter_name] = TensorBuffer(
            self.loras[adapter_name].lora_A[0].weight.data.clone()
        ).buffer
        self.lora_B_dense_tied_weights[adapter_name] = TensorBuffer(
            self.loras[adapter_name].lora_B[0].weight.data.clone()
        ).buffer

    
def restore_dense_tied_weights(self, adapter_name):
    if adapter_name in self.loras:
        lora_config = self.config.adapters.match(
            adapter_name,
            config_type=LoRAConfig,
            layer_idx=self.layer_idx,
            location_key=self.location_key,
        )
        for i in range(lora_config.num_layer):
        
            device = self.loras[adapter_name].lora_A[0].weight.device
            d_type = self.loras[adapter_name].lora_A[0].weight.dtype

            self.loras[adapter_name].lora_A[i].weight.data = (self.lora_A_dense_tied_weights[adapter_name].clone()).reshape(self.loras[adapter_name].lora_A[i].weight.data.shape).to(d_type).to(device)
            self.loras[adapter_name].lora_B[i].weight.data = (self.lora_B_dense_tied_weights[adapter_name].clone()).reshape(self.loras[adapter_name].lora_B[i].weight.data.shape).to(d_type).to(device)


def aggregate_grads_for_tied_weights(self, adapter_name, avg_tied_grads=False):
    # extract grads.
    if adapter_name in self.loras:
        lora_config = self.config.adapters.match(
            adapter_name,
            config_type=LoRAConfig,
            layer_idx=self.layer_idx,
            location_key=self.location_key,
        )
        buffers = []
        for i in range(lora_config.num_layer):
            hp_grad = self.loras[adapter_name].lora_A[i].weight.grad.clone()
            tb = TensorBuffer(
                [hp_grad]
            )
            buffers.append(tb.buffer)

        # aggregate grads.
        aggregated_grads = (
            sum(buffers) / len(buffers) if avg_tied_grads else sum(buffers)
        )

        # assign grads back (inplace).
        for i in range(lora_config.num_layer):
            grads = self.loras[adapter_name].lora_A[i].weight.grad
            tb = TensorBuffer(grads)
            tb.buffer = aggregated_grads.clone()
            tb.unpack(grads)

        #for B
        buffers = []
        for i in range(lora_config.num_layer):
            hp_grad = self.loras[adapter_name].lora_B[i].weight.grad.clone()
            tb = TensorBuffer(
                [hp_grad]
            )
            buffers.append(tb.buffer)
        # aggregate grads.
        aggregated_grads = (
            sum(buffers) / len(buffers) if avg_tied_grads else sum(buffers)
        )
        # assign grads back (inplace).
        for i in range(lora_config.num_layer):
            grads = self.loras[adapter_name].lora_B[i].weight.grad
            tb = TensorBuffer(grads)
            tb.buffer = aggregated_grads.clone()
            tb.unpack(grads)


def apply_masks_to_grads_of_tied_weights(self, adapter_name):
    if adapter_name in self.loras:
        lora_config = self.config.adapters.match(
            adapter_name,
            config_type=LoRAConfig,
            layer_idx=self.layer_idx,
            location_key=self.location_key,
        )
        for i in range(lora_config.num_layer):
            device = self.loras[adapter_name].lora_A[i].weight.device
            mask = self.lora_A_masks[adapter_name][i].to(device)
            if self.loras[adapter_name].lora_A[i].weight.grad is not None:
                self.loras[adapter_name].lora_A[i].weight.grad.data.mul_(mask)
            mask = self.lora_B_masks[adapter_name][i].to(device)
            if self.loras[adapter_name].lora_B[i].weight.grad is not None:
                self.loras[adapter_name].lora_B[i].weight.grad.data.mul_(mask)


def apply_masks_to_dense_tied_weights(self, adapter_name):

    if adapter_name in self.loras:
        lora_config = self.config.adapters.match(
            adapter_name,
            config_type=LoRAConfig,
            layer_idx=self.layer_idx,
            location_key=self.location_key,
        )
        for i in range(lora_config.num_layer):
            mask = self.lora_A_masks[adapter_name][i].to(self.loras[adapter_name].lora_A[i].weight.device)
            self.loras[adapter_name].lora_A[i].weight.data.mul_(mask)
            mask = self.lora_B_masks[adapter_name][i].to(self.loras[adapter_name].lora_B[i].weight.device)
            self.loras[adapter_name].lora_B[i].weight.data.mul_(mask)

