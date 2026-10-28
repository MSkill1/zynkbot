"""
2026.10.9
2026.10.9
5.17.0
1.15.0
__UNSLOTH_VERSIONING__
"""

# Unsloth auto generated code
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from unsloth_zoo.temporary_patches.common import torch_compile
from unsloth_zoo.temporary_patches.common import _maybe_compile
import functools
import transformers
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
from trl.trainer.distillation_trainer import (Any, AutoProcessor, BaseTunerLayer, BitsAndBytesConfig, Callable, DataLoader, Dataset, DistillationConfig, DistillationTrainer, DistributedBackend, GenerationConfig, IterableDataset, Path, PeftConfig, PeftModel, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, PromptLearningConfig, RepeatSampler, Sampler, TrainerCallback, VLLMGeneration, Version, _BaseTrainer, _ForwardRedirection, _SUPPORTS_RESPONSE_TEMPLATE, _chunked_divergence_loss, add_response_schema, apply_chat_template, copy, create_model_from_path, defaultdict, deque, disable_dropout_in_model, gather_object, get_config_model_id, get_peft_model, get_training_chat_template, identity, inspect, is_chat_template_prefix_preserving, is_conversational, is_jmespath_available, is_peft_available, is_peft_model, is_rich_available, is_vllm_available, logger, math, np, os, pad, parse_response, prepare_deepspeed, prepare_multimodal_messages, print_prompt_completions_sample, profiling_context, profiling_decorator, repeat_iterable_dataset, set_seed, shuffle_sequence_dict, split_pixel_values_by_grid, split_tensor_dict, supports_tool_calling, sys, textwrap, time, torch, transformers, unsplit_pixel_values_by_grid, unwrap_model_for_generation, warnings, AutoProcessor, BaseTunerLayer, BitsAndBytesConfig, Callable, Dataset, DistillationConfig, DistillationTrainer, DistributedBackend, GenerationConfig, IterableDataset, PeftConfig, PeftModel, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, PromptLearningConfig, RepeatSampler, Sampler, TrainerCallback, VLLMGeneration, Version, add_response_schema, copy, create_model_from_path, defaultdict, deque, disable_dropout_in_model, get_config_model_id, get_peft_model, get_training_chat_template, identity, inspect, is_chat_template_prefix_preserving, is_jmespath_available, is_peft_available, is_peft_model, is_vllm_available, logger, np, os, pad, prepare_deepspeed, repeat_iterable_dataset, set_seed, supports_tool_calling, sys, time, torch, transformers, warnings, Any, np, profiling_decorator, shuffle_sequence_dict, split_pixel_values_by_grid, split_tensor_dict, torch, unsplit_pixel_values_by_grid, PeftModel, PreTrainedModel, is_peft_available, logger, os, torch)


import os
import math
import logging
from typing import *
from dataclasses import dataclass, field
from packaging.version import Version
import torch
import numpy as np
from contextlib import nullcontext
from torch.nn import functional as F
import inspect
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling as TransformersDataCollatorForLanguageModeling
from transformers.training_args import ParallelMode
from unsloth_zoo.device_type import DEVICE_TYPE, DEVICE_TYPE_TORCH, device_synchronize

# Wrap trainer with padding to right and enable training mode
import functools
from types import MethodType
try:
    from unsloth_zoo.gradient_checkpointing import reset_unsloth_gradient_checkpointing_buffers
except:
    def reset_unsloth_gradient_checkpointing_buffers(): pass
# Canonical reset lives in unsloth.models._utils so the SFT auto-packing wrapper and the plain
# Trainer loop can import the same helper; fall back to a no-op only if it can't be imported.
try:
    from unsloth.models._utils import _unsloth_reset_stray_compile_cache
except Exception:
    def _unsloth_reset_stray_compile_cache(self): pass
try:
    from unsloth.models._utils import _unsloth_dataset_column_names
except Exception:
    def _unsloth_dataset_column_names(dataset): return dataset.column_names
# Drops/renames config arguments the installed TRL no longer accepts, so a
# script pinned to an older TRL keeps working after an upgrade. Falls back to
# the historical raw passthrough so this can never break trainer construction.
try:
    from unsloth.models.rl_config_compat import filter_config_init_kwargs as _unsloth_filter_config_init_kwargs
    # A cache file generated here can be imported by an older Unsloth whose filter
    # predates `mirrored_from`, so drop the argument rather than raise TypeError.
    if "mirrored_from" not in inspect.signature(_unsloth_filter_config_init_kwargs).parameters:
        _unsloth_filter_config_init_kwargs_old = _unsloth_filter_config_init_kwargs
        def _unsloth_filter_config_init_kwargs(config_class, kwargs, **kw):
            return _unsloth_filter_config_init_kwargs_old(config_class, kwargs)
except Exception:
    def _unsloth_filter_config_init_kwargs(config_class, kwargs, **kw): return kwargs
def prepare_for_training_mode(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        # Drop any torch.compile graph cache poisoned by a stray pre-train forward.
        try:
            _unsloth_reset_stray_compile_cache(self)
        except Exception:
            pass
        # Finish the previous W&B run if this is a subsequent train() call.
        # We do this at the START of train() (not the end) so that
        # evaluate() / log() still work after train() completes.
        # HF's WandbCallback.setup() will call wandb.init() for the new run.
        # See: https://github.com/unslothai/unsloth/issues/3954
        if getattr(self, '_unsloth_training_completed', False):
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
                    # Reset HF's WandbCallback so it calls wandb.init() for the new run
                    for cb in self.callback_handler.callbacks:
                        if type(cb).__name__ == 'WandbCallback':
                            cb._initialized = False
                            break
            except:
                pass
        # Enable training mode
        _was_training = None
        # Restore the GC mode the model was configured with at setup; fall back to
        # the training args only when it wasn't recorded (issue #4735). Use hasattr,
        # not a None sentinel, so a deliberately-recorded None is restored verbatim.
        _model = getattr(self, 'model', None)
        if hasattr(_model, '_unsloth_gradient_checkpointing'):
            use_gc = _model._unsloth_gradient_checkpointing
        else:
            use_gc = getattr(self.args, 'gradient_checkpointing', True)
        if hasattr(self, 'model') and hasattr(self.model, "training"):
            _was_training = self.model.training
        if hasattr(self, 'model') and hasattr(self.model, "for_training"):
            self.model.for_training(use_gradient_checkpointing=use_gc)
        output = f(self, *args, **kwargs)
        # Restore previous mode when possible
        if hasattr(self, 'model') and hasattr(self.model, "for_inference"):
            if _was_training is False:
                self.model.for_inference()
            elif _was_training is True and hasattr(self.model, "for_training"):
                self.model.for_training(use_gradient_checkpointing=use_gc)
        # Reset gradient checkpointing buffers to free memory while staying ready for next run
        try:
            reset_unsloth_gradient_checkpointing_buffers()
        except:
            pass
        # Mark that training completed so the next train() call can
        # finish this W&B run before starting a new one
        self._unsloth_training_completed = True
        return output
    return wrapper
pass

torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : False,
    "shape_padding"     : True,
    "trace.enabled"     : False,
    "triton.cudagraphs" : False,
}

@_maybe_compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def chunked_hidden_states_selective_log_softmax(
    hidden_states: torch.Tensor,
    lm_head: torch.Tensor,
    index: torch.Tensor,
    chunks: int = 4,
    logit_scale_multiply: float = 0.0,
    logit_scale_divide: float = 0.0,
    logit_softcapping: float = 0.0,
    temperature: float = 1.0,
    # Rows per chunk cap. Read HERE, in the default, not in the body: the body
    # is traced with fullgraph = True, and `os.environ` is an unsupported op
    # there on torch 2.4 -- Dynamo raises
    #   torch._dynamo.exc.Unsupported: const method call bytes.decode
    # from os._Environ.__getitem__, which the eager fallback does not catch
    # (it only catches recompile-limit and disabled-hook breaks), so the very
    # first call would die even with the variable unset. A default is evaluated
    # once when the def runs, which is import time, outside any traced region.
    # It is also a plain int argument, so Dynamo guards on it instead of
    # constant-folding an unguarded read (2.7+ never notice a later change).
    # A non-numeric value is ignored rather than raised on, so a typo cannot
    # break the import. 0 keeps the previous chunk boundaries exactly.
    max_rows_per_chunk: int = (
        int(os.environ.get("UNSLOTH_GRPO_MAX_ROWS_PER_CHUNK", "0").strip())
        if os.environ.get("UNSLOTH_GRPO_MAX_ROWS_PER_CHUNK", "0").strip().isdigit()
        else 0
    ),
    lm_head_bias: Optional[torch.Tensor] = None,
    support_top_k: int = 0,
    support_top_p: float = 1.0,
    support_min_p: float = 0.0,
    support_triton_sampler: bool = False,
    support_excluded_ids: Optional[torch.Tensor] = None,
    # Read in the default for the same Dynamo reason as max_rows_per_chunk. Pairs with vLLM's
    # head_dtype=float32 that load_vllm requests under the same flag.
    fp32_lm_head: bool = (
        os.environ.get("UNSLOTH_VLLM_FP32_LM_HEAD", "0").strip().lower() in ("1", "true", "on", "yes")
    ),
    # Older Unsloth scores old / ref log-probs here and the policy with the fused kernel: one scorer keeps ratio at 1.
    fused_linear: bool = (
        os.environ.get("UNSLOTH_FUSED_LINEAR", "1") != "0"
        and os.environ.get("UNSLOTH_GRPO_CHUNKED_LOGPROBS", "1") != "0"
    ),
) -> torch.Tensor:
    # All Unsloth Zoo code licensed under AGPL3
    from unsloth_zoo.fused_losses import chunked_logprob as _chunked_logprob
    if (fused_linear and _chunked_logprob._FUSED_LINEAR and not fp32_lm_head
            and support_top_k == 0 and support_top_p == 1.0 and support_min_p == 0.0
            and support_excluded_ids is None):
        return _chunked_logprob.unsloth_chunked_logprobs(
            hidden_states, lm_head, index,
            temperature = temperature,
            logit_scale_multiply = logit_scale_multiply,
            logit_scale_divide = logit_scale_divide,
            logit_softcapping = logit_softcapping,
            lm_head_bias = lm_head_bias,
            max_rows_per_chunk = max_rows_per_chunk if max_rows_per_chunk > 0 else None,
        ).reshape((hidden_states.shape[0], hidden_states.shape[1]))
    # Reshape on this tensor's own last dim: a no-op, so a wrong-width caller
    # cannot have its row count silently rewritten and instead fails at the
    # matmul below, which prints both operands. Do not swap in a bare
    # torch._check: it reports only "Expected cond to be True", naming neither
    # operand, and Dynamo rejects a message-carrying one. Callers dispatch on
    # the width first -- see `compute_logprobs_chunk`, the packed path and
    # `_pg_grad_forward`.
    flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_index = index.reshape(-1)

    # Each chunk materialises rows x vocab logits and then a float32 copy of
    # them, all on the device holding the output head. With a large vocabulary
    # and a fixed chunk count that grows with the batch, so the peak scales with
    # the batch rather than staying bounded. max_rows_per_chunk caps the rows
    # per chunk instead, which is pure loop splitting: more, smaller chunks,
    # same concatenated result. 0 (the default) keeps the previous chunk
    # boundaries exactly.
    if max_rows_per_chunk > 0:
        n_rows = flat_hidden_states.shape[0]
        chunks = max(chunks, -(-n_rows // max_rows_per_chunk))
        chunks = min(chunks, max(n_rows, 1))

    # A one-row chunk's head gradient is a K=1 matmul Inductor mis-lowers under dynamic shapes (illegal
    # memory access or NaN head gradients, torch 2.13), so keep chunks at 2+ rows when the head trains.
    # Exactly `chunks` row ranges, so the unrolled loop length never depends on T: torch.chunk
    # returns ceil(T / ceil(T / chunks)) pieces and recompiled per new count; tensor_split guards
    # on T % chunks. At most rows // 2 chunks when the head trains keeps every piece at 2+ rows.
    lm_head_grad = torch.is_grad_enabled() and lm_head.requires_grad
    if lm_head_grad and flat_hidden_states.shape[0] < 2 * chunks:
        chunks = max(flat_hidden_states.shape[0] // 2, 1)

    n_rows = flat_hidden_states.shape[0]
    chunks = max(1, min(chunks, n_rows))
    bounds = [(i * n_rows) // chunks for i in range(chunks + 1)]
    chunked_hidden_states = [flat_hidden_states[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    chunked_index = [flat_index[a:b] for a, b in zip(bounds[:-1], bounds[1:])]

    # Cast once, outside the loop, so every chunk shares one fp32 head copy (also for autograd).
    if fp32_lm_head:
        lm_head = lm_head.to(torch.float32)
        if lm_head_bias is not None:
            lm_head_bias = lm_head_bias.to(torch.float32)

    all_per_token_logps = []

    for chunk_hidden_states, chunk_index in zip(chunked_hidden_states, chunked_index):
        # When the model is dispatched over several devices, the output head can
        # sit on a different one from the hidden states, because accelerate
        # places the tail of the model on the last device it fills. Co-locate on
        # the head's device before the matmul, otherwise this raises
        #   Unhandled FakeTensor Device Propagation for aten.mm.default,
        #   found two different devices cuda:0, cuda:1
        # On a single device every .to() here is a no-op and the result is
        # bit-identical to before.
        chunk_hidden_states = chunk_hidden_states.to(device = lm_head.device, dtype = lm_head.dtype)
        chunk_index = chunk_index.to(lm_head.device)
        if fp32_lm_head:
            # The GRPO loss runs this under bf16 / fp16 autocast, which would downcast the matmul.
            with torch.autocast(device_type = chunk_hidden_states.device.type, enabled = False):
                chunk_logits = torch.nn.functional.linear(chunk_hidden_states, lm_head, lm_head_bias)
        elif lm_head_bias is None:
            chunk_logits = chunk_hidden_states @ lm_head.t()
        else:
            chunk_logits = torch.nn.functional.linear(chunk_hidden_states, lm_head, lm_head_bias)

        if logit_scale_multiply != 0.0:
            chunk_logits = chunk_logits * logit_scale_multiply
        if logit_scale_divide != 0.0:
            chunk_logits = chunk_logits / logit_scale_divide
        if logit_softcapping != 0.0:
            chunk_logits = logit_softcapping * torch.tanh(chunk_logits / logit_softcapping)

        chunk_logits = chunk_logits.to(torch.float32)

        if temperature != 1.0:
            chunk_logits = chunk_logits / temperature

        if support_top_k > 0 or support_top_p < 1.0 or support_min_p > 0.0 or support_excluded_ids is not None:
            per_token_logps = _truncated_support_logps(
                chunk_logits, chunk_index, support_top_k, support_top_p, support_min_p,
                support_triton_sampler, support_excluded_ids,
            )
        else:
            # Target excluded from the normaliser: x_i - logsumexp(x) drops confident tokens (softmax collapse).
            is_target = torch.arange(chunk_logits.shape[-1], device = chunk_logits.device) == chunk_index.unsqueeze(-1)
            selected_logits = torch.where(is_target, chunk_logits, 0.0).sum(-1)
            others = torch.where(is_target, torch.finfo(chunk_logits.dtype).min, chunk_logits)
            per_token_logps = -torch.nn.functional.softplus(torch.logsumexp(others, dim = -1) - selected_logits)
        # Return to the caller's device so the concatenation below and every
        # downstream consumer see the device they started on.
        all_per_token_logps.append(per_token_logps.to(hidden_states.device))

    all_per_token_logps = torch.concat(all_per_token_logps)

    all_per_token_logps = all_per_token_logps.reshape((hidden_states.shape[0], hidden_states.shape[1]))
    return all_per_token_logps

@_maybe_compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def chunked_selective_log_softmax(
    logits,
    index,
    temperature: float = 1.0,
    chunks: int = 4,
    support_top_k: int = 0,
    support_top_p: float = 1.0,
    support_min_p: float = 0.0,
    support_triton_sampler: bool = False,
    support_excluded_ids: Optional[torch.Tensor] = None,
):
    # Fixed `chunks` row ranges; see chunked_hidden_states_selective_log_softmax.
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_index = index.reshape(-1)
    # Fewer rows than chunks: one row per range, as torch.chunk did, never empty ranges.
    chunks = max(1, min(chunks, flat_logits.shape[0]))
    bounds = [(i * flat_logits.shape[0]) // chunks for i in range(chunks + 1)]
    chunked_logits = [flat_logits[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    chunked_index  = [flat_index[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    all_per_token_logps = []
    # Per-chunk selective_log_softmax.
    for chunk_logits, chunk_index in zip(chunked_logits, chunked_index):
        chunk_logits = chunk_logits.to(torch.float32)
        if temperature != 1.0:
            chunk_logits = chunk_logits / temperature
        if support_top_k > 0 or support_top_p < 1.0 or support_min_p > 0.0 or support_excluded_ids is not None:
            all_per_token_logps.append(_truncated_support_logps(
                chunk_logits, chunk_index, support_top_k, support_top_p, support_min_p,
                support_triton_sampler, support_excluded_ids,
            ))
            continue
        # Target excluded from the normaliser: x_i - logsumexp(x) drops confident tokens (softmax collapse).
        is_target = torch.arange(chunk_logits.shape[-1], device = chunk_logits.device) == chunk_index.unsqueeze(-1)
        selected_logits = torch.where(is_target, chunk_logits, 0.0).sum(-1)
        others = torch.where(is_target, torch.finfo(chunk_logits.dtype).min, chunk_logits)
        per_token_logps = -torch.nn.functional.softplus(torch.logsumexp(others, dim = -1) - selected_logits)
        all_per_token_logps.append(per_token_logps)
    pass
    all_per_token_logps = torch.concat(all_per_token_logps)
    all_per_token_logps = all_per_token_logps.reshape((logits.shape[0], logits.shape[1]))
    return all_per_token_logps

def calculate_pad_tokens_in_prompt(
    input_ids: torch.Tensor,
    logits_to_keep: int,
    pad_token_id: int
) -> torch.Tensor:
    """Count left-padded tokens per sequence, e.g. [pad, pad, pad, cat] -> 3."""
    if logits_to_keep >= input_ids.shape[1]:
        raise ValueError("logits_to_keep must be smaller than the sequence length.")

    prompt_section = input_ids[:, :-logits_to_keep]

    padding_mask = (prompt_section == pad_token_id)

    pad_token_counts = padding_mask.sum(dim=1)

    return pad_token_counts

def create_completion_attention_mask(
    completion_input_ids: torch.Tensor,
    left_pad_tokens_per_prompt: torch.Tensor,
    max_left_pad: int,
    pad_token_id: int
) -> torch.Tensor:
    """Build a completion mask that zeros leading prompt and trailing pad tokens.

    For [p,p,p,c,c,c,pad,pad,pad] (p=sliced prompt, c=completion, pad=padding)
    this returns [0,0,0,1,1,1,0,0,0].
    """
    batch_size, completion_len = completion_input_ids.shape
    device = completion_input_ids.device

    num_tokens_to_mask = max_left_pad - left_pad_tokens_per_prompt

    indices = torch.arange(completion_len, device=device).unsqueeze(0)
    shift_mask = indices >= num_tokens_to_mask.unsqueeze(1)

    non_padding_mask = (completion_input_ids != pad_token_id)

    final_mask = shift_mask & non_padding_mask

    return final_mask

def left_pack_padding(tensor: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Move all padding tokens in each sequence to the right."""
    mask = (tensor != pad_id)
    # stable=True since the binary mask is unordered.
    sorted_indices = torch.argsort(mask, dim=1, descending=True, stable=True)
    packed_tensor = torch.gather(tensor, 1, sorted_indices)
    return packed_tensor

def align_logprobs_with_mask(
    logprob_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_value: float = 0.0
) -> torch.Tensor:
    """Align a log probability tensor with a given attention mask."""

    device = logprob_tensor.device
    batch_size, logprob_seq_len = logprob_tensor.shape
    mask_seq_len = attention_mask.shape[1]

    padded_logprobs = torch.full(
        attention_mask.shape,
        fill_value=pad_value,
        dtype=logprob_tensor.dtype,
        device=device
    )

    left_pad_counts = torch.argmax(attention_mask, dim=1)

    cols = torch.arange(logprob_seq_len, device=device)
    dest_indices = left_pad_counts.unsqueeze(1) + cols

    # Destination row indices, shape [batch_size, logprob_seq_len].
    row_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(dest_indices)

    # Keep only in-bounds destinations, then scatter via advanced indexing.
    valid_mask = dest_indices < mask_seq_len
    valid_rows = row_indices[valid_mask]
    valid_cols = dest_indices[valid_mask]
    valid_vals = logprob_tensor[valid_mask]
    padded_logprobs[valid_rows, valid_cols] = valid_vals

    return padded_logprobs

def align_completion_tool_mask(
    tool_mask: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Align a raw completion-length tool/env mask with Unsloth's repacked loss mask."""
    if tool_mask is None:
        return completion_mask
    if tool_mask.shape[0] != completion_mask.shape[0]:
        raise ValueError("tool_mask batch size must match completion_mask batch size.")

    tool_mask = tool_mask.to(device=completion_mask.device)
    if tool_mask.shape == completion_mask.shape:
        aligned_tool_mask = tool_mask
    else:
        aligned_tool_mask = align_logprobs_with_mask(
            tool_mask,
            completion_mask,
            pad_value=0,
        )
    return completion_mask * aligned_tool_mask.to(dtype=completion_mask.dtype)

def autotune_batch_and_chunks(
    total_input_rows,
    seq_len,
    hidden_size,
    vocab_size,
    dtype_bytes=16,
    multiplier=None
):
    if multiplier is None:
        final_m = max(4, seq_len // 4096)
    else:
        final_m = multiplier

    if torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
        limit_gb = (free_bytes / (1024**3))*.80
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        # XPU: estimate free memory as total - reserved.
        total_mem = torch.xpu.get_device_properties(0).total_memory
        reserved_mem = torch.xpu.memory_reserved()
        free_bytes = total_mem - reserved_mem
        limit_gb = (free_bytes / (1024**3)) * 0.80
    else:
        # Fallback: assume 8GB available.
        limit_gb = 8.0

    bytes_to_gb = 1024**3

    b_vals = torch.arange(total_input_rows, 0, -1, device='cpu', dtype=torch.float32)

    hidden_gb = (b_vals * seq_len * hidden_size * dtype_bytes) / bytes_to_gb

    base_logits = ((b_vals/total_input_rows) * b_vals * seq_len * vocab_size * dtype_bytes) / bytes_to_gb
    logits_gb = base_logits / final_m

    total_mem_gb = hidden_gb + logits_gb

    valid_mask = total_mem_gb <= limit_gb
    valid_indices = torch.nonzero(valid_mask, as_tuple=False)

    if valid_indices.shape[0] == 0:
        #This means your GPU will OOM
        # Capped at the row count: unsloth's no-grad pass divides rows by this without max(1, ...).
        return max(1, min(4, total_input_rows)), final_m

    best_idx = valid_indices[0].item()
    final_b = int(b_vals[best_idx].item())

    return final_b, final_m

def sanitize_logprob(logprob):
    """Local port of trl.scripts.vllm_serve.sanitize_logprob.
    Filters NaN logprobs from vLLM outputs."""
    value = logprob.logprob
    if math.isnan(value):
        logging.getLogger(__name__).warning(
            f"Generated NaN logprob, token logprob '{logprob}' will be ignored"
        )
        return None
    return value
@dataclass
class UnslothDistillationConfig(DistillationConfig):
    """
    
    Configuration class for the [`DistillationTrainer`].

    Extends [`~transformers.TrainingArguments`] with parameters specific to knowledge distillation. All necessary
    fields are declared here.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model and the teacher model

        model_init_kwargs (`str` or `dict[str, Any]`, *optional*):
            Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of the trainer
            is provided as a string. The `revision` value is also used when loading the processing class.
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to
            [`~transformers.AutoModelForCausalLM.from_pretrained`] and [`~transformers.AutoTokenizer.from_pretrained`],
            for both the student and teacher.
        teacher_model_name_or_path (`str`, *optional*):
            Model name or path for the teacher model. Used when the teacher is loaded locally.
        teacher_model_revision (`str`, *optional*):
            Model revision of the teacher model (e.g., branch name, tag, or commit hash).
        teacher_model_init_kwargs (`str` or `dict[str, Any]`, *optional*):
            Keyword arguments passed to `AutoModelForCausalLM.from_pretrained` when instantiating the teacher model
            from a string.
        disable_dropout (`bool`, *optional*, defaults to `False`):
            Whether to disable dropout in the student model during training.

        > Parameters that control the data preprocessing

        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the column `"prompt"` in the dataset. The trainer consumes the raw prompt column and
            generates completions on-policy, so it defaults to `False`.
        max_completion_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum number of tokens to generate per completion during on-policy generation.
        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible
            with vLLM generation.
        shuffle_dataset (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training dataset.
        pad_to_multiple_of (`int`, *optional*):
            If set, the prompts ids and completions ids will be padded to a multiple of this value.

        > Parameters that control generation

        temperature (`float`, *optional*, defaults to `1.0`):
            Temperature for sampling during generation and for computing the distillation loss. Higher values produce
            softer probability distributions.
        top_p (`float`, *optional*, defaults to `1.0`):
            Top-p (nucleus) sampling parameter for on-policy generation.
        top_k (`int`, *optional*, defaults to `0`):
            Top-k sampling parameter for on-policy generation. `0` disables top-k filtering.
        min_p (`float`, *optional*):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between `0.0` and `1.0`. Typical values are in the `0.01-0.2` range.
        generation_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to [`~transformers.GenerationConfig`] (if using transformers) or
            `SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the
            generation behavior, such as setting `suppress_tokens`, `num_beams`, etc. If it contains keys that conflict
            with the other generation parameters (like `min_p`, `top_p`, etc.), they will override them.
        chat_template_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the `apply_chat_template` function when generating completions.
        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far.
            Values > `1.0` encourage the model to use new tokens, while values < `1.0` encourage the model to repeat
            tokens.
        cache_implementation (`str`, *optional*):
            Implementation of the cache method for faster generation when `use_vllm` is set to `False`.

        > Parameters that control generation acceleration powered by vLLM

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating on-policy completions from the student model.
        vllm_mode (`str`, *optional*, defaults to `"colocate"`):
            Mode for student vLLM integration. Either `"server"` or `"colocate"`.
        vllm_model_impl (`str`, *optional*, defaults to `"vllm"`):
            Model implementation backend for vLLM. Use `"vllm"` or `"transformers"`.
        vllm_enable_sleep_mode (`bool`, *optional*, defaults to `False`):
            Enable vLLM sleep mode to offload student weights during the optimizer step.
        vllm_structured_outputs_regex (`str`, *optional*):
            Regex pattern for vLLM structured outputs.

        > Parameters that control the vLLM server (only used when `vllm_mode` is `"server"`)

        vllm_server_base_url (`str`, *optional*):
            Base URL for the student vLLM server. If provided, `vllm_server_host` and `vllm_server_port` are ignored.
        vllm_server_host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host of the student vLLM server.
        vllm_server_port (`int`, *optional*, defaults to `8000`):
            Port of the student vLLM server.
        vllm_server_timeout (`float`, *optional*, defaults to `240.0`):
            Timeout for connecting to the student vLLM server.
        vllm_group_port (`int`, *optional*, defaults to `51216`):
            Port for the vLLM weight-update group (NCCL communicator).

        > Parameters that control colocated vLLM execution (only used when `vllm_mode` is `"colocate"`)

        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.3`):
            GPU memory utilization for the colocated student vLLM engine.
        vllm_max_model_length (`int`, *optional*):
            Maximum model sequence length for the colocated vLLM engine.
        vllm_tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Tensor parallel size for the colocated student vLLM engine.

        > Parameters that control the training

        beta (`float`, *optional*, defaults to `1.0`):
            Interpolation coefficient for the Generalized Jensen-Shannon Divergence loss. When `0.0`, the loss is the
            forward KL divergence. When `1.0`, the loss is the reverse KL divergence. When `0.5`, it is the standard
            JSD. Unlike GRPO's `beta` (a KL-penalty coefficient against a reference model), here it selects the
            divergence itself; there is no reference-model KL penalty.
        max_tool_calling_iterations (`int`, *optional*):
            Maximum number of tool-calling turns when training an agent. If `None`, there is no limit and generation
            stops when the model generates a response turn with no tool calls or when the total response length reaches
            `max_model_length`.

        > Parameters that control the logging

        log_completions (`bool`, *optional*, defaults to `False`):
            Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is installed,
            it prints the sample. If `wandb` and/or `trackio` logging is enabled, it logs it to `wandb` and/or
            `trackio`.
        num_completions_to_print (`int`, *optional*):
            Number of completions to print with `rich`. If `None`, all completions are logged.
        log_unique_prompts (`bool`, *optional*, defaults to `False`):
            Whether to log unique prompts. If `True`, only unique prompts are logged. If `False`, all prompts are
            logged.

    > [!NOTE]
    > These parameters have default values different from [`~transformers.TrainingArguments`]:
    > - `logging_steps`: Defaults to `10` instead of `500`.
    > - `gradient_checkpointing`: Defaults to `True` instead of `False`.
    > - `bf16`: Defaults to `True` if `fp16` is not set, instead of `False`.
    > - `learning_rate`: Defaults to `1e-6` instead of `5e-5`.
    
    """
    _unsloth_accepts_adamw_flash = True
    vllm_sampling_params: Optional[Any] = field(
        default = None,
        metadata = {'help': 'vLLM SamplingParams'},
    )
    unsloth_num_chunks : Optional[int] = field(
        default = -1,
        metadata = {'help': 'Chunk size to reduce memory usage. -1 is most efficient.'},
    )
    unsloth_logit_chunk_multiplier : Optional[int] = field(
            default = None,
            metadata = {'help': 'Multiplier for chunked logit computations.'},
        )
    unsloth_grpo_mini_batch : Optional[int] = field(
        default = None,
        metadata = {'help': 'Mini batch size for GRPO hidden state accumulation. Default is None unless user defines it.'},
    )
    
    def __init__(
        self,
        output_dir = None,
        per_device_train_batch_size = 4,
        num_train_epochs = 3.0,
        max_steps = -1,
        learning_rate = 5e-05,
        lr_scheduler_type = 'linear',
        lr_scheduler_kwargs = None,
        warmup_steps = 0.1,
        optim = 'adamw_8bit',
        optim_args = None,
        weight_decay = 0.001,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-08,
        optim_target_modules = None,
        gradient_accumulation_steps = 2,
        average_tokens_across_devices = True,
        max_grad_norm = 1.0,
        label_smoothing_factor = 0.0,
        bf16 = False,
        fp16 = False,
        bf16_full_eval = False,
        fp16_full_eval = False,
        tf32 = None,
        gradient_checkpointing = True,
        gradient_checkpointing_kwargs = None,
        torch_compile = False,
        torch_compile_backend = None,
        torch_compile_mode = None,
        use_liger_kernel = False,
        liger_kernel_config = None,
        use_cache = False,
        neftune_noise_alpha = None,
        torch_empty_cache_steps = 250,
        auto_find_batch_size = False,
        logging_strategy = 'steps',
        logging_steps = 1,
        logging_first_step = False,
        log_on_each_node = True,
        logging_nan_inf_filter = False,
        include_num_input_tokens_seen = False,
        log_level = 'passive',
        log_level_replica = 'warning',
        disable_tqdm = None,
        report_to = 'none',
        run_name = None,
        project = 'huggingface',
        trackio_space_id = None,
        trackio_bucket_id = None,
        trackio_static_space_id = None,
        eval_strategy = 'no',
        eval_steps = None,
        eval_delay = 0,
        per_device_eval_batch_size = 4,
        prediction_loss_only = False,
        eval_on_start = False,
        eval_do_concat_batches = True,
        eval_use_gather_object = False,
        eval_accumulation_steps = 2,
        batch_eval_metrics = False,
        save_only_model = False,
        save_strategy = 'steps',
        save_steps = 500,
        save_on_each_node = False,
        save_total_limit = None,
        enable_jit_checkpoint = False,
        push_to_hub = False,
        hub_token = None,
        hub_private_repo = None,
        hub_model_id = None,
        hub_strategy = 'every_save',
        hub_always_push = False,
        hub_revision = None,
        load_best_model_at_end = False,
        metric_for_best_model = None,
        greater_is_better = None,
        ignore_data_skip = False,
        restore_callback_states_from_checkpoint = False,
        full_determinism = False,
        seed = 3407,
        data_seed = None,
        use_cpu = False,
        accelerator_config = None,
        parallelism_config = None,
        dataloader_drop_last = False,
        dataloader_num_workers = 0,
        dataloader_pin_memory = True,
        dataloader_persistent_workers = False,
        dataloader_prefetch_factor = None,
        dataloader_multiprocessing_context = None,
        dataloader_in_order = True,
        remove_unused_columns = False,
        label_names = None,
        train_sampling_strategy = 'random',
        length_column_name = 'length',
        ddp_find_unused_parameters = None,
        ddp_bucket_cap_mb = None,
        ddp_broadcast_buffers = None,
        ddp_static_graph = None,
        ddp_backend = None,
        ddp_timeout = 1800,
        fsdp = None,
        fsdp_config = None,
        deepspeed = None,
        debug = '',
        skip_memory_metrics = True,
        do_train = False,
        do_eval = False,
        do_predict = False,
        resume_from_checkpoint = None,
        local_rank = -1,
        model_init_kwargs = None,
        trust_remote_code = False,
        teacher_model_name_or_path = None,
        teacher_model_revision = None,
        teacher_model_init_kwargs = None,
        disable_dropout = False,
        max_completion_length = 512,
        ds3_gather_for_generation = True,
        shuffle_dataset = True,
        pad_to_multiple_of = None,
        temperature = 1.0,
        top_p = 1.0,
        top_k = 0,
        min_p = None,
        generation_kwargs = {},
        chat_template_kwargs = None,
        repetition_penalty = 1.0,
        cache_implementation = None,
        use_vllm = False,
        vllm_mode = 'colocate',
        vllm_model_impl = 'vllm',
        vllm_enable_sleep_mode = False,
        vllm_structured_outputs_regex = None,
        vllm_server_base_url = None,
        vllm_server_host = '0.0.0.0',
        vllm_server_port = 8000,
        vllm_server_timeout = 240.0,
        vllm_group_port = 51216,
        vllm_gpu_memory_utilization = 0.3,
        vllm_max_model_length = None,
        vllm_tensor_parallel_size = 1,
        beta = 1.0,
        max_tool_calling_iterations = None,
        log_completions = False,
        num_completions_to_print = None,
        log_unique_prompts = False,
        vllm_sampling_params = None,
        unsloth_num_chunks = -1,
        unsloth_logit_chunk_multiplier = None,
        unsloth_grpo_mini_batch = None,
        
        **kwargs,
    ):
        if learning_rate < 1e-7: print(f'Unsloth: Your learning rate of `{learning_rate}` is too small and less than 1e-7! Consider increasing it, otherwise gradient updates will be close to 0!')
        if learning_rate > 1: print(f'Unsloth: Your learning rate of `{learning_rate}` is way too larger > 1! Consider decreasing it to 1e-1, otherwise gradient updates will explode!')
        _unsloth_fsdp2 = os.environ.get('FSDP_VERSION', '').strip() == '2'
        _unsloth_fsdp_cfg = fsdp_config
        if fsdp and isinstance(_unsloth_fsdp_cfg, str) and os.path.isfile(_unsloth_fsdp_cfg):
            try:
                import json as _unsloth_json
                with open(_unsloth_fsdp_cfg, encoding = 'utf-8') as _unsloth_f:
                    _unsloth_fsdp_cfg = _unsloth_json.load(_unsloth_f)
            except Exception:
                _unsloth_fsdp_cfg = None
        if fsdp and _unsloth_fsdp_cfg is None:
            _unsloth_fsdp_cfg = {}
        if fsdp and isinstance(_unsloth_fsdp_cfg, dict):
            _unsloth_fsdp2 = _unsloth_fsdp2 or str(_unsloth_fsdp_cfg.get('fsdp_version', _unsloth_fsdp_cfg.get('version', 2))) == '2'
        _unsloth_flash_adamw = str(getattr(optim, 'value', optim)).strip().lower() == 'adamw_flash'
        if _unsloth_flash_adamw: optim = 'adamw_torch'
        if _unsloth_fsdp2 and str(getattr(optim, 'value', optim)) == 'adamw_8bit':
            print("Unsloth: FSDP2 shards parameters as DTensors, which bitsandbytes' 8-bit optimizers cannot step. Switching optim = 'adamw_8bit' to 'adamw_torch_fused'.")
            optim = 'adamw_torch_fused'
        if num_train_epochs is None:
            num_train_epochs = 3.0  # Default to 3 epochs if None, max_steps will override
        if output_dir is None and save_strategy == 'steps' and save_steps == 500:
            output_dir = 'unsloth_training_checkpoints'
            save_strategy = 'no'
        if os.environ.get('UNSLOTH_ENABLE_FLEX_ATTENTION', '0') == '1':
            from unsloth_zoo.flex_attention import HAS_FLEX_ATTENTION
            if HAS_FLEX_ATTENTION and pad_to_multiple_of is None:
                from unsloth_zoo.flex_attention import FLEX_ATTENTION_BLOCK_SIZE
                pad_to_multiple_of = FLEX_ATTENTION_BLOCK_SIZE
        
        if eval_steps is not None and eval_strategy != 'steps':
            print(f'Unsloth: `eval_steps = {eval_steps}` is ignored because `eval_strategy` is {getattr(eval_strategy, "value", eval_strategy)!r}. Set `eval_strategy = "steps"` to evaluate every `eval_steps` steps.')
        
        if temperature <= 0:
            raise ValueError('Unsloth: Please set a positive non-zero temperature since your results will be wrong.')
        elif temperature >= 10:
            raise ValueError('Unsloth: Please set a positive non-zero temperature less than 10, since sampling will be quite erratic.')
        
        
        # One dict so the filter sees the mirrored parameters AND `**kwargs`:
        # filtering kwargs alone would double-bind any argument TRL renamed,
        # since the new name is itself a mirrored parameter.
        _unsloth_config_arguments = dict(
            output_dir = output_dir,
            per_device_train_batch_size = per_device_train_batch_size,
            num_train_epochs = num_train_epochs,
            max_steps = max_steps,
            learning_rate = learning_rate,
            lr_scheduler_type = lr_scheduler_type,
            lr_scheduler_kwargs = lr_scheduler_kwargs,
            warmup_steps = warmup_steps,
            optim = optim,
            optim_args = optim_args,
            weight_decay = weight_decay,
            adam_beta1 = adam_beta1,
            adam_beta2 = adam_beta2,
            adam_epsilon = adam_epsilon,
            optim_target_modules = optim_target_modules,
            gradient_accumulation_steps = gradient_accumulation_steps,
            average_tokens_across_devices = average_tokens_across_devices,
            max_grad_norm = max_grad_norm,
            label_smoothing_factor = label_smoothing_factor,
            bf16 = bf16,
            fp16 = fp16,
            bf16_full_eval = bf16_full_eval,
            fp16_full_eval = fp16_full_eval,
            tf32 = tf32,
            gradient_checkpointing = gradient_checkpointing,
            gradient_checkpointing_kwargs = gradient_checkpointing_kwargs,
            torch_compile = torch_compile,
            torch_compile_backend = torch_compile_backend,
            torch_compile_mode = torch_compile_mode,
            use_liger_kernel = use_liger_kernel,
            liger_kernel_config = liger_kernel_config,
            use_cache = use_cache,
            neftune_noise_alpha = neftune_noise_alpha,
            torch_empty_cache_steps = torch_empty_cache_steps,
            auto_find_batch_size = auto_find_batch_size,
            logging_strategy = logging_strategy,
            logging_steps = logging_steps,
            logging_first_step = logging_first_step,
            log_on_each_node = log_on_each_node,
            logging_nan_inf_filter = logging_nan_inf_filter,
            include_num_input_tokens_seen = include_num_input_tokens_seen,
            log_level = log_level,
            log_level_replica = log_level_replica,
            disable_tqdm = disable_tqdm,
            report_to = report_to,
            run_name = run_name,
            project = project,
            trackio_space_id = trackio_space_id,
            trackio_bucket_id = trackio_bucket_id,
            trackio_static_space_id = trackio_static_space_id,
            eval_strategy = eval_strategy,
            eval_steps = eval_steps,
            eval_delay = eval_delay,
            per_device_eval_batch_size = per_device_eval_batch_size,
            prediction_loss_only = prediction_loss_only,
            eval_on_start = eval_on_start,
            eval_do_concat_batches = eval_do_concat_batches,
            eval_use_gather_object = eval_use_gather_object,
            eval_accumulation_steps = eval_accumulation_steps,
            batch_eval_metrics = batch_eval_metrics,
            save_only_model = save_only_model,
            save_strategy = save_strategy,
            save_steps = save_steps,
            save_on_each_node = save_on_each_node,
            save_total_limit = save_total_limit,
            enable_jit_checkpoint = enable_jit_checkpoint,
            push_to_hub = push_to_hub,
            hub_token = hub_token,
            hub_private_repo = hub_private_repo,
            hub_model_id = hub_model_id,
            hub_strategy = hub_strategy,
            hub_always_push = hub_always_push,
            hub_revision = hub_revision,
            load_best_model_at_end = load_best_model_at_end,
            metric_for_best_model = metric_for_best_model,
            greater_is_better = greater_is_better,
            ignore_data_skip = ignore_data_skip,
            restore_callback_states_from_checkpoint = restore_callback_states_from_checkpoint,
            full_determinism = full_determinism,
            seed = seed,
            data_seed = data_seed,
            use_cpu = use_cpu,
            accelerator_config = accelerator_config,
            parallelism_config = parallelism_config,
            dataloader_drop_last = dataloader_drop_last,
            dataloader_num_workers = dataloader_num_workers,
            dataloader_pin_memory = dataloader_pin_memory,
            dataloader_persistent_workers = dataloader_persistent_workers,
            dataloader_prefetch_factor = dataloader_prefetch_factor,
            dataloader_multiprocessing_context = dataloader_multiprocessing_context,
            dataloader_in_order = dataloader_in_order,
            remove_unused_columns = remove_unused_columns,
            label_names = label_names,
            train_sampling_strategy = train_sampling_strategy,
            length_column_name = length_column_name,
            ddp_find_unused_parameters = ddp_find_unused_parameters,
            ddp_bucket_cap_mb = ddp_bucket_cap_mb,
            ddp_broadcast_buffers = ddp_broadcast_buffers,
            ddp_static_graph = ddp_static_graph,
            ddp_backend = ddp_backend,
            ddp_timeout = ddp_timeout,
            fsdp = fsdp,
            fsdp_config = fsdp_config,
            deepspeed = deepspeed,
            debug = debug,
            skip_memory_metrics = skip_memory_metrics,
            do_train = do_train,
            do_eval = do_eval,
            do_predict = do_predict,
            resume_from_checkpoint = resume_from_checkpoint,
            local_rank = local_rank,
            model_init_kwargs = model_init_kwargs,
            trust_remote_code = trust_remote_code,
            teacher_model_name_or_path = teacher_model_name_or_path,
            teacher_model_revision = teacher_model_revision,
            teacher_model_init_kwargs = teacher_model_init_kwargs,
            disable_dropout = disable_dropout,
            max_completion_length = max_completion_length,
            ds3_gather_for_generation = ds3_gather_for_generation,
            shuffle_dataset = shuffle_dataset,
            pad_to_multiple_of = pad_to_multiple_of,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            min_p = min_p,
            generation_kwargs = generation_kwargs,
            chat_template_kwargs = chat_template_kwargs,
            repetition_penalty = repetition_penalty,
            cache_implementation = cache_implementation,
            use_vllm = use_vllm,
            vllm_mode = vllm_mode,
            vllm_model_impl = vllm_model_impl,
            vllm_enable_sleep_mode = vllm_enable_sleep_mode,
            vllm_structured_outputs_regex = vllm_structured_outputs_regex,
            vllm_server_base_url = vllm_server_base_url,
            vllm_server_host = vllm_server_host,
            vllm_server_port = vllm_server_port,
            vllm_server_timeout = vllm_server_timeout,
            vllm_group_port = vllm_group_port,
            vllm_gpu_memory_utilization = vllm_gpu_memory_utilization,
            vllm_max_model_length = vllm_max_model_length,
            vllm_tensor_parallel_size = vllm_tensor_parallel_size,
            beta = beta,
            max_tool_calling_iterations = max_tool_calling_iterations,
            log_completions = log_completions,
            num_completions_to_print = num_completions_to_print,
            log_unique_prompts = log_unique_prompts,**kwargs)
        super().__init__(**_unsloth_filter_config_init_kwargs(DistillationConfig, _unsloth_config_arguments, mirrored_from = __class__))
        self.vllm_sampling_params = vllm_sampling_params
        self.unsloth_num_chunks = unsloth_num_chunks
        if unsloth_grpo_mini_batch is not None:
            if self.generation_batch_size >= unsloth_grpo_mini_batch:
                self.unsloth_grpo_mini_batch = unsloth_grpo_mini_batch
            else:
                raise ValueError(
                    f"Unsloth GRPO mini batch size needs to be less than or equal to the effective generation batch size, "
                    f"which is self.per_device_train_batch_size * gradient_accumulation_steps."
                )
        self.unsloth_logit_chunk_multiplier = unsloth_logit_chunk_multiplier
        
        # Unsloth: keep the reentrant checkpoint path
        if getattr(self, 'gradient_checkpointing', False):
            _gc_kwargs = getattr(self, 'gradient_checkpointing_kwargs', None) or {}
            if _gc_kwargs.get('context_fn') is None and not _gc_kwargs.get('debug', False):
                _gc_kwargs['use_reentrant'] = True
                self.gradient_checkpointing_kwargs = _gc_kwargs
        from unsloth.optimizers.flash_adamw import is_flash_adamw_marked, mark_flash_adamw, unmark_flash_adamw
        if _unsloth_flash_adamw or is_flash_adamw_marked(self):
            mark_flash_adamw(self)
        else:
            unmark_flash_adamw(self)

pass

class _UnslothDistillationTrainer(_BaseTrainer):
    """"""

    loss_is_scaled_for_ga = True

    _tag_names = ["trl", "distillation"]
    _name = "Distillation"
    _paper = {
        "title": "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes",
        "id": "2306.13649",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @inproceedings{agarwal2024on-policy,
                title        = {{On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes}},
                author       = {Rishabh Agarwal and Nino Vieillard and Yongchao Zhou and Piotr Stanczyk and Sabela Ramos Garea and Matthieu Geist and Olivier Bachem},
                year         = 2024,
                booktitle    = {The Twelfth International Conference on Learning Representations, {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
                publisher    = {OpenReview.net},
                url          = {https://openreview.net/forum?id=3zKtaqxLhW},
            }"""),
    }

    def __init__(
        self,
        model: "str | PreTrainedModel | PeftModel",
        teacher_model: str | PreTrainedModel | None = None,
        args: DistillationConfig | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        quantization_config: "BitsAndBytesConfig | None" = None,
        peft_config: "PeftConfig | None" = None,
        tools: list[Callable] | None = None,
    ):

        if hasattr(model, 'vllm_engine') and hasattr(args, 'use_vllm'):
            _unsloth_fsdp_cfg = getattr(args, 'fsdp_config', None)
            if getattr(args, 'fsdp', None) and isinstance(_unsloth_fsdp_cfg, dict) and str(_unsloth_fsdp_cfg.get('version', _unsloth_fsdp_cfg.get('fsdp_version', 2))) == '2':
                raise NotImplementedError('Unsloth: `fast_inference = True` (vLLM) is not supported with FSDP2.\nLoad the model with `fast_inference = False` to train with FSDP2, or launch without FSDP2 to use vLLM.')
            if (getattr(args, 'use_vllm', False) == False):
                args.use_vllm = True
            if getattr(args, 'top_k', -1) is None or getattr(args, 'top_k', -1) == 0:
                args.top_k = -1
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = DistillationConfig(f"{model_name}-Distillation")

        # Student model loading
        # PEFT initializes the adapter weights randomly, so set_seed must be done before creating the model to ensure
        # reproducibility.
        set_seed(args.seed)
        # `_VALID_DICT_FIELDS` already parses any JSON-string form of these in `DistillationConfig.__post_init__`, so
        # they are dicts [or None] here; copy so the setdefaults below don't mutate the config.
        model_init_kwargs = dict(args.model_init_kwargs or {})
        teacher_model_init_kwargs = dict(args.teacher_model_init_kwargs or {})
        if isinstance(model, str):
            model_name_or_path = model
            if quantization_config is not None:
                if "quantization_config" in model_init_kwargs:
                    raise ValueError(
                        "You set `quantization_config` both as a trainer argument and in `args.model_init_kwargs`. "
                        "Please set it in only one place, preferably as a trainer argument."
                    )
                model_init_kwargs["quantization_config"] = quantization_config
            # Distributed training requires device_map=None ["auto" fails]
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
            model_revision = model_init_kwargs.get("revision")
            model = create_model_from_path(model, **model_init_kwargs)
        else:
            model_name_or_path = get_config_model_id(model.config)
            model_revision = None
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `DistillationConfig`, but your model is already "
                    "instantiated. The `model_init_kwargs` will be ignored."
                )
            if quantization_config is not None:
                logger.warning(
                    "You passed `quantization_config` to the trainer, but your model is already instantiated. The "
                    "`quantization_config` will be ignored."
                )
        # Non-quantized models do not have the `is_loaded_in_{8,4}bit` attributes, whereas quantized models do
        _is_quantized_model = getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)

        # Some models [SmolVLM/Idefics3] don't support `logits_to_keep` argument and error out if we pass it
        # Inspect the forward method before we wrap the model with PEFT
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(
                model_name_or_path,
                revision=model_revision,
                truncation_side="left",
                padding_side="left",
                trust_remote_code=args.trust_remote_code,
            )

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            self._tokenizer = processing_class.tokenizer
            self._is_vlm = True
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            self._tokenizer = processing_class
            self._is_vlm = False
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # The model must agree with the tokenizer on the pad token from construction, so mirror it onto the model
        # configs.
        model.config.get_text_config().pad_token_id = self._tokenizer.pad_token_id
        model.generation_config.pad_token_id = self._tokenizer.pad_token_id

        # Every eos id the model declares [e.g. an end-of-turn token] ends a completion, not only the tokenizer's
        eos_token_ids = model.generation_config.eos_token_id
        if eos_token_ids is None:
            eos_token_ids = []
        elif isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        if self._tokenizer.eos_token_id not in eos_token_ids:
            eos_token_ids = [self._tokenizer.eos_token_id, *eos_token_ids]
        self.eos_token_ids = eos_token_ids

        # Resolve vision placeholder token IDs once. Used by the forward pass to rebuild mm_token_type_ids
        # when tool responses inject images into the completion [see _generate forward_kwargs block].
        self._image_pad_token_id = None
        self._video_pad_token_id = None
        if self._is_vlm:
            for candidate in ("<|image_pad|>", "<|image|>"):
                tid = self._tokenizer.convert_tokens_to_ids(candidate)
                if tid != self._tokenizer.unk_token_id:
                    self._image_pad_token_id = tid
                    break
            tid = self._tokenizer.convert_tokens_to_ids("<|video_pad|>")
            if tid != self._tokenizer.unk_token_id:
                self._video_pad_token_id = tid

        # Tools
        if tools:
            if not Version(transformers.__version__) >= Version("5.0.0"):
                raise ImportError(
                    "Using tools with DistillationTrainer requires transformers version 5.0.0 or higher. Please "
                    "upgrade transformers with `pip install --upgrade transformers` to use this feature."
                )
            # jmespath is only needed by the legacy `response_schema` parser, which is all transformers < 5.13 ships.
            # The new-style `response_template` parser doesn't use it, so don't require it on newer versions.
            if not _SUPPORTS_RESPONSE_TEMPLATE and not is_jmespath_available():
                raise ImportError(
                    "Using tools with DistillationTrainer on transformers below 5.13.0 requires the jmespath library "
                    "for response parsing. Please install it with `pip install jmespath`, or upgrade transformers to "
                    "5.13.0 or higher, which doesn't need it."
                )
            if not supports_tool_calling(processing_class):
                raise ValueError(
                    "The provided chat template does not support tool calling. The template must be able to render a "
                    "full tool-calling conversation (user -> assistant with tool_calls -> tool)."
                )
            # Async tools are deferred to a future PR; reject them now rather than fail cryptically in the tool loop.
            if any(inspect.iscoroutinefunction(tool) for tool in tools):
                raise ValueError("Async tools are not yet supported by `DistillationTrainer`. Pass synchronous tools.")
        self.tools = tools or []
        # Tools are constant across examples, so the tool dict is built once here.
        self._tool_dict = {tool.__name__: tool for tool in self.tools}

        # `add_response_schema` sets the response template [transformers >= 5.13] or legacy schema for known chat
        # templates, so tool calls can be parsed. Skip if one is already set; warn if it's a migratable legacy schema.
        if self.tools:
            has_template = getattr(self._tokenizer, "response_template", None) is not None
            has_schema = getattr(self._tokenizer, "response_schema", None) is not None
            if not has_template and not has_schema:
                processing_class = add_response_schema(processing_class)
            elif has_schema and not has_template and _SUPPORTS_RESPONSE_TEMPLATE:
                warnings.warn(
                    "The tokenizer has a legacy `response_schema` set but no `response_template`. The installed "
                    "transformers supports the new-style `response_template`; consider migrating, as `response_schema` "
                    "support will eventually be removed. See the Transformers response-parsing docs.",
                    FutureWarning,
                )
        # In multi-turn training, the chat template *must* be prefix-preserving. If the tokenizer's original template
        # isn't, we replace it at initialization with a training-safe, prefix-preserving template.
        if self.tools and not is_chat_template_prefix_preserving(processing_class):
            self.chat_template = get_training_chat_template(processing_class)
        else:
            self.chat_template = None

        # PEFT
        if False:
            if not is_peft_available():
                raise ImportError(
                    "You passed `peft_config` but the `peft` library is not installed. "
                    "Install it with `pip install trl[peft]`."
                )
            if not isinstance(peft_config, PeftConfig):
                raise TypeError(
                    f"`peft_config` must be a `peft.PeftConfig` instance (e.g. `peft.LoraConfig`), "
                    f"got {type(peft_config).__name__}."
                )
            if is_peft_model(model):
                raise ValueError(
                    "You passed a `PeftModel` instance together with a `peft_config` to the trainer. Please first merge "
                    "and unload the existing adapter, save the resulting base model, and then pass that base model along "
                    "with the new `peft_config` to the trainer."
                )
            # Create PEFT model
            # ZeRO-3 + PEFT for non-quantized models:
            # - PEFT's default autocast_adapter_dtype=True upcasts LoRA adapter params to fp32 even when the base model is bf16.
            # - ZeRO-3's _allgather_params_coalesced allocates output buffers using the dtype of the first persistent parameter,
            #   so mixed-dtype persistent_parameters [bf16 base + fp32 LoRA] cause a TypeError on the first optimizer step.
            # - Passing autocast_adapter_dtype=False keeps adapter params in the base model dtype [bf16], fixing the mismatch.
            # - This is safe: the fp32 upcast is a QLoRA-specific concern [low-bit quantized base models], not needed for
            #   non-quantized bf16 training.
            # - See:
            #   - TRL issue: https://github.com/huggingface/trl/issues/6089
            #   - Upstream issue: https://github.com/deepspeedai/DeepSpeed/issues/8072
            get_peft_model_kwargs = {}
            if args.deepspeed_plugin is not None and args.deepspeed_plugin.zero_stage == 3 and not _is_quantized_model:
                get_peft_model_kwargs["autocast_adapter_dtype"] = False
            model = get_peft_model(model, peft_config, **get_peft_model_kwargs)

        # PEFT + DeepSpeed ZeRO-3 requires reentrant checkpointing. For more details, see
        # https://github.com/huggingface/trl/issues/2514#issuecomment-2692152703.
        # Can be removed once https://github.com/deepspeedai/DeepSpeed/pull/8130 is merged and released.
        if (
            is_peft_model(model)
            and args.deepspeed_plugin is not None
            and args.deepspeed_plugin.zero_stage == 3
            and args.gradient_checkpointing
        ):
            args.gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
            use_reentrant = args.gradient_checkpointing_kwargs.get("use_reentrant")
            if use_reentrant is False:
                logger.warning(
                    "You are using PEFT with DeepSpeed ZeRO-3 and gradient checkpointing with `use_reentrant=False`. "
                    "`use_reentrant` is forced to `True` in this configuration to ensure correct training. To remove "
                    "this warning, unset `use_reentrant` in `gradient_checkpointing_kwargs` or set it to `True`."
                )
            args.gradient_checkpointing_kwargs["use_reentrant"] = True

        # When using gradient checkpointing with PEFT, we need to enable input gradients. transformers.Trainer normally
        # handles this, but a bug currently prevents it; see https://github.com/huggingface/transformers/issues/42489
        if is_peft_model(model) and args.gradient_checkpointing:
            model.enable_input_require_grads()

        # When using QLoRA, the PEFT adapter weights are converted to bf16 to follow the recommendations from the
        # original paper [see https://huggingface.co/papers/2305.14314, paragraph 3]. Normally, this can be done by
        # passing `autocast_adapter_dtype=False` to `get_peft_model`, but this option is not yet supported for
        # quantized models. See: https://github.com/huggingface/peft/issues/2889
        # The DoRA magnitude vector is excluded: unlike LoRA A/B, its optimizer updates can be smaller than bf16 can
        # represent, silently freezing it, see #7268.
        if False:
            for name, param in model.named_parameters():
                if param.requires_grad and "lora_magnitude_vector" not in name:
                    param.data = param.data.to(torch.bfloat16)

        # The chunked JSD loss reads `lm_head.weight` directly and runs the backbone via
        # `_get_last_hidden_state`, bypassing `PeftModel.forward[]` — so PEFT setups that live outside the backbone
        # weights are silently ignored. Fail loudly rather than train on a silently-wrong objective.
        if is_peft_model(model):
            # When the LM head is targeted by a PEFT adapter [`"lm_head"` in `target_modules`], `lm_head.weight` is the
            # frozen base weight and the trainable adapter lives in separate submodules the loss never sees, so the head
            # adapter would receive no gradient.
            output_embeddings = model.get_output_embeddings()
            if isinstance(output_embeddings, BaseTunerLayer):
                raise ValueError(
                    "Applying a PEFT adapter to `lm_head` is not supported. The distillation loss reads "
                    "`lm_head.weight` directly, so the adapter on the head is ignored and never trained. Remove "
                    "`'lm_head'` from your `target_modules`."
                )
            # Prompt-learning methods [PromptTuning, PrefixTuning, P-Tuning] inject virtual tokens via
            # `PeftModel.forward[]`, which the loss bypasses by calling the backbone directly, so virtual tokens are
            # never prepended and the loss is computed on the wrong sequence.
            if any(isinstance(cfg, PromptLearningConfig) for cfg in model.peft_config.values()):
                raise ValueError(
                    "Prompt-learning PEFT methods (PromptTuning, PrefixTuning, P-Tuning) are not supported. The "
                    "distillation loss bypasses `PeftModel.forward()` by calling the backbone directly, so virtual "
                    "tokens are never prepended and the loss is computed on the wrong sequence. Use a weight-based "
                    "adapter such as LoRA instead."
                )

        # The chunked JSD loss calls the student backbone directly, bypassing the DDP/FSDP wrapper's forward; route it
        # through `_forward_redirection` so `prepare_for_backward[]` still fires.
        self._forward_redirection = _ForwardRedirection()

        # Teacher model setup
        # `teacher_model` may be None: subclasses [e.g. ServerDistillationTrainer] supply the teacher another way.
        if teacher_model is not None:
            if isinstance(teacher_model, str):
                dtype = teacher_model_init_kwargs.get("dtype")
                teacher_model_init_kwargs["dtype"] = dtype if dtype in ["auto", None] else getattr(torch, dtype)
                if args.teacher_model_revision is not None:
                    teacher_model_init_kwargs.setdefault("revision", args.teacher_model_revision)
                # Distributed training requires device_map=None ["auto" fails]
                if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                    teacher_model_init_kwargs["device_map"] = None
                teacher_model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
                teacher_model = create_model_from_path(teacher_model, **teacher_model_init_kwargs)
            elif args.teacher_model_init_kwargs is not None:
                raise ValueError(
                    "You passed teacher_model_init_kwargs to the config, but your teacher_model is already "
                    "instantiated."
                )

        # Iterable datasets can't be indexed, so the RepeatSampler can't be attached to them. Instead, the sampler's
        # ordering is reproduced by streaming [see `get_train_dataloader`/`get_eval_dataloader` and
        # `repeat_iterable_dataset`]. This requires `dispatch_batches=False`: with the default dispatch path, batches
        # are collated on the main process and Accelerate tries to concatenate the string `prompt` column, which fails;
        # `dispatch_batches=False` also lets each process shard the stream into contiguous slices, as the sampler does.
        # See https://github.com/huggingface/trl/issues/3213
        uses_iterable_dataset = (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        )
        if uses_iterable_dataset:
            if args.accelerator_config.dispatch_batches:
                raise ValueError(
                    "Iterable datasets require `dispatch_batches=False`, but it is set to `True` in "
                    "`accelerator_config`. Please set it to `False`."
                )
            args.accelerator_config.dispatch_batches = False
        # An iterable train set bakes the generation-batch repeats into the stream, so it must be read by a single
        # worker: multiple workers would shard and interleave it, breaking the generation-batch ordering that
        # `_prepare_inputs` relies on. Map-style train keeps its workers.
        if isinstance(train_dataset, IterableDataset) and args.dataloader_num_workers != 0:
            logger.warning(
                f"Iterable datasets require `dataloader_num_workers=0` to preserve prompt grouping; overriding the "
                f"provided value ({args.dataloader_num_workers})."
            )
            args.dataloader_num_workers = 0

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in distillation
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # With several GPUs visible and no distributed launcher, `Trainer` wraps the model in `nn.DataParallel`, whose
        # replicas would all run the forward bound to the original model. Checked after `Trainer.__init__`, which sets
        # `n_gpu` to 1 for a model split across devices with `device_map`.
        if args.n_gpu > 1:
            raise ValueError(
                f"{type(self).__name__} does not support `nn.DataParallel`, which `Trainer` uses when several GPUs are "
                "visible to a single process. Launch the script with `accelerate launch` or `torchrun`, or make a "
                "single GPU visible with `CUDA_VISIBLE_DEVICES`."
            )

        self._dist = DistributedBackend(self.accelerator)

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        # Prepare teacher model [after super[].__init__ so accelerator is ready]
        if teacher_model is not None:
            # The divergence compares the full next-token distribution of the student against the teacher's, so both
            # must be defined over the same vocabulary.
            student_vocab_size = self.model.config.get_text_config().vocab_size
            teacher_vocab_size = teacher_model.config.get_text_config().vocab_size
            if student_vocab_size != teacher_vocab_size:
                raise ValueError(
                    f"The student model has vocab_size {student_vocab_size} but the teacher model has vocab_size "
                    f"{teacher_vocab_size}. Distillation compares the teacher's full next-token distribution, which "
                    f"requires a shared vocabulary. Use a teacher with the same vocab_size, or GOLD for "
                    f"cross-tokenizer distillation."
                )
            if self.is_deepspeed_enabled:
                self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
            else:
                self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
        else:
            self.teacher_model = None

        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        # Store config values
        self.beta = args.beta
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.max_completion_length = args.max_completion_length
        self.max_tool_calling_iterations = (
            args.max_tool_calling_iterations if args.max_tool_calling_iterations is not None else sys.maxsize
        )
        self.chat_template_kwargs = args.chat_template_kwargs or {}
        self.pad_to_multiple_of = args.pad_to_multiple_of
        self.shuffle_dataset = args.shuffle_dataset

        # Tracks the number of iterations [forward + backward passes], including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # Ensure each process receives a unique seed so different processes generate different completions when
        # generating with transformers. We could skip it if we use vLLM, but it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        # Generation config
        generation_kwargs = {
            "max_new_tokens": self.max_completion_length,
            "do_sample": True,
            "pad_token_id": self._tokenizer.pad_token_id,
            "bos_token_id": self._tokenizer.bos_token_id,
            "eos_token_id": self.eos_token_ids,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "cache_implementation": args.cache_implementation,
        }
        if args.generation_kwargs is not None:
            generation_kwargs.update(args.generation_kwargs)
        self.generation_config = GenerationConfig(**generation_kwargs, disable_compile=True)
        # Keep training-specific generation kwargs to overwrite model's original generation config
        self.generation_kwargs = generation_kwargs

        # Metrics & Logging
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self._current_train_step_time = 0.0
        self.log_completions = args.log_completions
        self.log_unique_prompts = args.log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # Keep logs sized to the generation batch to record only outputs from the latest model update.
        generation_batch_size = (
            args.per_device_train_batch_size * self.accelerator.num_processes * args.gradient_accumulation_steps
        )
        self._logs = {
            "prompt": deque(maxlen=generation_batch_size),
            "completion": deque(maxlen=generation_batch_size),
        }

        if self.accelerator.is_main_process and self.log_completions:
            os.makedirs(os.path.join(self.args.output_dir, "completions"), exist_ok=True)

        # vLLM for student generation
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install trl[vllm]` to use it."
                )
            self.vllm_generation = VLLMGeneration(
                model=self.model,
                accelerator=self.accelerator,
                processing_class=self.processing_class,
                # vLLM configuration
                mode=args.vllm_mode,
                structured_outputs_regex=args.vllm_structured_outputs_regex,
                # Server mode configuration
                server_base_url=args.vllm_server_base_url,
                server_host=args.vllm_server_host,
                server_port=args.vllm_server_port,
                group_port=args.vllm_group_port,
                server_timeout=args.vllm_server_timeout,
                # Colocate mode configuration
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_length=args.vllm_max_model_length,
                max_num_seqs=args.per_device_train_batch_size
                * args.vllm_tensor_parallel_size
                * args.gradient_accumulation_steps,
                enable_sleep_mode=args.vllm_enable_sleep_mode,
                model_impl=args.vllm_model_impl,
                trust_remote_code=args.trust_remote_code,
                # Generation configuration
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                max_completion_length=self.max_completion_length,
                logprobs=None,  # distillation trains on the teacher distribution, not sampled logprobs
                generation_kwargs=args.generation_kwargs,
            )
            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        # and "attention_mask"). In DistillationTrainer, we preprocess data, so using the model's signature columns
        # doesn't work. Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "image", "images"]

    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × gradient_accumulation_steps`). This allows us to generate
    # completions once every gradient_accumulation_steps step—rather than once per accumulation step—which is
    # significantly more efficient. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    # Maintenance note: this method is a copy-paste of the original `Trainer.get_train_dataloader` with two changes: the
    # batch size is multiplied by `gradient_accumulation_steps`, and iterable datasets are wrapped (see
    # `repeat_iterable_dataset`).
    def get_train_dataloader(self):
        dataset = self.train_dataset
        if isinstance(dataset, IterableDataset):
            # Iterable datasets can't be indexed, so RepeatSampler can't be attached. Reproduce its ordering by
            # transforming the stream instead (see `repeat_iterable_dataset`). The full permutation done by
            # RepeatSampler becomes a buffered shuffle here.
            if self.shuffle_dataset:
                dataset = dataset.shuffle(seed=self.args.seed)
            # Effective training batch = the generation batch (the deferred `generation_batch_size` slot-in).
            generation_batch_size = (
                self.args.per_device_train_batch_size
                * self.accelerator.num_processes
                * self.args.gradient_accumulation_steps
            )
            dataset = repeat_iterable_dataset(
                dataset,
                mini_repeat_count=1,
                batch_size=generation_batch_size,
                repeat_count=self.args.gradient_accumulation_steps,
            )
        return self._get_dataloader(
            dataset=dataset,
            description="Training",
            batch_size=self._train_batch_size * self.args.gradient_accumulation_steps,  # < this is the change
            sampler_fn=self._get_train_sampler,
            is_training=True,
        )

    def _get_train_sampler(self, dataset: Dataset | None = None) -> Sampler:
        # Repeat each generation batch `gradient_accumulation_steps` times so the completions generated once per
        # generation batch (see `_prepare_inputs`) are reused across the accumulation window. Distillation is n=1,
        # so there is no per-prompt repeat (`mini_repeat_count=1`).
        if dataset is None:
            dataset = self.train_dataset
        # The generation batch is the full effective training batch. GRPO names it `generation_batch_size` (paired with
        # the deferred `steps_per_generation`); both are deferred here, so derive it inline — slot-in point.
        generation_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=1,
            batch_size=generation_batch_size,
            repeat_count=self.args.gradient_accumulation_steps,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=1,
            seed=self.args.seed,
        )

    # This method overrides `Trainer.get_eval_dataloader` to wrap iterable eval datasets, reproducing the
    # RepeatSampler ordering that can't be attached to them (see `get_train_dataloader`). Map-style datasets keep the
    # default path via `_get_eval_sampler`, which shuffles with `seed`, so the iterable wrap shuffles too (buffered)
    # to walk prompts in a matching order.
    # Maintenance note: this method is a copy-paste of the original `Trainer.get_eval_dataloader`, with the iterable
    # wrapping as the only addition.
    def get_eval_dataloader(self, eval_dataset: str | Dataset | IterableDataset | None = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self._eval_dataloaders[dataloader_key]

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )

        if isinstance(eval_dataset, IterableDataset):
            # Apply the `__init__` iterable config here too
            if self.args.accelerator_config.dispatch_batches:
                raise ValueError(
                    "Iterable datasets require `dispatch_batches=False`, but it is set to `True` in "
                    "`accelerator_config`. Please set it to `False`."
                )
            self.accelerator.dataloader_config.dispatch_batches = False
            eval_dataset = eval_dataset.shuffle(seed=self.args.seed)
            eval_dataset = repeat_iterable_dataset(eval_dataset, mini_repeat_count=1)
            # Force a single worker for this loader only, without persisting the change
            num_workers = self.args.dataloader_num_workers
            self.args.dataloader_num_workers = 0

        try:
            return self._get_dataloader(
                dataset=eval_dataset,
                description="Evaluation",
                batch_size=self.args.eval_batch_size,
                sampler_fn=self._get_eval_sampler,
                dataloader_key=dataloader_key,
            )
        finally:
            if isinstance(eval_dataset, IterableDataset):
                self.args.dataloader_num_workers = num_workers

    def _tokenize_prompts(self, prompts: list):
        """Tokenize prompts and extract images/multimodal fields for generation."""
        if is_conversational({"prompt": prompts[0]}):
            # Normalize string content to content blocks for VLM processors that don't handle plain strings.
            if self._is_vlm:
                prompts = [prepare_multimodal_messages(prompt) for prompt in prompts]

            # Extract images from messages for VLM support
            images = []
            has_images = False
            for prompt in prompts:
                prompt_images = []
                for message in prompt:
                    if isinstance(message["content"], list):
                        for part in message["content"]:
                            if part["type"] == "image":
                                prompt_images.append(part["image"])
                                has_images = True
                images.append(prompt_images if prompt_images else None)
            images = images if has_images else None

            # Workaround for a bug in transformers 5.3.0 where some processors (e.g. Qwen2.5-VL) crash on
            # batched unpadded input (transformers#44514).
            # Fixed in transformers 5.4.0 (transformers#44563).
            needs_padding_workaround = Version("5.3.0") <= Version(transformers.__version__) < Version("5.4.0")
            tokenized = self.processing_class.apply_chat_template(
                conversation=prompts,
                tools=self.tools or None,  # `or None`: Llama bug: it renders tool boilerplate for tools=[]
                chat_template=self.chat_template,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **({"padding": True} if needs_padding_workaround else {}),
                **self.chat_template_kwargs,
            )
            if needs_padding_workaround:
                # Unpad input_ids: remove padding tokens using attention_mask to get per-sequence lists
                prompt_ids = [
                    [tok for tok, m in zip(ids, mask, strict=True) if m]
                    for ids, mask in zip(tokenized["input_ids"], tokenized["attention_mask"], strict=True)
                ]
            else:
                prompt_ids = tokenized["input_ids"]
            # For VLMs, the processor returns extra multimodal fields (pixel_values, image_grid_thw, etc.)
            multimodal_fields = {k: v for k, v in tokenized.items() if k not in ("input_ids", "attention_mask")}
        else:
            prompt_ids = self.processing_class(text=prompts)["input_ids"]
            images = None
            multimodal_fields = {}
        return prompt_ids, images, multimodal_fields

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields, has_tool_images=False):
        device = self.accelerator.device

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # Sync weights if training step changed
            if self.state.global_step != self._last_loaded_step:
                with profiling_context(self, "sync_weights"):
                    self.vllm_generation.sync_weights()
                self._last_loaded_step = self.state.global_step

            # Generate using vLLM with raw token IDs. Distillation is n=1 and uses the teacher distribution rather than
            # sampled logprobs, so we request one completion per prompt and discard vLLM's logprobs.
            _, completion_ids, _, _ = self.vllm_generation.generate(
                prompts=prompt_ids,
                images=images,
                num_generations=1,
                profiler=profiling_context(self, "vLLM.generate"),
            )

        else:
            # Regular generation path: left-pad token IDs into tensors
            prompt_tensors = [torch.tensor(ids) for ids in prompt_ids]
            padded_ids = pad(prompt_tensors, padding_value=self._tokenizer.pad_token_id, padding_side="left")
            attention_mask = pad([torch.ones_like(t) for t in prompt_tensors], padding_value=0, padding_side="left")
            generate_inputs = {"input_ids": padded_ids, "attention_mask": attention_mask}
            # For VLMs, include multimodal fields as tensors (pixel_values, image_grid_thw, etc.)
            for k, v in multimodal_fields.items():
                if isinstance(v, torch.Tensor):
                    generate_inputs[k] = v
                elif isinstance(v, list) and v and isinstance(v[0], list):
                    # Per-token field (e.g., token_type_ids): left-pad like input_ids
                    generate_inputs[k] = pad([torch.tensor(x) for x in v], padding_value=0, padding_side="left")
                else:
                    generate_inputs[k] = torch.tensor(np.array(v))

            # For VLM tool images: build token type IDs from the padded input IDs.
            if self._is_vlm and self.tools and has_tool_images:
                mm_ids = torch.zeros_like(padded_ids)
                if self._image_pad_token_id is not None:
                    mm_ids[padded_ids == self._image_pad_token_id] = 1
                if self._video_pad_token_id is not None:
                    mm_ids[padded_ids == self._video_pad_token_id] = 2

                # Use the same key the model expects: token_type_ids for models like Gemma,
                # mm_token_type_ids for models like Qwen.
                if "image_grid_thw" in generate_inputs:
                    generate_inputs["mm_token_type_ids"] = mm_ids
                else:
                    generate_inputs["token_type_ids"] = mm_ids

            generate_inputs = super()._prepare_inputs(generate_inputs)

            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                    generation_kwargs=self.generation_kwargs,  # Override model.generation_config with generation_kwargs to fix transformers#42762
                ) as unwrapped_model,
                torch.no_grad(),
                self._dist.summon_full_params(self.model_wrapped, recurse=False),
            ):
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs, generation_config=self.generation_config
                )
            # Compute prompt length and extract completion ids
            prompt_length = generate_inputs["input_ids"].size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Mask everything after the first EOS token
            is_eos = torch.isin(completion_ids, torch.tensor(self.eos_token_ids, device=device))
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            completion_ids = [
                c[m].tolist() for c, m in zip(completion_ids.cpu(), completion_mask.bool().cpu(), strict=True)
            ]

        return completion_ids

    def _get_tool_suffix_ids(self, tool_messages):
        """Get token IDs for tool result formatting by using a minimal dummy conversation."""
        # Use the real tool name instead of a dummy: some templates (e.g. GPT-OSS) derive the tool response
        # header from the assistant's tool call name.
        dummy_tool_calls = [{"type": "function", "function": {"name": tool_messages[0]["name"], "arguments": {}}}]
        dummy_messages = [
            {"role": "user", "content": "dummy"},
            {
                "role": "assistant",
                # "content" is required here because VLM processors crash on tokenize=True without it
                # (KeyError in processing_utils.py). See huggingface/transformers#45290.
                "content": "",
                "tool_calls": dummy_tool_calls,
            },
        ]
        if self._is_vlm:
            dummy_messages = prepare_multimodal_messages(dummy_messages)
            tool_messages = prepare_multimodal_messages(tool_messages)

        prefix_ids = self.processing_class.apply_chat_template(
            dummy_messages,
            add_generation_prompt=False,
            tokenize=True,
            chat_template=self.chat_template,
            return_dict=False,
            **self.chat_template_kwargs,
        )
        full_ids = self.processing_class.apply_chat_template(
            dummy_messages + tool_messages,
            add_generation_prompt=True,
            tokenize=True,
            chat_template=self.chat_template,
            return_dict=False,
            **self.chat_template_kwargs,
        )
        # VLM processors return batched output (list of lists), unbatch for single conversation
        if self._is_vlm:
            prefix_ids = prefix_ids[0]
            full_ids = full_ids[0]

        # Some chat templates (notably Qwen3/Qwen3.5) render "...<|im_end|>\n" after an assistant/tool block.
        # When we compute `suffix_ids` by slicing `full_ids`, we must align the slicing boundary to
        # EOS (not EOS + newline). Templates that don't use EOS as end-of-turn (e.g. Gemma uses
        # <turn|>) skip this trimming.
        eos_positions = [i for i, tok_id in enumerate(prefix_ids) if tok_id == self._tokenizer.eos_token_id]
        if eos_positions:
            prefix_ids = prefix_ids[: eos_positions[-1] + 1]

        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError("Unexpected tokenization: the EOS-trimmed prefix IDs are not a prefix of the full IDs.")
        return full_ids[len(prefix_ids) :]

    def _tool_call_loop(self, prompts, prompt_ids, completion_ids, completions, images, multimodal_fields):
        # Tool execution loop: execute tools, then regenerate completions with tool results appended to the prompt
        tool_calls = [completion[0].get("tool_calls") for completion in completions]
        idxs_with_tool = [idx for idx, tool_call in enumerate(tool_calls) if tool_call]
        tool_calls = [tool_calls[idx] for idx in idxs_with_tool]
        tool_mask = [[1] * len(ids) for ids in completion_ids]  # 0 for tool result tokens, 1 elsewhere
        # Collect images from multimodal tool responses for the forward pass
        tool_images = [[] for _ in completion_ids]
        tool_call_count = 0
        tool_failure_count = 0
        iteration_num = 0

        while idxs_with_tool and iteration_num < self.max_tool_calling_iterations:
            prompt_completion_tools = [prompts[i] for i in idxs_with_tool]  # select only prompts that need tool calls
            # Snapshot state so we can rollback tool results that would exceed max_completion_length
            completions_len_before = [len(completions[i]) for i in idxs_with_tool]
            tool_images_len_before = [len(tool_images[i]) for i in idxs_with_tool]
            prompts_len_before = [len(prompts[i]) for i in idxs_with_tool]

            # Call the tools, and build the new prompt for generation
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                tool_call_list = tool_calls[idx]
                prompt_completion_tool = prompt_completion_tools[idx]
                # Append the last assistant message (which triggered tool_calls) to the prompt
                prompt_completion_tool.append(completions[idx_with_tool][-1])
                tool_call_results = []
                for tool_call in tool_call_list:
                    tool_call_count += 1
                    if tool_call["type"] == "function":
                        function = tool_call["function"]
                        name = function["name"]
                        try:
                            if name in self._tool_dict:
                                tool_call_results.append((name, self._tool_dict[name](**function["arguments"])))
                            else:
                                raise ValueError(f"Tool {name} not found.")
                        except Exception as e:
                            tool_failure_count += 1
                            result = {"error": str(e)}
                            tool_call_results.append((name, result))
                    else:
                        tool_failure_count += 1
                        name = tool_call.get("name", "unknown")
                        tool_call_results.append((name, {"error": f"Unsupported tool call type: {tool_call['type']}"}))

                for name, result in tool_call_results:
                    # Support multimodal tool responses: if the tool returns a list of content blocks
                    # (e.g., [{"type": "image", "image": ...}, {"type": "text", "text": "..."}]),
                    # pass them through directly so _tokenize_prompts can extract images for VLMs.
                    content = result if isinstance(result, list) else str(result)
                    tool_message = {"role": "tool", "name": name, "content": content}
                    # Collect images from multimodal tool responses
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "image":
                                tool_images[idx_with_tool].append(part["image"])
                    prompt_completion_tool.append(tool_message)
                    completions[idx_with_tool].append(tool_message)

            # Build token IDs by concatenation: prompt + completion + tool_suffix.
            prompt_completion_tool_ids = []
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                # Extract trailing tool messages from completions
                tool_messages = []
                for message in reversed(completions[idx_with_tool]):
                    if message["role"] == "tool":
                        tool_messages.insert(0, message)
                    else:
                        break
                suffix_ids = self._get_tool_suffix_ids(tool_messages)
                prompt_completion_tool_ids.append(
                    prompt_ids[idx_with_tool] + completion_ids[idx_with_tool] + suffix_ids
                )

            # Drop tool results whose addition would push the sequence past max_completion_length (the completion
            # budget) or past the backend context ceiling (vLLM and transformers will error out on inputs longer than
            # the model's max length). The sample exits the loop with its completion as-is, and the tool
            # messages/images appended this iteration are rolled back so completions and tool_images stay consistent
            # with completion_ids.
            if self.use_vllm and self.vllm_mode == "colocate":
                max_model_len = self.vllm_generation.llm.llm_engine.model_config.max_model_len
            else:
                config = self.model.config.text_config if self._is_vlm else self.model.config
                max_model_len = config.max_position_embeddings
            overlong = [
                len(pct) - len(prompt_ids[i]) > self.max_completion_length or len(pct) >= max_model_len
                for i, pct in zip(idxs_with_tool, prompt_completion_tool_ids, strict=True)
            ]
            for idx in range(len(idxs_with_tool)):
                if overlong[idx]:
                    idx_with_tool = idxs_with_tool[idx]
                    del completions[idx_with_tool][completions_len_before[idx] :]
                    del tool_images[idx_with_tool][tool_images_len_before[idx] :]
                    del prompts[idx_with_tool][prompts_len_before[idx] :]
            # Keep only non-overlong items for further processing
            idxs_with_tool = [idx for idx, o in zip(idxs_with_tool, overlong, strict=True) if not o]
            prompt_completion_tool_ids = [
                pct for pct, o in zip(prompt_completion_tool_ids, overlong, strict=True) if not o
            ]
            if not idxs_with_tool:
                break  # all overlong, exit tool loop

            # Filter images and multimodal fields to match the current subset (index into full batch).
            # Merge tool response images so the model can see visual feedback during generation.
            merged_images = images
            if any(imgs for imgs in tool_images):
                if merged_images is None:
                    merged_images = [imgs if imgs else None for imgs in tool_images]
                else:
                    merged_images = [
                        (existing or []) + new for existing, new in zip(merged_images, tool_images, strict=True)
                    ]
            loop_images = [merged_images[i] for i in idxs_with_tool] if merged_images else None
            if self._is_vlm and self.tools and any(imgs for imgs in tool_images):
                flat_loop_images = [img for img_list in loop_images if img_list for img in img_list]
                if flat_loop_images:
                    image_inputs = self.processing_class.image_processor(images=flat_loop_images, return_tensors="pt")
                    image_inputs = super()._prepare_inputs(image_inputs)
                    loop_multimodal_fields = dict(image_inputs)
                else:
                    loop_multimodal_fields = {}
            elif multimodal_fields:
                if "num_images" not in multimodal_fields and images is not None:
                    multimodal_fields = {
                        **multimodal_fields,
                        "num_images": [len(img_list) if img_list else 0 for img_list in images],
                    }
                split_fields = split_pixel_values_by_grid(multimodal_fields)
                loop_multimodal_fields = {}
                for k, v in split_fields.items():
                    selected = [v[i] for i in idxs_with_tool]
                    # Per-token type-id fields may arrive as tensors (e.g. via the padding workaround);
                    # convert them to lists so they follow the same left-padding path as list-valued
                    # per-token fields below, keeping them aligned with the left-padded input_ids.
                    if k in ("token_type_ids", "mm_token_type_ids") and isinstance(selected[0], torch.Tensor):
                        selected = [s.tolist() for s in selected]
                    # Per-token fields (e.g. token_type_ids) need zero-padding to match extended prompt length
                    if isinstance(selected[0], list):
                        selected = [
                            s + [0] * (len(pct) - len(s))
                            for s, pct in zip(selected, prompt_completion_tool_ids, strict=True)
                        ]
                    elif isinstance(v, torch.Tensor):
                        selected = torch.stack(selected)
                    loop_multimodal_fields[k] = selected
                loop_multimodal_fields = unsplit_pixel_values_by_grid(loop_multimodal_fields)
                loop_multimodal_fields.pop("num_images", None)
                if "pixel_values" in loop_multimodal_fields and loop_multimodal_fields["pixel_values"].numel() == 0:
                    for key in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
                        loop_multimodal_fields.pop(key, None)
            else:
                loop_multimodal_fields = {}

            # Generate new completions after tool execution (using concatenated IDs, no re-tokenization)
            post_tool_ids = self._generate_single_turn(
                prompt_completion_tool_ids,
                loop_images,
                loop_multimodal_fields,
                has_tool_images=any(imgs for imgs in tool_images),
            )

            # Truncate so that pct[len(prompt_ids[idx]) :] + post_tool does not exceed max_completion_length.
            # The pre-regen check guarantees len(completion_tool_ids) <= max_completion_length, so any
            # excess can only come from post_tool_ids. post_tool_ids is model-generated text and never
            # contains image tokens, so a plain slice is safe.
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                completion_tool_length = len(prompt_completion_tool_ids[idx]) - len(prompt_ids[idx_with_tool])
                excess_length = completion_tool_length + len(post_tool_ids[idx]) - self.max_completion_length
                if excess_length > 0:
                    new_len = len(post_tool_ids[idx]) - excess_length
                    post_tool_ids[idx] = post_tool_ids[idx][:new_len]

            # Update tool_mask: the tool result should be 0 and the post-tool 1
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_completion_tool_length = len(prompt_completion_tool_ids[idx])
                prompt_length = len(prompt_ids[idx_with_tool])
                completion_length = len(completion_ids[idx_with_tool])
                post_tool_length = len(post_tool_ids[idx])
                tool_length = prompt_completion_tool_length - prompt_length - completion_length
                tool_mask[idx_with_tool] += [0] * tool_length + [1] * post_tool_length

            # Update completion_ids with the new completions (after tool execution)
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_length = len(prompt_ids[idx_with_tool])
                pct = prompt_completion_tool_ids[idx]  # = prompt-completion-tool
                completion_ids[idx_with_tool] = pct[prompt_length:] + post_tool_ids[idx]

            # Decode post-tool completions
            post_tool_completions = [
                parse_response(self._tokenizer, ids, prefix=prompt_completion_tool_ids[idx]) if ids else {}
                for idx, ids in enumerate(post_tool_ids)
            ]

            # Add post-tool completions to the existing completions
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                if post_tool_completions[idx]:  # {} if post-tool completions completely truncated
                    completions[idx_with_tool].append(post_tool_completions[idx])

            # Check for further tool calls
            tool_calls = [completion.get("tool_calls") for completion in post_tool_completions]
            idxs_with_tool = [idx for idx, tool_call in zip(idxs_with_tool, tool_calls, strict=True) if tool_call]
            tool_calls = [tool_call for tool_call in tool_calls if tool_call]
            iteration_num += 1

        return tool_mask, completions, completion_ids, tool_call_count, tool_failure_count, tool_images

    def _generate(self, prompts: list):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Copy the prompts to avoid modifying the original list
        prompts = copy.deepcopy(prompts)

        prompt_ids, images, multimodal_fields = self._tokenize_prompts(prompts)
        completion_ids = self._generate_single_turn(prompt_ids, images, multimodal_fields)

        # Extract tool calls from the completions and (possibly) execute them
        tool_images = []
        if self.tools:
            # Decode completions. It's important to use `parse_response` when possible, because it handles tool calls.
            if is_conversational({"prompt": prompts[0]}):
                if Version(transformers.__version__) >= Version("5.0.0") and (  # parse_response added in v5
                    getattr(self._tokenizer, "response_template", None) is not None  # new-style
                    or getattr(self._tokenizer, "response_schema", None) is not None  # old-style
                ):
                    completions = [
                        [parse_response(self._tokenizer, ids, prefix=prompt_ids[i])]
                        for i, ids in enumerate(completion_ids)
                    ]
                else:
                    contents = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
                    completions = [[{"role": "assistant", "content": content}] for content in contents]
            else:
                completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

            (
                tool_mask,
                completions,
                completion_ids,
                tool_call_count,
                tool_failure_count,
                tool_images,
            ) = self._tool_call_loop(prompts, prompt_ids, completion_ids, completions, images, multimodal_fields)
            # Merge tool response images into the images list for the forward pass
            if any(imgs for imgs in tool_images):
                if images is None:
                    images = [imgs if imgs else None for imgs in tool_images]
                else:
                    images = [(existing or []) + new for existing, new in zip(images, tool_images, strict=True)]
        else:
            tool_mask = None

        # Get completion length per sequence, used for logging
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        if tool_mask is not None:  # count only model-generated tokens (tool_mask=1)
            completion_lengths = torch.tensor([sum(mask) for mask in tool_mask], device=device)
        else:
            completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        # Fail clearly if the generation backend returned no completions (avoids a cryptic min() error below).
        if agg_completion_lengths.numel() == 0:
            raise RuntimeError(
                "No completions were generated. This usually means the generation backend failed to return any "
                "results; see the generation logs above for the underlying error."
            )
        total_prompt_tokens = agg_prompt_lengths.sum()

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + agg_completion_lengths.sum()).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        eos_and_pad = [*self.eos_token_ids, self._tokenizer.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        if self.tools:
            agg_tool_call_count = self.accelerator.gather(torch.tensor(tool_call_count, device=device)).sum()
            tool_call_frequency = (agg_tool_call_count / len(agg_prompt_lengths)).item()
            self._metrics[mode]["tools/call_frequency"].append(tool_call_frequency)
            agg_tool_failure_count = self.accelerator.gather(torch.tensor(tool_failure_count, device=device)).sum()
            failure_frequency = (
                (agg_tool_failure_count / agg_tool_call_count).item() if agg_tool_call_count > 0 else 0.0
            )
            self._metrics[mode]["tools/failure_frequency"].append(failure_frequency)

        return prompt_ids, completion_ids, tool_mask, images, tool_images

    @profiling_decorator
    def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        spatial_shapes=None,
        image_sizes=None,
        token_type_ids=None,
        mm_token_type_ids=None,
        image_position_ids=None,
    ):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model

        # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # For Qwen models:
        if image_grid_thw is not None and pixel_values is not None:
            model_inputs["image_grid_thw"] = image_grid_thw
        # For Gemma, SmolVLM2, LLaVa-Next etc.:
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        # For SmolVLM2
        if pixel_attention_mask is not None:
            model_inputs["pixel_attention_mask"] = pixel_attention_mask
        # For LFM2-VL
        if spatial_shapes is not None:
            model_inputs["spatial_shapes"] = spatial_shapes
        # For LLaVa-Next
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        if mm_token_type_ids is not None:
            model_inputs["mm_token_type_ids"] = mm_token_type_ids
        if image_position_ids is not None:
            model_inputs["image_position_ids"] = image_position_ids

        # Only add logits_to_keep if the model supports it
        if "logits_to_keep" in self.model_kwarg_keys:
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

        # `base_model` gives the backbone model (skipping `lm_head`) — text decoder for LMs, multimodal wrapper for
        # VLMs (so vision-token injection runs before the text decoder). `get_decoder()` won't do: on VLMs it
        # returns just the text stack and feeds image-placeholder IDs through it.
        # Pre-5.0 transformers VLMs set `base_model_prefix = ""` so `base_model is self` (re-runs `lm_head`).
        # Fall back to `.model` there.
        if self._is_vlm and Version(transformers.__version__) < Version("5.0.0"):
            backbone = unwrapped_model.model
        else:
            backbone = unwrapped_model.base_model
        last_hidden_state = backbone(**model_inputs).last_hidden_state
        # Exclude the last value: it corresponds to the next token pred
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
        last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    # Name kept aligned with GRPO/RLOO for consistency; distillation has no rewards, so nothing is actually scored.
    def _generate_and_score_completions(self, inputs: list[dict[str, torch.Tensor | Any]]) -> dict[str, Any]:
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image", "image": <Image>}, {"type": "text", "text": "What color is the sky?"}]}]
        if images is not None:
            if not is_conversational(inputs[0]):
                raise ValueError(
                    "Multimodal training requires conversational prompts. It looks like the dataset contains "
                    "non-conversational inputs, likely because a chat template was applied before passing the dataset "
                    "to the trainer. Please provide the raw conversational prompts and let the trainer apply the chat "
                    "template internally."
                )
            prompts = [
                prepare_multimodal_messages(prompt, images=image_list)
                for prompt, image_list in zip(prompts, images, strict=True)
            ]

        prompt_ids_list, completion_ids_list, tool_mask_list, images, tool_images = self._generate(prompts)

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(
            prompt_ids,
            padding_value=self._tokenizer.pad_token_id,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        prompt_mask = pad(
            prompt_mask, padding_value=0, padding_side="left", pad_to_multiple_of=self.pad_to_multiple_of
        ).to(device=device)
        completion_ids = [torch.tensor(ids) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(
            completion_ids,
            padding_value=self._tokenizer.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        completion_mask = pad(
            completion_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
        ).to(device=device)
        if tool_mask_list is not None:
            tool_mask = [torch.tensor(mask) for mask in tool_mask_list]
            tool_mask = pad(
                tool_mask, padding_value=1, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            ).to(device=device)
        else:
            tool_mask = None

        loss_mask = completion_mask if tool_mask is None else completion_mask * tool_mask
        num_items_in_batch = self.accelerator.gather(loss_mask.sum()).sum()

        num_images = [len(img_list) if img_list else 0 for img_list in images] if images is not None else None

        # Get forward_kwargs for models with multimodal inputs.
        # When tool images are present (from _tool_call_loop), use image_processor directly and build
        # mm_token_type_ids from prompt_completion_ids. Otherwise, use the full processor pipeline
        # which returns model-specific keys (image_sizes, pixel_attention_mask, etc.).
        if self.tools and any(imgs for imgs in tool_images) and self._is_vlm:
            flat_images = [img for img_list in images if img_list for img in img_list]
            image_inputs = self.processing_class.image_processor(images=flat_images, return_tensors="pt")
            image_inputs = super()._prepare_inputs(image_inputs)
            forward_kwargs = dict(image_inputs)
        elif images is not None:
            prompts_text = [
                apply_chat_template(
                    {"prompt": prompt}, self.processing_class, tools=self.tools or None, **self.chat_template_kwargs
                )["prompt"]
                for prompt in prompts
            ]
            prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        # Recover LFM2-VL tile counts; the full processor drops row/column metadata.
        num_tiles = None
        if images is not None and "spatial_shapes" in forward_kwargs:
            image_info = self.processing_class.image_processor(
                images=images, return_tensors="pt", return_row_col_info=True
            )
            tiles_per_image = image_info["image_rows"] * image_info["image_cols"]
            if self.processing_class.image_processor.use_thumbnail:
                tiles_per_image = tiles_per_image + (tiles_per_image > 1).to(tiles_per_image.dtype)
            num_tiles = [group.sum().item() for group in torch.split(tiles_per_image, num_images)]
        # Same for InternVL, whose pixel_values is tile-indexed ([total_tiles, channels, height, width]).
        elif (
            images is not None
            and forward_kwargs["pixel_values"].ndim == 4
            and forward_kwargs["pixel_values"].size(0) != sum(num_images)
        ):
            num_patches = self.processing_class.image_processor(
                images=images, crop_to_patches=True, return_tensors="pt"
            )["num_patches"]
            num_tiles = [group.sum().item() for group in torch.split(num_patches, num_images)]

        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            if self.pad_to_multiple_of is not None:
                # Needed only with pad_to_multiple_of: otherwise prompt_ids and token_type_ids must have equal len
                padding_size = prompt_ids.size(1) - token_type_ids.size(1)
                if padding_size > 0:
                    token_type_ids = torch.cat(
                        [token_type_ids.new_zeros((token_type_ids.size(0), padding_size)), token_type_ids], dim=1
                    )
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )
        # If mm_token_type_ids are used, extend them with zeros for the completion part
        if "mm_token_type_ids" in forward_kwargs:
            mm_token_type_ids = forward_kwargs["mm_token_type_ids"]
            if self.pad_to_multiple_of is not None:
                # Needed only with pad_to_multiple_of: otherwise prompt_ids and mm_token_type_ids must have equal len
                padding_size = prompt_ids.size(1) - mm_token_type_ids.size(1)
                if padding_size > 0:
                    mm_token_type_ids = torch.cat(
                        [mm_token_type_ids.new_zeros((mm_token_type_ids.size(0), padding_size)), mm_token_type_ids],
                        dim=1,
                    )
            forward_kwargs["mm_token_type_ids"] = torch.cat(
                [mm_token_type_ids, mm_token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        # For VLM tool images: build token type IDs from the full prompt_completion_ids.
        # This must happen AFTER the token_type_ids/mm_token_type_ids extension blocks above,
        # because our version already covers the full sequence (images are in the completion,
        # not just the prompt).
        if self.tools and any(imgs for imgs in tool_images) and self._is_vlm:
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
            mm_ids = torch.zeros_like(prompt_completion_ids)
            if self._image_pad_token_id is not None:
                mm_ids[prompt_completion_ids == self._image_pad_token_id] = 1
            if self._video_pad_token_id is not None:
                mm_ids[prompt_completion_ids == self._video_pad_token_id] = 2

            # Use the same key the model expects: token_type_ids for models like Gemma,
            # mm_token_type_ids for models like Qwen.
            image_grid_thw = forward_kwargs.get("image_grid_thw")
            if image_grid_thw is not None:
                forward_kwargs["mm_token_type_ids"] = mm_ids
            else:
                forward_kwargs["token_type_ids"] = mm_ids

            # Truncation safety (Qwen-style models with image_grid_thw only): if
            # max_completion_length truncated some image tokens, the number of image pad tokens
            # in input_ids won't match pixel_values features. Check per-sample and drop ALL
            # images for any sample with a mismatch (safe fallback).
            if image_grid_thw is not None and num_images is not None:
                merge_length = getattr(self.processing_class.image_processor, "merge_size", 2) ** 2
                img_offset = 0
                has_mismatch = False
                for b in range(mm_ids.shape[0]):
                    sample_tokens = (mm_ids[b] == 1).sum().item()
                    sample_features = 0
                    for i in range(num_images[b]):
                        grid_idx = img_offset + i
                        if grid_idx < image_grid_thw.shape[0]:
                            sample_features += image_grid_thw[grid_idx].prod().item() // merge_length
                    if sample_tokens != sample_features:
                        has_mismatch = True
                        break
                    img_offset += num_images[b]

                if has_mismatch:
                    # Drop all images: safer than partial trim which is error-prone
                    forward_kwargs.pop("pixel_values", None)
                    forward_kwargs.pop("image_grid_thw", None)
                    mm_ids.zero_()
                    forward_kwargs["mm_token_type_ids"] = mm_ids
                    num_images = None

        # Log prompt and completion texts
        if self.log_completions:
            prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            self._logs["prompt"].extend(gather_object(prompts_text))
            self._logs["completion"].extend(gather_object(completions_text))

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "num_items_in_batch": num_items_in_batch,
        }
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "spatial_shapes" in forward_kwargs:
            output["spatial_shapes"] = forward_kwargs["spatial_shapes"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if "mm_token_type_ids" in forward_kwargs:
            output["mm_token_type_ids"] = forward_kwargs["mm_token_type_ids"]
        if "image_position_ids" in forward_kwargs:
            output["image_position_ids"] = forward_kwargs["image_position_ids"]
        if images is not None:
            output["num_images"] = num_images
            if num_tiles is not None:
                output["num_tiles"] = num_tiles
        if tool_mask is not None:
            output["tool_mask"] = tool_mask
        return output

    @profiling_decorator
    def _prepare_inputs(self, generation_batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × gradient accumulation steps)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every gradient_accumulation_steps)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.gradient_accumulation_steps
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = split_pixel_values_by_grid(generation_batch)

                try: generation_batch = shuffle_sequence_dict(generation_batch)

                except: pass
                generation_batches = split_tensor_dict(generation_batch, self.args.gradient_accumulation_steps)
                self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # transformers computes `num_items_in_batch` from the raw dataloader labels, before on-policy generation
        # replaces the completions; use the count over the generated completions instead (computed in
        # `_generate_and_score_completions`). Divide by the process count so the per-process loss compensates for DDP
        # gradient averaging.
        if self.model.training and inputs.get("num_items_in_batch") is not None:
            num_items_in_batch = inputs["num_items_in_batch"].clamp(min=1.0) / self.accelerator.num_processes

        # Route the whole loss (backbone + `lm_head` projection + JSD) through the DDP/FSDP wrapper via
        # `_forward_redirection`, so DDP.forward() fires `prepare_for_backward()` and FSDP/DeepSpeed keep the student's
        # sharded parameters (including the `lm_head`) materialized for the projection.
        unwrapped_student = self.accelerator.unwrap_model(model)
        loss, entropy_sum, num_valid_tokens = self._forward_redirection(
            model, unwrapped_student, self._compute_loss, unwrapped_student, inputs, num_items_in_batch
        )

        # Log the mean per-token student entropy (in nats). The reduction runs here, after `_forward_redirection`
        # returns, so the `gather_for_metrics` collective does not run inside the DDP/FSDP-wrapped forward (a hang/
        # ordering risk). Mirrors `SFTTrainer.compute_loss`.
        mode = "train" if self.model.training else "eval"
        num_valid_tokens = self.accelerator.gather_for_metrics(num_valid_tokens).sum()
        entropy_sum = self.accelerator.gather_for_metrics(entropy_sum).sum()
        entropy = (entropy_sum / num_valid_tokens).item() if num_valid_tokens > 0 else 0.0
        self._metrics[mode]["entropy"].append(entropy)

        return (loss, None) if return_outputs else loss

    def _compute_loss(self, unwrapped_student, inputs, num_items_in_batch):
        # Chunked JSD path: project the teacher/student hidden states to vocab logits one chunk at a time (never
        # materializing the full `(B, C, V)` logits), so the teacher's dense distribution can be matched without
        # buffering it. Runs inside the student wrapper's forward (see `compute_loss`).
        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        completion_mask = inputs["completion_mask"]
        # Apply tool_mask for loss computation in multi-turn training scenarios
        loss_mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]
        logits_to_keep = inputs["completion_ids"].size(1)  # only the completion tokens are trained on

        # Multimodal (VLM) fields, extracted during generation and split to this micro-batch by `_prepare_inputs`.
        multimodal_keys = (
            "pixel_values",
            "image_grid_thw",
            "pixel_attention_mask",
            "spatial_shapes",
            "image_sizes",
            "token_type_ids",
            "mm_token_type_ids",
            "image_position_ids",
        )
        multimodal_inputs = {k: inputs[k] for k in multimodal_keys if k in inputs}

        student_hidden_states = self._get_last_hidden_state(
            unwrapped_student, input_ids, attention_mask, logits_to_keep, **multimodal_inputs
        )

        # Route the teacher backbone through its own wrapper via `_forward_redirection` too, so FSDP/DeepSpeed
        # materialize its sharded parameters before the forward runs (the backbone call would otherwise see shards).
        self.teacher_model.eval()
        unwrapped_teacher = self.accelerator.unwrap_model(self.teacher_model)
        with torch.no_grad():
            teacher_hidden_states = self._forward_redirection(
                self.teacher_model,
                unwrapped_teacher,
                self._get_last_hidden_state,
                unwrapped_teacher,
                input_ids,
                attention_mask,
                logits_to_keep,
                **multimodal_inputs,
            )

        student_lm_head = unwrapped_student.get_output_embeddings()
        teacher_lm_head = unwrapped_teacher.get_output_embeddings()

        # On VLMs the logit post-processing lives on `text_config`, so read it through `get_text_config()`.
        student_config = unwrapped_student.config.get_text_config()
        teacher_config = unwrapped_teacher.config.get_text_config()
        # `logit_scale` is None on models that don't scale (e.g. MPT); read that as unscaled (1.0). A real 0.0 is kept
        # as-is. Muse Glimmer applies the same pre-softcap multiplier under the name `output_multiplier`.
        student_logit_scale = getattr(student_config, "logit_scale", None)
        if student_logit_scale is None:
            student_logit_scale = getattr(student_config, "output_multiplier", None)
        # Falcon-H1 multiplies the logits by `lm_head_multiplier`. Granite and MiniCPM3 divide them by
        # `logits_scaling`, while HyperCLOVA X multiplies them by it.
        if student_logit_scale is None:
            student_logit_scale = getattr(student_config, "lm_head_multiplier", None)
        if student_logit_scale is None and getattr(student_config, "logits_scaling", None) is not None:
            logits_scaling = student_config.logits_scaling
            student_logit_scale = logits_scaling if student_config.model_type == "hyperclovax" else 1 / logits_scaling
        teacher_logit_scale = getattr(teacher_config, "logit_scale", None)
        if teacher_logit_scale is None:
            teacher_logit_scale = getattr(teacher_config, "output_multiplier", None)
        # Falcon-H1 multiplies the logits by `lm_head_multiplier`. Granite and MiniCPM3 divide them by
        # `logits_scaling`, while HyperCLOVA X multiplies them by it.
        if teacher_logit_scale is None:
            teacher_logit_scale = getattr(teacher_config, "lm_head_multiplier", None)
        if teacher_logit_scale is None and getattr(teacher_config, "logits_scaling", None) is not None:
            logits_scaling = teacher_config.logits_scaling
            teacher_logit_scale = logits_scaling if teacher_config.model_type == "hyperclovax" else 1 / logits_scaling
        student_logit_scale = 1.0 if student_logit_scale is None else student_logit_scale
        teacher_logit_scale = 1.0 if teacher_logit_scale is None else teacher_logit_scale
        loss, entropy_sum, n_valid = _chunked_divergence_loss(
            student_hidden_states,
            teacher_hidden_states,
            student_lm_head.weight,
            teacher_lm_head.weight,
            loss_mask,
            self.beta,
            num_items_in_batch=num_items_in_batch,
            student_lm_head_bias=student_lm_head.bias,
            teacher_lm_head_bias=teacher_lm_head.bias,
            student_logit_scale=student_logit_scale,
            teacher_logit_scale=teacher_logit_scale,
            student_final_logit_softcapping=getattr(student_config, "final_logit_softcapping", None),
            teacher_final_logit_softcapping=getattr(teacher_config, "final_logit_softcapping", None),
            temperature=self.temperature,
        )
        # Return the raw entropy sum and valid-token count for `compute_loss` to aggregate and log after the forward
        # returns (see there). Detached: the metric is gradient-free.
        return loss, entropy_sum.detach(), n_valid

    def training_step(self, model, inputs, num_items_in_batch):
        time_before = time.perf_counter()
        output = super().training_step(model, inputs, num_items_in_batch)
        self._step += 1
        time_after = time.perf_counter()
        self._current_train_step_time += time_after - time_before
        if self._step % self.current_gradient_accumulation_steps == 0:
            self._metrics["train"]["step_time"].append(self._current_train_step_time)
            self._current_train_step_time = 0.0
        return output

    # During eval, Trainer calls prediction_step. If no labels are present in the inputs, it only runs forward and
    # returns logits. We override prediction_step to force compute_loss, because this trainer doesn't involve labels.
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: list[str] | None = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        # Average the metrics
        metrics = {}
        for key, val in self._metrics[mode].items():
            # Filter out NaN values before averaging. With logging_steps > 1, a naive sum()/len() would let a single
            # NaN contaminate valid data from other batches. Only return None when no valid values remain (e.g. JSON
            # loggers crash on float NaN).
            valid = [v for v in val if not math.isnan(v)]
            metrics[key] = sum(valid) / len(valid) if valid else None

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs.update(metrics)
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._logs["prompt"],
                    self._logs["completion"],
                    {},
                    None,
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            logging_backends = []
            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                logging_backends.append(wandb)
            if self.args.report_to and "trackio" in self.args.report_to:
                logging_backends.append(trackio)

            import pandas as pd

            table = {
                "step": [self.state.global_step] * len(self._logs["prompt"]),
                "prompt": self._logs["prompt"],
                "completion": self._logs["completion"],
            }
            df_base = pd.DataFrame(table)
            df_base.to_parquet(
                os.path.join(
                    self.args.output_dir,
                    "completions",
                    f"completions_{self.state.global_step:05d}.parquet",
                )
            )

            for logging_backend in logging_backends:
                df = df_base
                if self.log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                logging_backend.log({"completions": logging_backend.Table(dataframe=df)})

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
class UnslothDistillationTrainer(_UnslothDistillationTrainer):
    """
    
    Trainer for knowledge distillation. The student is trained on-policy — it generates the completions itself — to
    match the teacher's next-token distribution under a generalized Jensen-Shannon divergence (interpolating forward
    KL, reverse KL, and JSD via `beta`), as introduced in [On-Policy Distillation of Language
    Models](https://huggingface.co/papers/2306.13649).

    Example:

    ```python
    >>> from trl import DistillationTrainer
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("trl-lib/tldr", split="train")

    >>> trainer = DistillationTrainer(
    ...     model="Qwen/Qwen2.5-0.5B-Instruct",
    ...     teacher_model="Qwen/Qwen2.5-1.5B-Instruct",
    ...     train_dataset=dataset,
    ... )
    >>> trainer.train()
    ```

    Args:
        model (`str` or [`~transformers.PreTrainedModel`] or [`~peft.PeftModel`]):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using `<ModelArchitecture>.from_pretrained` (where `<ModelArchitecture>` is derived from the model
              config) with the keyword arguments in `args.model_init_kwargs`. If `dtype` is not specified in
              `args.model_init_kwargs`, it defaults to `float32`. This differs from
              [`~transformers.PreTrainedModel.from_pretrained`], where (since Transformers v5) the dtype is inferred
              from the model config.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
            - A [`~peft.PeftModel`] object. Only causal language models are supported.
        teacher_model (`str` or [`~transformers.PreTrainedModel`], *optional*):
            Teacher model whose next-token distribution the student is trained to match. Can be a *model id* / path
            (loaded like `model`, using `args.teacher_model_init_kwargs`) or an instantiated
            [`~transformers.PreTrainedModel`]. It must share the student's vocabulary. May be omitted by subclasses
            that supply the teacher another way (e.g. a remote server).
        args ([`DistillationConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).

            When `train_dataset` is an [`~datasets.IterableDataset`] (e.g. a streaming dataset), `max_steps` must be
            set in the training arguments, since its length cannot be inferred and the total number of training steps
            is required to bound the training loop and configure the learning rate scheduler.
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`], [`~datasets.DatasetDict`], [`~datasets.IterableDatasetDict`] or `dict[str, Dataset | IterableDataset]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of `AdamW` on your
            model and a scheduler given by [`~transformers.get_linear_schedule_with_warmup`] controlled by `args`.
        quantization_config ([`~transformers.BitsAndBytesConfig`], *optional*):
            Quantization configuration used when loading the model from a model identifier. Combine with `peft_config`
            for QLoRA training. Ignored if the model is already instantiated.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        tools (list of `Callable`, *optional*):
            A list of callable tool functions that the model can invoke during generation. Each tool should be a
            standard Python function with properly type-hinted arguments and return values, and a Google-style
            docstring describing its purpose, arguments, and return value. For more details, see:
            https://huggingface.co/docs/transformers/en/chat_extras#passing-tools. The model uses the function's name,
            type hints, and docstring to determine how to call it. Ensure that the model's chat template supports tool
            use and that it has been fine-tuned for tool calling.
    
    """
    def __init__(
        self,
        model,
        teacher_model = None,
        args = None,
        train_dataset = None,
        eval_dataset = None,
        processing_class = None,
        callbacks = None,
        quantization_config = None,
        peft_config = None,
        tools = None,
        **kwargs
    ):
        if args is None: args = UnslothDistillationConfig()
        _unsloth_precision_requested = tuple(getattr(args, k, False) for k in ('fp16', 'bf16', 'fp16_full_eval', 'bf16_full_eval'))
        use_bf16 = getattr(args, 'bf16', False)
        if type(use_bf16) is not bool: use_bf16 = False
        use_fp16 = getattr(args, 'fp16', False)
        if type(use_fp16) is not bool: use_fp16 = False
        force_float32 = False
        try:
            from unsloth_zoo.device_type import device_is_bf16_supported as _bf16_supported
        except Exception:
            _bf16_supported = torch.cuda.is_bf16_supported
        try:
            from unsloth_zoo.device_type import DEVICE_TYPE as _unsloth_device_type
        except Exception:
            _unsloth_device_type = None
        full_finetuning = getattr(model, '_unsloth_full_finetuning', None)
        if full_finetuning is None: full_finetuning = os.environ.get('UNSLOTH_ENABLE_FULL_FINETUNING', '0') == '1'
        model_forced_float32 = getattr(model, '_unsloth_forced_float32', None)
        if model_forced_float32 is None: model_forced_float32 = os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1'
        if model_forced_float32 and not (full_finetuning and _bf16_supported()):
            print('Unsloth: Switching to float32 training since model cannot work with float16')
            force_float32 = True
        mixed_precision_dtype = os.environ.get('UNSLOTH_MIXED_PRECISION', 'float32')
        dtype = getattr(model.config, 'dtype', None) or getattr(model.config, 'torch_dtype', None)
        if dtype is None: dtype = model.get_input_embeddings().weight.dtype
        from unsloth_zoo.utils import _get_dtype
        dtype = _get_dtype(dtype)
        float16 = dtype == torch.float16
        bfloat16 = dtype == torch.bfloat16
        float32 = dtype == torch.float32
        user_float32 = bool(getattr(model, '_unsloth_user_float32', False))
        if full_finetuning:
            if bfloat16 and use_fp16: use_fp16 = False
            if float16 and use_bf16: use_bf16 = False
        if not force_float32 and bfloat16 and use_fp16 and _unsloth_device_type == 'xpu' and not _bf16_supported():
            print('Unsloth: The model was loaded in bfloat16 on purpose, so training uses bfloat16 mixed precision instead of float16.')
            use_fp16 = False
            use_bf16 = True
            args.fp16 = False
            args.bf16 = True
        if not force_float32 and (float16 and use_bf16): raise TypeError('Unsloth: Model is in float16 precision but you want to use bfloat16 precision. Set fp16 to `True` and bf16 to `False`')
        if not force_float32 and (bfloat16 and use_fp16): raise TypeError('Unsloth: Model is in bfloat16 precision but you want to use float16 precision. Set fp16 to `False` and bf16 to `True`')
        if force_float32:
            # Forced float32 training
            args.fp16 = False
            args.bf16 = False
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'
            if hasattr(args, 'mixed_precision'): args.mixed_precision = 'no'
            # args.mixed_precision is a new argument which needs to be set now
        elif (not use_bf16 and not use_fp16) and mixed_precision_dtype == 'float32' and float32 and user_float32 and not _bf16_supported():
            print('Unsloth: Model is in float32 and this GPU has no bfloat16 support, so training stays in float32. Pass fp16 = True to force float16 mixed precision instead.')
            args.fp16 = False
            args.bf16 = False
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'
            if hasattr(args, 'mixed_precision'): args.mixed_precision = 'no'
        elif (not use_bf16 and not use_fp16) and mixed_precision_dtype == 'float32':
            # Mixed precision training. bf16 only if the GPU supports it; V100/T4 use fp16.
            use_bf16_amp = (not float16) and (_bf16_supported() or (bfloat16 and _unsloth_device_type == 'xpu'))
            args.fp16 = not use_bf16_amp
            args.bf16 = use_bf16_amp
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'bf16' if use_bf16_amp else 'fp16'
            if hasattr(args, 'mixed_precision'): args.mixed_precision = 'bf16' if use_bf16_amp else 'fp16'
            # args.mixed_precision is a new argument which needs to be set now
        elif mixed_precision_dtype == 'bfloat16':
            # Both False since bfloat16 full finetuning doesn't do any autocasting.
            args.fp16 = False
            args.bf16 = False
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'
            if hasattr(args, 'mixed_precision'): args.mixed_precision = 'no'
            # args.mixed_precision is a new argument which needs to be set now
        elif use_bf16 or use_fp16:
            # transformers <5 exported this itself from fp16/bf16; 5.x dropped the write, so an
            # explicit flag left it unset and GRPO readers defaulted to 'fp16', wrapping a
            # bfloat16 model in a float16 autocast. See unslothai/unsloth#4891.
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'bf16' if use_bf16 else 'fp16'
            if hasattr(args, 'mixed_precision'): args.mixed_precision = 'bf16' if use_bf16 else 'fp16'
        
        if getattr(args, 'eval_dataset', None) is not None and getattr(args, 'eval_strategy', 'no') == 'no':
            args.eval_strategy = 'steps'
            if getattr(args, 'eval_steps', None) is None: args.eval_steps = 0.1
        ga_steps = getattr(args, 'gradient_accumulation_steps', None)
        if ga_steps is not None and ga_steps > 1:
            from transformers import __version__ as transformers_version
            if Version(transformers_version) <= Version('4.45.2'):
                print('**** Unsloth: Please use our fixed gradient_accumulation_steps by updating transformers, TRL and Unsloth!\n'
                      '`pip install --upgrade --no-cache-dir --force-reinstall --no-deps unsloth transformers trl unsloth_zoo`')
        if getattr(args, 'eval_strategy', 'no') != 'no':
            eval_bsz = getattr(args, 'per_device_eval_batch_size', 8)
            eval_generations = getattr(args, 'num_generations_eval', None) or getattr(args, 'num_generations', None) or 1
            if eval_bsz == 8 and args.per_device_train_batch_size < eval_bsz and (args.per_device_train_batch_size * args.world_size) % eval_generations == 0: args.per_device_eval_batch_size = args.per_device_train_batch_size
            if getattr(args, 'eval_accumulation_steps', None) is None and ga_steps is not None: args.eval_accumulation_steps = ga_steps
        fp16_full_eval = getattr(args, 'fp16_full_eval', False)
        if type(fp16_full_eval) is not bool: fp16_full_eval = False
        bf16_full_eval = getattr(args, 'bf16_full_eval', False)
        if type(bf16_full_eval) is not bool: bf16_full_eval = False
        if args.fp16 and bf16_full_eval: args.bf16_full_eval = False; args.fp16_full_eval = True
        if args.bf16 and fp16_full_eval: args.bf16_full_eval = True; args.fp16_full_eval = False
        if force_float32:
            args.bf16_full_eval = False
            args.fp16_full_eval = False
        elif os.environ.get('UNSLOTH_MIXED_PRECISION', 'float32') == 'bfloat16':
            args.bf16_full_eval = True
            args.fp16_full_eval = False
        elif not bf16_full_eval and not fp16_full_eval:
            args.bf16_full_eval = args.bf16
            args.fp16_full_eval = args.fp16
        if tuple(getattr(args, k, False) for k in ('fp16', 'bf16', 'fp16_full_eval', 'bf16_full_eval')) != _unsloth_precision_requested:
            from unsloth.models.loader_utils import sync_deepspeed_mixed_precision
            sync_deepspeed_mixed_precision(args)
        if model is not None:
            from unsloth.models.loader_utils import apply_deepspeed_zero3_model_setup
            apply_deepspeed_zero3_model_setup(model)
        if model is not None:
            _warnings_issued = getattr(model, 'warnings_issued', None)
            if _warnings_issued is None:
                model.warnings_issued = {}
            elif not isinstance(_warnings_issued, dict):
                try:
                    model.warnings_issued = dict(_warnings_issued)
                except Exception:
                    model.warnings_issued = {}
        if 'max_seq_length' not in locals() and not hasattr(args, 'max_seq_length'):
            pass
        else:
            model_max_seq_length = getattr(model, 'max_seq_length', None)
            args_max_seq_length  = getattr(args,  'max_seq_length', None)
            if args_max_seq_length is None and model_max_seq_length is not None:
                max_seq_length = model.max_seq_length
                if hasattr(args, 'max_seq_length'): args.max_seq_length = max_seq_length
            elif args_max_seq_length is not None and model_max_seq_length is not None:
                if args_max_seq_length > model_max_seq_length:
                    print('Unsloth: You set `max_seq_length` as ' + str(args_max_seq_length) + ' but '
                           'the maximum the model supports is ' + str(model_max_seq_length) + '. We shall reduce it.')
                    args.max_seq_length = model_max_seq_length
        if model is not None and hasattr(model, 'for_training'):
            _use_gc = model._unsloth_gradient_checkpointing if hasattr(model, '_unsloth_gradient_checkpointing') else getattr(args, 'gradient_checkpointing', True)
            model.for_training(use_gradient_checkpointing=_use_gc)
        if 'tokenizer' in locals() and hasattr(tokenizer, 'padding_side'): tokenizer.padding_side = 'right'
        if 'processing_class' in locals():
            if hasattr(processing_class, 'padding_side'): processing_class.padding_side = 'right'
            if hasattr(processing_class, 'tokenizer') and hasattr(processing_class.tokenizer, 'padding_side'): processing_class.tokenizer.padding_side = 'right'
        other_metrics = []
        
        from unsloth_zoo.logging_utils import PatchRLStatistics
        PatchRLStatistics('distillation_trainer', other_metrics)
        
        # [TODO] Fix up DataParallel multiplying batch sizes
        # [TODO] DDP works, but DP seems to not work? [TODO]
        if getattr(args, "parallel_mode", None) == ParallelMode.NOT_DISTRIBUTED and args.n_gpu > 1:
            if getattr(args, "_n_gpu", 1) != 1:
                args._n_gpu = 1
        if "model" in locals():
            try:
                from unsloth.models._gc_config_default import honour_gradient_checkpointing_off
                honour_gradient_checkpointing_off(args, model)
            except Exception:
                pass
        if "model" in locals() and hasattr(model, "for_training"):
            _use_gc = model._unsloth_gradient_checkpointing if hasattr(model, '_unsloth_gradient_checkpointing') else getattr(args, 'gradient_checkpointing', True)
            model.for_training(use_gradient_checkpointing=_use_gc)
        if "model" in locals():
            try:
                from unsloth.models._ddp_defaults import unsloth_default_ddp_find_unused_parameters
                unsloth_default_ddp_find_unused_parameters(args, model)
            except Exception:
                pass
        super().__init__(
            model = model,
            teacher_model = teacher_model,
            args = args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            processing_class = processing_class,
            callbacks = callbacks,
            quantization_config = quantization_config,
            peft_config = peft_config,
            tools = tools,**kwargs)
        # A tensor parallel model (tp_native.tp_trainer_setup) stays on its training path: for_inference would
        # route the decoder layers onto decode paths outside the tensor parallel regions.
        if "model" in locals() and hasattr(model, "for_inference") and not getattr(self, "_unsloth_tensor_parallel", False):
            model.for_inference()
        if hasattr(self, 'neftune_hook_handle'):
            self.neftune_hook_handle.remove()
            if hasattr(self, 'neftune_hook_handle'): del self.neftune_hook_handle
        if getattr(args, 'neftune_noise_alpha', None) is not None:
            model.get_input_embeddings().neftune_noise_alpha = self.neftune_noise_alpha
        pass
        if hasattr(self, 'accelerator'):
            scaler = self.accelerator.scaler
            current_model = model
            while hasattr(current_model, 'model'):
                current_model.accelerator_scaler = scaler
                current_model = current_model.model
            current_model.accelerator_scaler = scaler
        pass
        if hasattr(self, 'train'):
            self.train = MethodType(prepare_for_training_mode(self.__class__.train), self)
        pass
        if hasattr(self, 'llm') and self.llm is not None and hasattr(self.llm, 'get_tokenizer'):
            _vllm_tok = self.llm.get_tokenizer()
            _pc = getattr(self, 'processing_class', None) or getattr(self, 'tokenizer', None)
            if _vllm_tok is not None and _pc is not None and getattr(_pc, 'chat_template', None) is not None and getattr(_vllm_tok, 'chat_template', None) is None:
                _vllm_tok.chat_template = _pc.chat_template
        pass
        if getattr(self, 'aux_loss_enabled', False) and hasattr(getattr(self, 'model', None), 'config'):
            _text_config = self.model.config
            if hasattr(_text_config, 'get_text_config'): _text_config = _text_config.get_text_config()
            _n_experts = [getattr(_text_config, _k) for _k in ('num_local_experts', 'num_experts', 'n_routed_experts', 'moe_num_experts') if isinstance(getattr(_text_config, _k, None), int)]
            if _n_experts and all(_n <= 1 for _n in _n_experts):
                self.aux_loss_enabled = False
                _text_config.output_router_logits = False
        pass
        
pass


if hasattr(logger, "addFilter"):
    import logging
    class HideLoggingMessage(logging.Filter):
        def __init__(self, text): self.text = text
        def filter(self, x): return not (self.text in x.getMessage())
    pass
    logger.addFilter(HideLoggingMessage("`use_cache=True`"))

