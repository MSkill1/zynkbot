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
from trl.experimental.gkd.gkd_trainer import (Any, AutoModelForCausalLM, BaseImageProcessor, Callable, DataCollator, DataCollatorForChatML, Dataset, EvalPrediction, F, FeatureExtractionMixin, GKDConfig, GKDTrainer, GenerationConfig, PeftConfig, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, SFTTrainer, TrainerCallback, disable_dropout_in_model, empty_cache, logger, nn, prepare_deepspeed, random, textwrap, torch, unwrap_model_for_generation, AutoModelForCausalLM, BaseImageProcessor, Callable, DataCollator, DataCollatorForChatML, Dataset, EvalPrediction, F, FeatureExtractionMixin, GKDConfig, GKDTrainer, GenerationConfig, PeftConfig, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, SFTTrainer, TrainerCallback, disable_dropout_in_model, logger, nn, prepare_deepspeed, torch, F, empty_cache, torch, Any, nn, random, torch, unwrap_model_for_generation)


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
def _unsloth_get_model_config(model):
    """Return HuggingFace model config, unwrapping DDP/Accelerate wrappers."""
    config = getattr(model, "config", None)
    if config is None and hasattr(model, "module"):
        config = getattr(model.module, "config", None)
    return config

def _unsloth_text_configs(config):
    """``config`` and its text sub-config, the two places a transform can live: Gemma-4-style configs keep it on ``config.text_config``, T5Gemma-style ones only reach it via ``config.get_text_config()``."""
    if config is None:
        return []
    holders = [config]
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is None:
        get_text_config = getattr(config, "get_text_config", None)
        if callable(get_text_config):
            try:
                text_cfg = get_text_config()
            except (TypeError, ValueError):
                text_cfg = None
    if text_cfg is not None and text_cfg is not config:
        holders.append(text_cfg)
    return holders

def _unsloth_get_final_logit_softcapping(model):
    """The soft cap the loss applies, under any of its three spellings: ``final_logit_softcapping`` (Gemma), ``logits_soft_cap`` (RecurrentGemma), ``output_logit_soft_cap`` (xLSTM). Returns 0 if unset."""
    config = _unsloth_get_model_config(model)
    if config is None:
        return 0
    for holder in _unsloth_text_configs(config):
        for name in ("final_logit_softcapping", "logits_soft_cap", "output_logit_soft_cap"):
            softcap = getattr(holder, name, None)
            if softcap:
                return softcap
    return 0

def _unsloth_resolve_logit_scales(model_config):
    """``(multiply, divide)`` for the logits, read with the same field table and per-family overrides ``detect_logit_transforms`` uses.

    Only reached when the installed unsloth_zoo predates that helper. Must stay in step with ``resolve_logit_transforms`` in unsloth/models/llama.py: if the forward applies a transform GRPO does not, the policy log-probabilities come from different logits than the ones generated, which silently shifts every importance ratio.
    """
    # Granite divides, HyperCLOVA X multiplies, MiniCPM3 scales hidden states (not a logit transform).
    overrides = {
        ("logits_scaling", "hyperclovax"): "multiply",
        ("logits_scaling", "minicpm3"): None,
    }
    found = {"multiply": 0, "divide": 0}
    for holder in _unsloth_text_configs(model_config):
        model_type = getattr(holder, "model_type", "") or ""
        for bucket, names in (
            ("multiply", ("logit_scale", "lm_head_multiplier", "output_multiplier")),
            ("divide", ("logits_scaling",)),
        ):
            for name in names:
                target = overrides.get((name, model_type), bucket)
                if target is None or found[target]:
                    continue
                value = getattr(holder, name, 0) or 0
                if value:
                    found[target] = value
                    break
    return found["multiply"], found["divide"]

def _unsloth_grpo_returns_hidden_states(model, tensor, lm_head):
    """Does ``tensor`` (a forward's ``.logits``) carry hidden states or real logits?

    ``_get_per_token_logps_and_entropies`` sets ``UNSLOTH_RETURN_HIDDEN_STATES=1``, but only a forward that honours the name hands hidden states back as ``.logits``; any other forward returns a real ``[.., vocab]`` tensor that must not reach the ``lm_head`` matmul.

    The primary test is an explicit signal that the forward honours the flag, both set outside this file: ``__UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__`` on the generated class, written by ``unsloth_zoo.compiler.create_standalone_class`` exactly when ``apply_fused_lm_head`` gave that forward its own branch; and ``_unsloth_grpo_hidden_states_forward_wrapped``, set by ``_install_grpo_hidden_states_forward_wrapper`` in ``unsloth/models/rl.py`` for models the compiler did not rewrite. That wrapper degrades to real logits when the model cannot produce hidden states and records whether it did so in ``_unsloth_grpo_hidden_states_degraded`` before returning, so reading the pair after a forward describes the call that just finished; degradation is per call, not per model.

    The width comparison stays as the fallback, for an ``unsloth_zoo`` old enough that it never writes the marker. It is decisive whenever ``vocab_size != hidden_size``, and the signal may only overrule it when it is not, which is the one case the shape cannot answer.
    """
    if tensor.shape[-1] != lm_head.shape[1]:
        return False  # vocab-wide: real logits, whatever any signal claims
    if lm_head.shape[0] != lm_head.shape[1]:
        return True  # hidden-wide and vocab_size != hidden_size: hidden states
    return _unsloth_grpo_hidden_states_signal(model) is not False

def _unsloth_grpo_hidden_states_signal(model):
    """``True``/``False`` if the forward honours ``UNSLOTH_RETURN_HIDDEN_STATES``, ``None`` when neither marker is present. See ``_unsloth_grpo_returns_hidden_states`` for where each marker is set. Walks the wrapper chain because the markers are set on whichever object the trainer saw, which may be the DDP module or the PEFT base model rather than the object handed to the logprob loop."""
    candidates = []
    pending = [model]
    while pending and len(candidates) < 8:
        candidate = pending.pop(0)
        if candidate is None or any(candidate is seen for seen in candidates):
            continue
        candidates.append(candidate)
        get_base_model = getattr(candidate, "get_base_model", None)
        if callable(get_base_model):
            try:
                pending.append(get_base_model())
            except Exception:
                pass
        for _attr in ("module", "base_model", "model"):
            child = getattr(candidate, _attr, None)
            if child is not None and hasattr(child, "forward"):
                pending.append(child)
    for candidate in candidates:
        if getattr(candidate, "__UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__", False):
            return True
        if getattr(type(candidate), "__UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__", False):
            return True
    if any(
        getattr(candidate, "_unsloth_grpo_hidden_states_forward_wrapped", False)
        for candidate in candidates
    ):
        # The wrapper honours the flag unless it recorded that this call could not.
        if any(
            hasattr(candidate, "_unsloth_grpo_hidden_states_degraded") for candidate in candidates
        ):
            return not any(
                getattr(candidate, "_unsloth_grpo_hidden_states_degraded", False)
                for candidate in candidates
            )
        # An unsloth/models/rl.py predating the per-call attribute set only the warn-once flag; it is the best signal such a wrapper offers.
        return not any(
            getattr(candidate, "_unsloth_grpo_hidden_states_warning_issued", False)
            for candidate in candidates
        )
    return None

def _unsloth_gkd_canonical(source):
    """``ast.unparse`` without docstrings/comments so checks survive reformatting; ``None`` if unparsable."""
    import ast as _ast
    import textwrap as _textwrap

    try:
        tree = _ast.parse(_textwrap.dedent(source))
    except Exception:
        return None
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (
                isinstance(first, _ast.Expr)
                and isinstance(getattr(first, "value", None), _ast.Constant)
                and isinstance(first.value.value, str)
            ):
                node.body = node.body[1:] or [_ast.Pass()]
    try:
        return _ast.unparse(tree)
    except Exception:
        return None

def _unsloth_gkd_jsd_supported(trainer_class):
    """Is ``trainer_class.generalized_jsd_loss`` one of TRL's own generalized JSDs, unchanged? Overrides keep the dense loss."""
    cache = trainer_class.__dict__.get("_unsloth_gkd_jsd_supported_cache", None)
    if cache is not None:
        return cache
    required = (
        "student_logits = student_logits / temperature",
        "teacher_logits = teacher_logits / temperature",
        "student_log_probs = F.log_softmax(student_logits, dim=-1)",
        "teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)",
        "if beta == 0:",
        "jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction='none', log_target=True)",
        "elif beta == 1:",
        "jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction='none', log_target=True)",
        "kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction='none', log_target=True)",
        "kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction='none', log_target=True)",
        "jsd = beta * kl_teacher + (1 - beta) * kl_student",
        "mask = labels != -100",
        "jsd = jsd[mask]",
        "if reduction == 'batchmean':",
    )
    known = set(required) | {
        "@staticmethod",
        "def generalized_jsd_loss(student_logits, teacher_logits, labels=None, beta=0.5, temperature=1.0, reduction='batchmean'):",
        "def generalized_jsd_loss(student_logits, teacher_logits, labels=None, beta=0.5, temperature=1.0, reduction='batchmean', num_items_in_batch=None):",
        "else:",
        "if labels is not None:",
        "beta = torch.tensor(beta, dtype=student_log_probs.dtype)",
        "beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)",
        "mixture_log_probs = torch.logsumexp(torch.stack([student_log_probs + torch.log(1 - beta), teacher_log_probs + torch.log(beta)]), dim=0)",
        "mixture_log_probs = torch.logsumexp(torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]), dim=0)",
        "if num_items_in_batch is not None:",
        "jsd_sum = jsd.sum()",
        "if isinstance(num_items_in_batch, torch.Tensor):",
        "num_items_in_batch = num_items_in_batch.to(jsd_sum.device)",
        "return jsd_sum / num_items_in_batch",
        "return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / (jsd.size(0) * jsd.size(1))",
        "return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)",
        "denom = mask.sum().clamp_min(1) if labels is not None else max(jsd.size(0), 1)",
        "return jsd.sum() / denom",
        "elif reduction == 'sum':",
        "return jsd.sum()",
        "elif reduction == 'mean':",
        "return jsd.mean()",
        "return jsd",
    }
    ok = False
    try:
        source = _unsloth_gkd_canonical(inspect.getsource(trainer_class.generalized_jsd_loss))
        lines = [line.strip() for line in source.splitlines() if line.strip()]
        signature = next((line for line in lines if line.startswith("def ")), "")
        ok = (
            all(line in known for line in lines)
            and all(line in lines for line in required)
            # a num_items_in_batch parameter must come with its reduction, and vice versa
            and (
                ("num_items_in_batch=None" in signature)
                == ("return jsd_sum / num_items_in_batch" in lines)
            )
        )
    except Exception:
        ok = False
    try:
        setattr(trainer_class, "_unsloth_gkd_jsd_supported_cache", ok)
    except Exception:
        pass
    return ok

def _unsloth_gkd_note_fallback(trainer, reason):
    """Count why the chunked path declined (``trainer._unsloth_gkd_chunked_fallbacks``), log each reason once."""
    try:
        fallbacks = trainer.__dict__.setdefault("_unsloth_gkd_chunked_fallbacks", {})
    except Exception:
        return None
    if reason not in fallbacks:
        try:
            from unsloth_zoo.log import logger as _gkd_logger
            _gkd_logger.info(
                f"Unsloth: GKD chunked JSD not used ({reason}); using TRL's dense loss."
            )
        except Exception:
            pass
    fallbacks[reason] = fallbacks.get(reason, 0) + 1
    return None

def _unsloth_gkd_dense_head(model):
    """The output head when it is a plain dense ``[vocab, hidden]`` projection the chunked loss can read directly.
    Rejects bnb heads (``nn.Linear`` subclasses with packed weights), PEFT heads and ZeRO-3 partitioned weights.
    """
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    if not callable(get_output_embeddings):
        return None
    try:
        head = get_output_embeddings()
    except Exception:
        return None
    if type(head) is not torch.nn.Linear:
        return None
    # An lm_head that is not the output embeddings runs before them (ModernBERT decoder, RoBERTa-style heads).
    lm_head = getattr(model, "lm_head", None)
    if isinstance(lm_head, torch.nn.Module) and lm_head is not head:
        return None
    weight = getattr(head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2 or weight.numel() == 0:
        return None
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return None
    if type(weight) not in (torch.Tensor, torch.nn.Parameter):
        return None
    bias = getattr(head, "bias", None)
    if bias is not None and (bias.dim() != 1 or bias.shape[0] != weight.shape[0]):
        return None
    if weight.is_meta or (bias is not None and bias.is_meta):
        return None
    return head

def _unsloth_gkd_logit_transforms(model):
    """``(scale, softcap)`` exactly as the model's own forward applies them to its logits."""
    model_config = _unsloth_get_model_config(model)
    if detect_logit_transforms is not None:
        _transforms = detect_logit_transforms(model_config)
        logit_softcapping = _transforms["logit_softcapping"]
        logit_scale_multiply = _transforms["logit_scale_multiply"]
        logit_scale_divide = _transforms["logit_scale_divide"]
    else:
        logit_softcapping = _unsloth_get_final_logit_softcapping(model)
        logit_scale_multiply, logit_scale_divide = _unsloth_resolve_logit_scales(model_config)
    scale = 1.0
    if logit_scale_multiply:
        scale = scale * float(logit_scale_multiply)
    if logit_scale_divide:
        scale = scale / float(logit_scale_divide)
    return scale, float(logit_softcapping or 0.0)

def _unsloth_gkd_project(hidden_states, head, scale, softcap):
    """Full logits from hidden states, for the rare call where only one of the two forwards honoured the flag."""
    hidden_states = hidden_states.to(device = head.weight.device, dtype = head.weight.dtype)
    logits = torch.nn.functional.linear(hidden_states, head.weight, head.bias)
    if scale != 1.0:
        logits = logits * scale
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    return logits

def _unsloth_gkd_chunk_size(vocab_size):
    """Rows per chunk: ``UNSLOTH_GKD_CHUNK_SIZE`` when set, else about 2**26 logits per chunk, bounded to [64, 1024]."""
    requested = os.environ.get("UNSLOTH_GKD_CHUNK_SIZE", "")
    if requested.strip().isdigit() and int(requested) > 0:
        return int(requested)
    rows = (1 << 26) // max(int(vocab_size), 1)
    return int(min(1024, max(64, 1 << (max(rows, 1).bit_length() - 1))))

def _unsloth_gkd_right_align(
    inputs,
    layout,
    liger = False,
):
    """Roll left-padded rows (TRL's ChatML collator and generate both left-pad) so they start at column 0: Unsloth's training forward drops the 2D mask (#11885).
    The prompt layout's ``[:, P:]`` slice becomes ``labels[:, :P] = -100`` plus a one-column ``prompts``; TRL's Liger branch scores every label with no slice, so it only rolls.
    """
    try:
        attention_mask = inputs["attention_mask"]
        left_pad = (attention_mask.cumsum(dim = 1) == 0).sum(dim = 1, keepdim = True)
    except Exception:
        return inputs
    if not bool(left_pad.any()):
        return inputs
    aligned = dict(inputs)
    labels = inputs["labels"]
    if layout["shift"] == "prompt" and not liger:
        labels = labels.clone()
        labels[:, : inputs["prompts"].shape[1]] = -100
        aligned["prompts"] = inputs["prompts"][:, :1]
    width = attention_mask.shape[1]
    index = (torch.arange(width, device = attention_mask.device).unsqueeze(0) + left_pad) % width
    for key, value in (
        ("input_ids", inputs["input_ids"]),
        ("attention_mask", attention_mask),
        ("labels", labels),
    ):
        aligned[key] = value.gather(1, index)
    return aligned

def _unsloth_gkd_chunked_loss(self, model, inputs, num_items_in_batch, layout):
    """TRL's GKD loss without either full logits tensor, or ``None`` to run TRL's own ``compute_loss``."""
    if layout is None:
        return None
    if distillation_chunked_jsd is None and _unsloth_fused_linear_jsd is None:
        return _unsloth_gkd_note_fallback(self, "unsloth_zoo has no distillation_chunked_jsd")
    if os.environ.get("UNSLOTH_GKD_CHUNKED", "1").lower() in ("0", "false", "no", "off"):
        return _unsloth_gkd_note_fallback(self, "UNSLOTH_GKD_CHUNKED=0")
    # Read per call like the other fused lm_head routes: UNSLOTH_FUSED_LINEAR=0 restores TRL's dense loss.
    if os.environ.get("UNSLOTH_FUSED_LINEAR", "1") == "0":
        return _unsloth_gkd_note_fallback(self, "UNSLOTH_FUSED_LINEAR=0")
    use_fused = _unsloth_fused_linear_jsd is not None
    if getattr(self, "use_liger_gkd_loss", False):
        return _unsloth_gkd_note_fallback(self, "use_liger_gkd_loss")
    if getattr(self, "is_fsdp_enabled", False) or getattr(self, "is_deepspeed_enabled", False):
        return _unsloth_gkd_note_fallback(self, "FSDP / DeepSpeed")
    jsd_kwargs = {}
    scaler = getattr(self.accelerator, "scaler", None)
    if scaler is not None and scaler.is_enabled() and not use_fused:
        if "scaler" not in inspect.signature(distillation_chunked_jsd).parameters:
            return _unsloth_gkd_note_fallback(self, "unsloth_zoo JSD has no GradScaler support")
        jsd_kwargs["scaler"] = scaler
    try:
        missing = any(k not in inputs for k in ("input_ids", "attention_mask", "labels"))
    except Exception:
        missing = True
    if missing:
        return _unsloth_gkd_note_fallback(self, "inputs")
    if layout["shift"] == "prompt" and "prompts" not in inputs:
        return _unsloth_gkd_note_fallback(self, "inputs")
    teacher_model = getattr(self, "teacher_model", None)
    if teacher_model is None:
        return _unsloth_gkd_note_fallback(self, "no teacher_model")
    if not _unsloth_gkd_jsd_supported(type(self)):
        return _unsloth_gkd_note_fallback(self, "generalized_jsd_loss is not TRL's")
    try:
        unwrapped_student = self.accelerator.unwrap_model(model)
        unwrapped_teacher = self.accelerator.unwrap_model(teacher_model)
    except Exception:
        return _unsloth_gkd_note_fallback(self, "unwrap_model")
    student_head = _unsloth_gkd_dense_head(unwrapped_student)
    teacher_head = _unsloth_gkd_dense_head(unwrapped_teacher)
    if student_head is None or teacher_head is None:
        return _unsloth_gkd_note_fallback(self, "output head is not a dense nn.Linear")
    if student_head.weight.shape[0] != teacher_head.weight.shape[0]:
        return _unsloth_gkd_note_fallback(self, "vocab mismatch")
    # MiniCPM3 divides hidden states by logits_scaling before lm_head; a wrapped forward's hidden_states[-1] predates it.
    for unwrapped in (unwrapped_student, unwrapped_teacher):
        for config in _unsloth_text_configs(_unsloth_get_model_config(unwrapped)):
            if getattr(config, "model_type", None) == "minicpm3":
                return _unsloth_gkd_note_fallback(
                    self, "minicpm3 scales hidden states before the head"
                )
    # DDP(find_unused_parameters=True) marks a head skipped in forward as unused, then its grad hook fires twice.
    if getattr(model, "find_unused_parameters", False) and any(
        p is not None and p.requires_grad for p in (student_head.weight, student_head.bias)
    ):
        return _unsloth_gkd_note_fallback(self, "DDP find_unused_parameters with a trainable head")

    prior_hidden_states = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    try:
        student_outputs = model(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
        )
        teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids = inputs["input_ids"],
                attention_mask = inputs["attention_mask"],
            )
    finally:
        if prior_hidden_states is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = prior_hidden_states

    student_states = student_outputs.logits
    teacher_states = teacher_outputs.logits
    if layout["shift"] == "prompt":
        prompt_lengths = inputs["prompts"].shape[1]
        student_states = student_states[:, prompt_lengths - 1 : -1, :]
        teacher_states = teacher_states[:, prompt_lengths - 1 : -1, :]
        shifted_labels = inputs["labels"][:, prompt_lengths:]
    else:
        student_states = student_states[:, :-1, :]
        teacher_states = teacher_states[:, :-1, :]
        shifted_labels = inputs["labels"][:, 1:]

    student_scale, student_softcap = _unsloth_gkd_logit_transforms(unwrapped_student)
    teacher_scale, teacher_softcap = _unsloth_gkd_logit_transforms(unwrapped_teacher)
    items = num_items_in_batch if layout["num_items_in_batch"] else None

    student_hidden = _unsloth_grpo_returns_hidden_states(
        unwrapped_student, student_states, student_head.weight
    )
    teacher_hidden = _unsloth_grpo_returns_hidden_states(
        unwrapped_teacher, teacher_states, teacher_head.weight
    )
    if not (student_hidden and teacher_hidden):
        if student_hidden:
            student_states = _unsloth_gkd_project(
                student_states, student_head, student_scale, student_softcap
            )
        if teacher_hidden:
            with torch.no_grad():
                teacher_states = _unsloth_gkd_project(
                    teacher_states, teacher_head, teacher_scale, teacher_softcap
                )
        _unsloth_gkd_note_fallback(self, "a forward returned logits, not hidden states")
        # TRL's dense path gets both logits on the input device (accelerate's top-level hook), beside the labels.
        student_states = student_states.to(shifted_labels.device)
        teacher_states = teacher_states.to(shifted_labels.device)
        extra = {}
        if layout["num_items_in_batch"]:
            extra["num_items_in_batch"] = num_items_in_batch
        return self.generalized_jsd_loss(
            student_logits = student_states,
            teacher_logits = teacher_states,
            labels = shifted_labels,
            beta = self.beta,
            **extra,
        )

    # A dispatched model can return hidden states off its head's device; the zoo's accumulators follow the states.
    student_states = student_states.to(student_head.weight.device)
    teacher_states = teacher_states.to(teacher_head.weight.device)
    if use_fused:
        # Both heads one row chunk at a time, student gradient taken in forward (zoo fused_linear).
        rows = os.environ.get("UNSLOTH_GKD_CHUNK_SIZE", "")
        loss = _unsloth_fused_linear_jsd(
            student_states,
            student_head.weight,
            teacher_states.detach(),
            teacher_head.weight.detach(),
            labels = shifted_labels,
            beta = float(self.beta),
            # No TRL release passes a temperature to generalized_jsd_loss (GKDConfig.temperature is for sampling).
            temperature = 1.0,
            student_bias = student_head.bias,
            teacher_bias = None if teacher_head.bias is None else teacher_head.bias.detach(),
            student_logit_scale = student_scale,
            student_logit_softcapping = student_softcap,
            teacher_logit_scale = teacher_scale,
            teacher_logit_softcapping = teacher_softcap,
            num_items_in_batch = items,
            scaling = scaler.get_scale() if scaler is not None and scaler.is_enabled() else None,
            rows_per_chunk = int(rows) if rows.strip().isdigit() else None,
        )
        try:
            self._unsloth_gkd_chunked_calls = getattr(self, "_unsloth_gkd_chunked_calls", 0) + 1
            self._unsloth_gkd_fused_calls = getattr(self, "_unsloth_gkd_fused_calls", 0) + 1
        except Exception:
            pass
        return loss
    loss, _entropy_sum, _n_valid = distillation_chunked_jsd(
        student_states,
        teacher_states.detach(),
        student_head.weight,
        teacher_head.weight.detach(),
        shifted_labels != -100,
        beta = float(self.beta),
        chunk_size = _unsloth_gkd_chunk_size(student_head.weight.shape[0]),
        num_items_in_batch = items,
        student_lm_head_bias = student_head.bias,
        teacher_lm_head_bias = None if teacher_head.bias is None else teacher_head.bias.detach(),
        student_logit_scale = student_scale,
        teacher_logit_scale = teacher_scale,
        student_final_logit_softcapping = student_softcap,
        teacher_final_logit_softcapping = teacher_softcap,
        # No TRL release passes a temperature to generalized_jsd_loss (GKDConfig.temperature is for sampling).
        temperature = 1.0,
        **jsd_kwargs,
    )
    try:
        self._unsloth_gkd_chunked_calls = getattr(self, "_unsloth_gkd_chunked_calls", 0) + 1
    except Exception:
        pass
    return loss

import re
import inspect
try:
    from unsloth_zoo.device_map_planner import detect_logit_transforms
except Exception:
    detect_logit_transforms = None
try:
    from unsloth_zoo.rl_replacements import distillation_chunked_jsd
except Exception:
    distillation_chunked_jsd = None
try:
    from unsloth_zoo.fused_losses.fused_linear import fused_linear_jsd as _unsloth_fused_linear_jsd
except Exception:
    _unsloth_fused_linear_jsd = None
@dataclass
class UnslothGKDConfig(GKDConfig):
    """
    
    Configuration class for [`experimental.gkd.GKDTrainer`].

    This class includes only the parameters that are specific to GKD training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] and [`SFTConfig`] documentation.

    Args:
        temperature (`float`, *optional*, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        lmbda (`float`, *optional*, defaults to `0.5`):
            Lambda parameter that controls the student data fraction (i.e., the proportion of on-policy
            student-generated outputs).
        beta (`float`, *optional*, defaults to `0.5`):
            Interpolation coefficient between `0.0` and `1.0` of the Generalized Jensen-Shannon Divergence loss. When
            beta is `0.0`, the loss is the KL divergence. When beta is `1.0`, the loss is the Inverse KL Divergence.
        max_new_tokens (`int`, *optional*, defaults to `128`):
            Maximum number of tokens to generate per completion.
        teacher_model_name_or_path (`str`, *optional*):
            Model name or path of the teacher model. If `None`, the teacher model will be the same as the model being
            trained.
        teacher_model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the teacher model
            from a string.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model.
        seq_kd (`bool`, *optional*, defaults to `False`):
            Seq_kd parameter that controls whether to perform Sequence-Level KD (can be viewed as supervised FT on
            teacher-generated output).
    
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
    max_seq_length : Optional[int] = field(
        default = None,
        metadata = {'help': 'Maximum sequence length to truncate to.'},
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
        remove_unused_columns = True,
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
        router_aux_loss_coef = None,
        chat_template_path = None,
        dataset_text_field = 'text',
        dataset_kwargs = None,
        dataset_num_proc = None,
        eos_token = None,
        max_length = 1024,
        truncation_mode = 'keep_start',
        shuffle_dataset = False,
        packing = False,
        packing_strategy = 'bfd',
        padding_free = None,
        pad_to_multiple_of = None,
        eval_packing = None,
        completion_only_loss = None,
        assistant_only_loss = False,
        loss_type = 'nll',
        activation_offloading = False,
        pad_token = None,
        temperature = 0.9,
        lmbda = 0.5,
        beta = 0.5,
        max_new_tokens = 128,
        teacher_model_name_or_path = None,
        teacher_model_init_kwargs = None,
        disable_dropout = True,
        seq_kd = False,
        vllm_sampling_params = None,
        unsloth_num_chunks = -1,
        unsloth_logit_chunk_multiplier = None,
        unsloth_grpo_mini_batch = None,
        max_seq_length = None,
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
        try:
            from unsloth_zoo.dataset_num_proc import get_dataset_num_proc as _unsloth_get_dataset_num_proc
        except Exception:
            try:
                from unsloth.dataset_num_proc import get_dataset_num_proc as _unsloth_get_dataset_num_proc
            except Exception:
                _unsloth_get_dataset_num_proc = None
        if _unsloth_get_dataset_num_proc is not None:
            dataset_num_proc = _unsloth_get_dataset_num_proc(dataset_num_proc, serial_as_none = True)
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
            router_aux_loss_coef = router_aux_loss_coef,
            chat_template_path = chat_template_path,
            dataset_text_field = dataset_text_field,
            dataset_kwargs = dataset_kwargs,
            dataset_num_proc = dataset_num_proc,
            eos_token = eos_token,
            max_length = max_length,
            truncation_mode = truncation_mode,
            shuffle_dataset = shuffle_dataset,
            packing = packing,
            packing_strategy = packing_strategy,
            padding_free = padding_free,
            pad_to_multiple_of = pad_to_multiple_of,
            eval_packing = eval_packing,
            completion_only_loss = completion_only_loss,
            assistant_only_loss = assistant_only_loss,
            loss_type = loss_type,
            activation_offloading = activation_offloading,
            pad_token = pad_token,
            temperature = temperature,
            lmbda = lmbda,
            beta = beta,
            max_new_tokens = max_new_tokens,
            teacher_model_name_or_path = teacher_model_name_or_path,
            teacher_model_init_kwargs = teacher_model_init_kwargs,
            disable_dropout = disable_dropout,
            seq_kd = seq_kd,**kwargs)
        super().__init__(**_unsloth_filter_config_init_kwargs(GKDConfig, _unsloth_config_arguments, mirrored_from = __class__))
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
        self.max_seq_length = max_seq_length
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

class _UnslothGKDTrainer(SFTTrainer):
    """"""

    _tag_names = ["trl", "gkd"]
    _name = "GKD"
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
        model: PreTrainedModel | nn.Module | str | None = None,
        teacher_model: PreTrainedModel | nn.Module | str = None,
        args: GKDConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: "PeftConfig | None" = None,
        formatting_func: Callable | None = None,
    ):
        # Ensure Trainer does not drop non-signature columns used by the collator [e.g., "prompts"]
        args.remove_unused_columns = False
        # Respect a user-provided data_collator; otherwise, provide a ChatML collator that
        if data_collator is None:
            data_collator = DataCollatorForChatML(tokenizer=processing_class, max_length=args.max_length)

        # Ensure SFTTrainer does not pre-process the dataset when using a ChatML collator,
        # so that raw conversational fields [e.g., "messages"] remain available to the collator.
        if args.dataset_kwargs is None:
            args.dataset_kwargs = {"skip_prepare_dataset": True}
        else:
            args.dataset_kwargs["skip_prepare_dataset"] = True

        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            formatting_func=formatting_func,
        )

        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the GKDConfig, but your teacher_model is already instantiated."
            )
        else:
            teacher_model_init_kwargs = args.teacher_model_init_kwargs
            teacher_model_init_kwargs["dtype"] = (
                teacher_model_init_kwargs["dtype"]
                if teacher_model_init_kwargs["dtype"] in ["auto", None]
                else getattr(torch, teacher_model_init_kwargs["dtype"])
            )

        if isinstance(teacher_model, str):
            teacher_model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
            teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model, **teacher_model_init_kwargs)

        student_vocab_size = self.model.config.get_text_config().vocab_size
        teacher_vocab_size = teacher_model.config.get_text_config().vocab_size
        if student_vocab_size != teacher_vocab_size:
            raise ValueError(
                f"The student model has vocab_size {student_vocab_size} but the teacher model has "
                f"vocab_size {teacher_vocab_size}. GKD compares the teacher's full next-token "
                f"distribution, which requires a shared vocabulary. Use a teacher with the same vocab_size, or "
                f"GOLD for cross-tokenizer distillation."
            )

        # Disable dropout in the model
        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        if self.is_deepspeed_enabled:
            self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.seq_kd = args.seq_kd

        # With `lmbda=1.0` training is fully on-policy and `seq_kd` is never reached, and with `temperature=1.0` the
        # sampling temperature matches the one `DistillationTrainer` also applies to the divergence. In that setting,
        # `DistillationTrainer` covers this run and additionally supports vLLM generation and a chunked loss that
        # never materializes the full logits.
        if self.lmbda == 1.0 and self.temperature == 1.0:
            logger.warning(
                "This GKD configuration (`lmbda=1.0`, `temperature=1.0`) is fully covered by `DistillationTrainer`, "
                "which is maintained in the main codebase and supports vLLM generation and a memory-efficient "
                "chunked loss. Consider migrating: replace `GKDConfig`/`GKDTrainer` with "
                "`DistillationConfig`/`DistillationTrainer`, `max_new_tokens` with `max_completion_length`, and set "
                f"`beta={self.beta}` explicitly (`beta` means the same in both, but defaults to 0.5 here and 1.0 "
                "there)."
            )

        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "do_sample": True,
            "top_k": 0,
            "top_p": 1.0,
            "use_cache": False if args.gradient_checkpointing else True,
            "pad_token_id": self.processing_class.pad_token_id,
        }
        self.generation_config = GenerationConfig(**generation_kwargs)
        # Keep training-specific generation kwargs to overwrite model's original generation config
        self.generation_kwargs = generation_kwargs
        # Set custom EOS tokens if they are specified by the model's generation
        # config. This is important for models with the Llama 3 chat template,
        # which use special tokens <|eot_id|> and <|eom_id|> to mark the end of
        # turns or messages.
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

    @staticmethod
    def generalized_jsd_loss(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        reduction="batchmean",
        num_items_in_batch=None,
    ):
        """
        Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation using F.kl_div. See Eq. (1)
        of https://huggingface.co/papers/2306.13649 for the definition.

        Args:
            student_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            labels:
                Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing
                loss
            beta:
                Interpolation coefficient between 0 and 1 (default: 0.5)
            temperature:
                Softmax temperature (default: 1.0)
            reduction:
                Specifies the reduction to apply to the output (default: 'batchmean')

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """

        # Apply temperature scaling
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # Compute log probabilities for student and probabilities for teacher
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]),
                dim=0,
            )

            # Compute KL divergences using F.kl_div
            # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)

            # Compute the Generalized Jensen-Shannon Divergence
            jsd = beta * kl_teacher + (1 - beta) * kl_student

        # Masking
        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]

        # Apply reduction
        if num_items_in_batch is not None:
            # Normalize by the global number of valid tokens for gradient-accumulation-correct loss (see issue #4719).
            jsd_sum = jsd.sum()
            if isinstance(num_items_in_batch, torch.Tensor):
                num_items_in_batch = num_items_in_batch.to(jsd_sum.device)
            return jsd_sum / num_items_in_batch
        if reduction == "batchmean":
            # clamp_min(1) avoids 0/0 -> nan when a sample has no unmasked positions
            # (e.g. completion fully truncated). jsd[mask] is empty -> jsd.sum() == 0,
            # so 0/1 == 0 with a valid grad path.
            denom = mask.sum().clamp_min(1) if labels is not None else max(jsd.size(0), 1)
            return jsd.sum() / denom
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Unsloth: chunked generalized JSD over hidden states; TRL's own loss below is the fallback.
        if not return_outputs:
            inputs = _unsloth_gkd_right_align(
                inputs, {'shift': 'shift', 'num_items_in_batch': True}, getattr(self, 'use_liger_gkd_loss', False),
            )
            loss = _unsloth_gkd_chunked_loss(
                self, model, inputs, num_items_in_batch, {'shift': 'shift', 'num_items_in_batch': True},
            )
            if loss is not None:
                return loss
        return self._unsloth_trl_compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch,
        )

    def _unsloth_trl_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # compute student output
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        # compute teacher output in eval mode
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        # Standard causal shift: logits at position i predict the token at i + 1. The `labels != -100` mask
        # inside `generalized_jsd_loss` already excludes prompt (and padding) positions, so we do not slice by
        # prompt length. Slicing by `inputs["prompts"].shape[1]` (the batch-max prompt width) would drop real
        # completion tokens for samples whose prompt is shorter than the batch maximum, since `labels` is padded
        # to the full-sequence width independently of `prompts`.
        shifted_student_logits = student_outputs.logits[:, :-1, :]
        shifted_teacher_logits = teacher_outputs.logits[:, :-1, :]
        shifted_labels = inputs["labels"][:, 1:]

        # compute loss
        loss = self.generalized_jsd_loss(
            student_logits=shifted_student_logits,
            teacher_logits=shifted_teacher_logits,
            labels=shifted_labels,
            beta=self.beta,
            num_items_in_batch=num_items_in_batch,
        )

        # empty cache
        empty_cache()

        # Return loss
        return (loss, student_outputs) if return_outputs else loss

    @staticmethod
    def generate_on_policy_outputs(model, inputs, generation_config):
        # Generate output with respect to the prompt-only
        generated_outputs = model.generate(
            input_ids=inputs["prompts"],
            attention_mask=inputs["prompt_attention_mask"],
            generation_config=generation_config,
            return_dict_in_generate=True,
        )

        # Get the generated token IDs
        generated_tokens = generated_outputs.sequences
        device = generated_tokens.device
        prompt_mask = inputs["prompt_attention_mask"]
        prompt_length = inputs["prompts"].shape[1]
        completion_ids = generated_tokens[:, prompt_length:]

        # Mask everything after the first EOS token. eos_token_id can be a list of stop tokens (Llama 3),
        # so match with torch.isin; with no eos id nothing stops early, so keep the whole completion.
        if generation_config.eos_token_id is None:
            completion_mask = torch.ones_like(completion_ids)
        else:
            is_eos = torch.isin(completion_ids, torch.tensor(generation_config.eos_token_id, device=device))
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Pad ids can appear inside real prompt text, so use the prompt mask instead of matching ids
        new_attention_mask = torch.ones_like(generated_tokens)
        new_attention_mask[:, prompt_length:] = completion_mask
        new_attention_mask[:, :prompt_length] = prompt_mask

        new_labels = generated_tokens.clone()
        new_labels[new_attention_mask == 0] = -100

        # Mask the prompt so only the generated completion contributes to the loss. `generate` echoes
        # the prompt back as the first `prompt_length` columns, so masking them with -100 matches the
        # collator convention (`labels[:len(prompt)] = -100`) that `compute_loss` relies on.
        new_labels[:, :prompt_length] = -100

        return generated_tokens, new_attention_mask, new_labels

    def training_step(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        """
        Perform a training step for the Generalized Knowledge Distillation (GKD) model.

        This method implements the on-policy learning approach described in the GKD paper. With probability
        `self.lmbda`, it generates new responses using the student model, which are then used for training instead of
        the original inputs.
        """
        pending = getattr(self, "_unsloth_gkd_pending_batches", None)
        if pending is not None:
            batches, batch_args, batch_kwargs = pending
            self._unsloth_gkd_pending_batches = None

            def prepared_batches():
                for batch in batches:
                    self._unsloth_prepare_gkd_batch(model, batch)
                    yield batch

            _, count = super().get_batch_samples(
                prepared_batches(), len(batches), *batch_args, **batch_kwargs
            )
            # TRL 1.7-1.8 mask EOS when it is the pad token, so an all-EOS window counts 0: sum / 0 is NaN.
            if torch.is_tensor(count):
                count = count.clamp_min(1)
            elif count is not None:
                count = max(count, 1)
            for batch in batches:
                batch["_unsloth_gkd_num_items"] = count

        if "_unsloth_gkd_num_items" in inputs:
            num_items_in_batch = inputs.pop("_unsloth_gkd_num_items")
        else:
            self._unsloth_prepare_gkd_batch(model, inputs)
        loss = super().training_step(model, inputs, num_items_in_batch)
        return loss

    def _unsloth_prepare_gkd_batch(self, model, inputs):
        if random.random() <= self.lmbda:
            with (
                unwrap_model_for_generation(
                    model,
                    self.accelerator,
                    generation_kwargs=self.generation_kwargs,  # Override model.generation_config with generation_kwargs to fix transformers#42762
                ) as unwrapped_model
            ):
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels
        elif self.seq_kd:
            with (
                unwrap_model_for_generation(
                    self.teacher_model,
                    self.accelerator,
                    generation_kwargs=self.generation_kwargs,  # Override model.generation_config with generation_kwargs to fix transformers#42762
                ) as unwrapped_model
            ):
                new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config
                )
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels
    def get_batch_samples(self, epoch_iterator, num_batches, *args, **kwargs):
        batches, count = super().get_batch_samples(epoch_iterator, num_batches, *args, **kwargs)
        if self.lmbda > 0 or self.seq_kd:
            self._unsloth_gkd_pending_batches = (batches, args, kwargs)
        return batches, count
class UnslothGKDTrainer(_UnslothGKDTrainer):
    """
    Trainer for Generalized Knowledge Distillation (GKD) of language models.

    For details on GKD, see the paper: [On-Policy Distillation of Language Models: Learning from Self-Generated
    Mistakes](https://huggingface.co/papers/2306.13649).

    Args:
        model ([`~transformers.PreTrainedModel`] or `torch.nn.Module` or `str`, *optional*):
            Model to be trained, or the string identifier of the model to be instantiated from a pretrained model.
        teacher_model ([`~transformers.PreTrainedModel`] or `torch.nn.Module` or `str`, *optional*):
            Teacher model for knowledge distillation, or the string identifier of the model to be instantiated from a
            pretrained model.
        args ([`experimental.gkd.GKDConfig`], *optional*):
            Training arguments.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Data collator to batch samples from the dataset. It defaults to a
            [`experimental.utils.DataCollatorForChatML`] using the `processing_class`.
        train_dataset ([`~datasets.Dataset`], *optional*):
            Dataset for training.
        eval_dataset ([`~datasets.Dataset`] or `dict` of [`~datasets.Dataset`], *optional*):
            Dataset for evaluation.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.BaseImageProcessor`], [`~transformers.FeatureExtractionMixin`] or [`~transformers.ProcessorMixin`], *optional*):
           Class to process the data.
        compute_metrics (`Callable`, *optional*):
            Function to compute metrics at evaluation. Must take in an [`~transformers.EvalPrediction`] and return a
            dictionary string to float.
        callbacks (`list` of [`~transformers.TrainerCallback`], *optional*):
            Callbacks to use during training.
        optimizers (`tuple` of `torch.optim.Optimizer` and `torch.optim.lr_scheduler.LambdaLR`, *optional*, defaults to `(None, None)`):
            Tuple containing the optimizer and the learning rate scheduler to use for training.
        preprocess_logits_for_metrics (`Callable`, *optional*):
            Function to preprocess the logits before computing the metrics. Must take in the `logits` and `labels` and
            return the logits to be used for metrics computation.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration to use PEFT for training. If `None`, PEFT is not used. If provided, the `model` will be
            wrapped with the specified PEFT adapter.
        formatting_func (`Callable`, *optional*):
            Function to format the dataset. Must take in an example and return an example.
    
    """
    def __init__(
        self,
        model = None,
        teacher_model = None,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        processing_class = None,
        compute_metrics = None,
        callbacks = None,
        preprocess_logits_for_metrics = None,
        peft_config = None,
        formatting_func = None,
        **kwargs
    ):
        if args is None: args = UnslothGKDConfig()
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
        __tokenizer = processing_class if 'processing_class' in locals() else tokenizer
        from unsloth_zoo.vision_utils import UnslothVisionDataCollator
        if not isinstance(data_collator, UnslothVisionDataCollator):
            if isinstance(data_collator, DataCollatorForSeq2Seq) and 'labels' not in _unsloth_dataset_column_names(train_dataset):
                data_collator = TransformersDataCollatorForLanguageModeling(
                    __tokenizer,
                    mlm = False,
                    mlm_probability = 0.0,
                    pad_to_multiple_of = getattr(args, 'pad_to_multiple_of', None),
                )
            elif isinstance(data_collator, TransformersDataCollatorForLanguageModeling) and 'labels' in _unsloth_dataset_column_names(train_dataset):
                data_collator = DataCollatorForSeq2Seq(
                    __tokenizer,
                    pad_to_multiple_of = getattr(args, 'pad_to_multiple_of', None),
                )
        else:
            if hasattr(args, 'remove_unused_columns'): args.remove_unused_columns = False
            if hasattr(args, 'dataset_text_field'): args.dataset_text_field = ''
            if hasattr(args, 'dataset_kwargs'): args.dataset_kwargs = {'skip_prepare_dataset': True}
        if not isinstance(data_collator, UnslothVisionDataCollator):
            if not hasattr(__tokenizer, 'pad') and hasattr(__tokenizer, 'tokenizer'):
                if isinstance(data_collator, DataCollatorForSeq2Seq):
                    data_collator = DataCollatorForSeq2Seq(
                        __tokenizer.tokenizer,
                        pad_to_multiple_of = getattr(args, 'pad_to_multiple_of', None),
                    )
                elif isinstance(data_collator, TransformersDataCollatorForLanguageModeling):
                    data_collator = TransformersDataCollatorForLanguageModeling(
                        __tokenizer.tokenizer,
                        mlm = False,
                        mlm_probability = 0.0,
                        pad_to_multiple_of = getattr(args, 'pad_to_multiple_of', None),
                    )
        other_metrics = []
        
        from unsloth_zoo.logging_utils import PatchRLStatistics
        PatchRLStatistics('gkd_trainer', other_metrics)
        
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
            data_collator = data_collator,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            processing_class = processing_class,
            compute_metrics = compute_metrics,
            callbacks = callbacks,
            preprocess_logits_for_metrics = preprocess_logits_for_metrics,
            peft_config = peft_config,
            formatting_func = formatting_func,**kwargs)
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
        if getattr(self, 'liger_jsd_loss', None) is not None:
            self.liger_jsd_loss.weight_hard_loss = 0.0
            self.liger_jsd_loss.weight_soft_loss = 1.0
            self.liger_jsd_loss.temperature = 1.0
        
pass


if hasattr(logger, "addFilter"):
    import logging
    class HideLoggingMessage(logging.Filter):
        def __init__(self, text): self.text = text
        def filter(self, x): return not (self.text in x.getMessage())
    pass
    logger.addFilter(HideLoggingMessage("`use_cache=True`"))

