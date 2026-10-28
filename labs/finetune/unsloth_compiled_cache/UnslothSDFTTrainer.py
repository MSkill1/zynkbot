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
from trl.experimental.sdft.sdft_trainer import (Any, AutoProcessor, DataLoader, Dataset, DemonstrationTeacherContextBuilder, DistillationLogits, FSDP, GenerationConfig, IterableDataset, PEFTAdapterEMACallback, PeftConfig, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, RepeatSampler, SDFTConfig, SDFTTrainer, Sampler, SyncTeacherModelCallback, TrainerCallback, TrainingBatch, _BaseTrainer, add_tail_bucket, apply_importance_sampling_clipping, build_teacher_request_inputs, compute_divergence, compute_full_logit_self_distillation_loss, compute_sampled_token_self_distillation_loss, compute_topk_self_distillation_loss, create_model_from_path, dataclass, datasets, defaultdict, disable_dropout_in_model, get_config_model_id, identity, inspect, is_conversational, is_datasets_available, is_peft_available, is_peft_model, is_pure_lora_training, logger, nn, nullcontext, pad, partial, prepare_deepspeed, prepare_fsdp, seed_worker, selective_log_softmax, set_seed, split_tensor_dict, textwrap, torch, unwrap_model_for_generation, use_adapter, AutoProcessor, Dataset, DemonstrationTeacherContextBuilder, FSDP, GenerationConfig, IterableDataset, PeftConfig, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, SDFTConfig, SDFTTrainer, TrainerCallback, create_model_from_path, datasets, defaultdict, disable_dropout_in_model, get_config_model_id, identity, inspect, is_peft_available, is_peft_model, logger, nn, pad, set_seed, torch)


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
class UnslothSDFTConfig(SDFTConfig):
    """
    
    Configuration class for the [`experimental.sdft.SDFTTrainer`].

    Parameters:
        > Parameters that control the SDFT loss

        distillation_alpha (`float`, *optional*, defaults to `0.5`):
            Divergence interpolation coefficient for SDFT top-k logit distillation.
        distillation_mode (`Literal["sampled_token", "full_logits", "topk_logits"]`, *optional*, defaults to `"topk_logits"`):
            Distillation objective mode. SDFT defaults to top-k logit distillation.
        distillation_topk (`int`, *optional*, defaults to `100`):
            Number of top tokens used by the default SDFT top-k logit objective.
        distillation_is_clip (`float`, *optional*, defaults to `2.0`):
            Clipping coefficient for importance sampling in self-distillation. `None` disables clipping.
        distillation_add_tail (`bool`, *optional*, defaults to `False`):
            Whether to add a tail bucket for non-top-k probability mass.
        num_loss_tokens_to_skip (`int`, *optional*, defaults to `0`):
            Number of initial completion tokens to exclude from the distillation loss.

        > Parameters that control the teacher

        teacher_model_kind (`str`, *optional*, defaults to `"base"`):
            Semantic teacher choice for SDFT. `base` uses the initial student, `live` uses the current student, and
            `ema` uses an exponentially averaged teacher.
        teacher_update_rate (`float`, *optional*, defaults to `0.05`):
            EMA update rate used when `teacher_model_kind="ema"`. A value of `1.0` reduces the update to a hard
            overwrite, periodically resyncing the teacher to the current student weights.
        teacher_sync_steps (`int`, *optional*, defaults to `1`):
            Number of optimizer steps between teacher updates.

        > Parameters that control teacher-conditioned generation

        generate_from_teacher (`bool`, *optional*, defaults to `False`):
            Whether on-policy generation should use the teacher-conditioned prompt instead of the student prompt.
        teacher_prompt_template (`str`, *optional*, defaults to `"{prompt}\n\n{privileged_context}"`):
            Template used to combine the student prompt and privileged context into the teacher prompt.

        > Parameters that control the model

        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` argument
            of the `SDFTTrainer` is provided as a string. The `revision` value is also used when loading the processing
            class.
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to
            [`~transformers.AutoModelForCausalLM.from_pretrained`] and [`~transformers.AutoProcessor.from_pretrained`],
            for both the student and teacher.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the student and teacher models.

        > Parameters that control data preprocessing

        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the columns required by the trainer in the dataset. Keep this to `False` if you
            provide extra columns (such as `privileged_context`) that the trainer needs.
        max_prompt_length (`int`, *optional*, defaults to `512`):
            Maximum prompt length. Longer prompts are truncated from the left.
        shuffle_dataset (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training dataset.

        > Parameters that control generation

        num_generations (`int`, *optional*, defaults to `8`):
            Number of generations to sample. The effective batch size (num_processes * per_device_batch_size *
            gradient_accumulation_steps) must be evenly divisible by this value.
        num_generations_eval (`int`, *optional*):
            Number of generations to sample during evaluation. This allows using fewer generations during evaluation to
            save computation. If `None`, uses the value of `num_generations`.
        max_completion_length (`int`, *optional*, defaults to `256`):
            Maximum length of the generated completion.
        temperature (`float`, *optional*, defaults to `1.0`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        top_p (`float`, *optional*, defaults to `1.0`):
            Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1.0
            to consider all tokens.
        top_k (`int`, *optional*, defaults to `0`):
            Number of highest probability vocabulary tokens to keep for top-k-filtering. If `0`, top-k-filtering is
            disabled and all tokens are considered.
        min_p (`float`, *optional*):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range.
        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far.
            Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model to repeat
            tokens.
        cache_implementation (`str`, *optional*):
            Implementation of the cache method for faster generation when use_vllm is set to False.
        generation_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to `GenerationConfig` (if using transformers) or `SamplingParams` (if
            using vLLM) when sampling completions. This can be used to further customize the generation behavior, such
            as setting `suppress_tokens`, `num_beams`, etc. If it contains keys that conflict with the other generation
            parameters (like `min_p`, `top_p`, etc.), they will override them.
        chat_template_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the `apply_chat_template` function when generating completions.
        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible
            with vLLM generation.

        > Parameters that control generation acceleration powered by vLLM

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for generation
            instead of the default model.generate(). Requires `vllm` to be installed.
        use_teacher_server (`bool`, *optional*, defaults to `False`):
            Compute teacher logprobs from the running vLLM generation server instead of a local teacher forward. Only
            supported for `teacher_model_kind='live'` with `use_vllm=True` and `vllm_mode='server'`, and
            `distillation_mode` in {'sampled_token', 'topk_logits'} (the server returns the teacher's top-k logprobs,
            not the full vocabulary; `topk_logits` distills over the teacher's own top-k support).
        vllm_mode (`str`, *optional*, defaults to `"colocate"`):
            Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `'server'` or
            `'colocate'`. `'server'`: The trainer will send generation requests to a separate vLLM server. Make sure a
            vLLM server is running (start with `vllm serve`). `'colocate'`: vLLM will run in the same process and share
            the training GPUs. This avoids the need for a separate server but may cause resource contention with
            training.
        vllm_model_impl (`str`, *optional*, defaults to `"vllm"`):
            Model implementation to use for vLLM. Must be one of `transformers` or `vllm`. `transformers`: Use the
            `transformers` backend for model implementation. `vllm`: Use the `vllm` library for model implementation.
        vllm_enable_sleep_mode (`bool`, *optional*, defaults to `False`):
            Enable vLLM sleep mode to offload weights/cache during the optimizer step. Keeps GPU memory usage low, but
            waking the engine adds host–device transfer latency.
        vllm_server_base_url (`str`, *optional*):
            Base URL for the vLLM server (e.g., 'http://localhost:8000'). If provided, `vllm_server_host` and
            `vllm_server_port` are ignored.
        vllm_server_host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host of the vLLM server to connect to. Ignored if vllm_server_base_url is provided.
        vllm_server_port (`int`, *optional*, defaults to `8000`):
            Port of the vLLM server to connect to. Ignored if vllm_server_base_url is provided.
        vllm_group_port (`int`, *optional*, defaults to `51216`):
            Port number for the weight update group. This is used to communicate with the vLLM server. Unless the port
            is occupied, there is no need to change it.
        vllm_server_timeout (`float`, *optional*, defaults to `240.0`):
            Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up after the
            timeout, a `ConnectionError` is raised.
        vllm_tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set to
            `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when
            launching the vLLM server via the `--vllm_tensor_parallel_size` flag.
        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.3`):
            Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set to
            `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when
            launching the vLLM server via the `--vllm_gpu_memory_utilization` flag.
        vllm_max_model_length (`int`, *optional*):
            Context window for vLLM. Set it to at least the maximum prompt length in the dataset plus
            `max_completion_length`; if omitted, it is inferred from the model config.

        > Parameters that control the training

        num_iterations (`int`, *optional*, defaults to `1`):
            Number of iterations per batch (denoted as μ in the algorithm).
        generation_batch_size (`int`, *optional*):
            Batch size to use for generation. If `None`, it defaults to the effective training batch size:
            `per_device_train_batch_size * num_processes * steps_per_generation`.
        steps_per_generation (`int`, *optional*):
            Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`.
    
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
        per_device_eval_batch_size = 8,
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
        disable_dropout = True,
        max_prompt_length = 512,
        num_generations = 8,
        num_generations_eval = None,
        max_completion_length = 256,
        ds3_gather_for_generation = True,
        shuffle_dataset = True,
        generation_batch_size = None,
        steps_per_generation = None,
        temperature = 1.0,
        top_p = 1.0,
        top_k = 0,
        min_p = None,
        generation_kwargs = {},
        chat_template_kwargs = None,
        repetition_penalty = 1.0,
        cache_implementation = None,
        use_vllm = False,
        use_teacher_server = False,
        vllm_mode = 'colocate',
        vllm_model_impl = 'vllm',
        vllm_enable_sleep_mode = False,
        vllm_server_base_url = None,
        vllm_server_host = '0.0.0.0',
        vllm_server_port = 8000,
        vllm_group_port = 51216,
        vllm_server_timeout = 240.0,
        vllm_tensor_parallel_size = 1,
        vllm_gpu_memory_utilization = 0.3,
        vllm_max_model_length = None,
        num_iterations = 1,
        teacher_model_kind = 'base',
        teacher_update_rate = 0.05,
        teacher_sync_steps = 1,
        distillation_alpha = 0.5,
        distillation_mode = 'topk_logits',
        distillation_topk = 100,
        distillation_is_clip = 2.0,
        distillation_add_tail = False,
        generate_from_teacher = False,
        teacher_prompt_template = '{prompt}\n\n{privileged_context}',
        num_loss_tokens_to_skip = 0,
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
        if steps_per_generation is None and generation_batch_size is None:
            ga = gradient_accumulation_steps
            world_size = int(os.environ.get('WORLD_SIZE', '1'))
            if (ga * world_size * per_device_train_batch_size) % num_generations != 0:
                print('Unsloth: We now expect `per_device_train_batch_size` * `gradient_accumulation_steps` * `world_size` to be a multiple of `num_generations`.\nWe will change the batch size of ' + str(per_device_train_batch_size) + ' to the `num_generations` of ' + str(num_generations))
                per_device_train_batch_size = num_generations
        
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
            disable_dropout = disable_dropout,
            max_prompt_length = max_prompt_length,
            num_generations = num_generations,
            num_generations_eval = num_generations_eval,
            max_completion_length = max_completion_length,
            ds3_gather_for_generation = ds3_gather_for_generation,
            shuffle_dataset = shuffle_dataset,
            generation_batch_size = generation_batch_size,
            steps_per_generation = steps_per_generation,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            min_p = min_p,
            generation_kwargs = generation_kwargs,
            chat_template_kwargs = chat_template_kwargs,
            repetition_penalty = repetition_penalty,
            cache_implementation = cache_implementation,
            use_vllm = use_vllm,
            use_teacher_server = use_teacher_server,
            vllm_mode = vllm_mode,
            vllm_model_impl = vllm_model_impl,
            vllm_enable_sleep_mode = vllm_enable_sleep_mode,
            vllm_server_base_url = vllm_server_base_url,
            vllm_server_host = vllm_server_host,
            vllm_server_port = vllm_server_port,
            vllm_group_port = vllm_group_port,
            vllm_server_timeout = vllm_server_timeout,
            vllm_tensor_parallel_size = vllm_tensor_parallel_size,
            vllm_gpu_memory_utilization = vllm_gpu_memory_utilization,
            vllm_max_model_length = vllm_max_model_length,
            num_iterations = num_iterations,
            teacher_model_kind = teacher_model_kind,
            teacher_update_rate = teacher_update_rate,
            teacher_sync_steps = teacher_sync_steps,
            distillation_alpha = distillation_alpha,
            distillation_mode = distillation_mode,
            distillation_topk = distillation_topk,
            distillation_is_clip = distillation_is_clip,
            distillation_add_tail = distillation_add_tail,
            generate_from_teacher = generate_from_teacher,
            teacher_prompt_template = teacher_prompt_template,
            num_loss_tokens_to_skip = num_loss_tokens_to_skip,**kwargs)
        super().__init__(**_unsloth_filter_config_init_kwargs(SDFTConfig, _unsloth_config_arguments, mirrored_from = __class__))
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

class _UnslothSDFTTrainer(_BaseTrainer):
    """"""

    loss_is_scaled_for_ga = True

    _tag_names = ["trl", "sdft"]
    _name = "SDFT"
    config_cls = SDFTConfig
    # docstyle-ignore
    _paper = {
        "title": "Self-Distillation Enables Continual Learning",
        "id": "2601.19897",
        "citation": textwrap.dedent("""\
            @article{shenfeld2026selfdistillation,
                title        = {{Self-Distillation Enables Continual Learning}},
                author       = {Idan Shenfeld and Mehul Damani and Jonas H\\"ubotter and Pulkit Agrawal},
                year         = 2026,
                eprint       = {arXiv:2601.19897}
            }"""),
    }

    def __init__(
        self,
        model: str | PreTrainedModel | nn.Module,
        args: SDFTConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: "PeftConfig | None" = None,
    ):

        if hasattr(model, 'vllm_engine') and hasattr(args, 'use_vllm'):
            _unsloth_fsdp_cfg = getattr(args, 'fsdp_config', None)
            if getattr(args, 'fsdp', None) and isinstance(_unsloth_fsdp_cfg, dict) and str(_unsloth_fsdp_cfg.get('version', _unsloth_fsdp_cfg.get('fsdp_version', 2))) == '2':
                raise NotImplementedError('Unsloth: `fast_inference = True` (vLLM) is not supported with FSDP2.\nLoad the model with `fast_inference = False` to train with FSDP2, or launch without FSDP2 to use vLLM.')
            if (getattr(args, 'use_vllm', False) == False):
                args.use_vllm = True
            if getattr(args, 'top_k', -1) is None or getattr(args, 'top_k', -1) == 0:
                args.top_k = -1
        if isinstance(train_dataset, IterableDataset):
            raise NotImplementedError("Iterable datasets are not yet supported in SDFTTrainer.")
        if isinstance(eval_dataset, IterableDataset) or (
            isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
        ):
            raise NotImplementedError("Iterable eval datasets are not yet supported in SDFTTrainer.")

        self.num_loss_tokens_to_skip = args.num_loss_tokens_to_skip
        self.teacher_context_builder = DemonstrationTeacherContextBuilder(self)

        if train_dataset is None:
            raise ValueError("`train_dataset` is required")

        # PEFT initializes the adapter weights randomly, so set_seed must be done before creating the model to ensure
        # reproducibility.
        set_seed(args.seed)
        model_revision = None
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
            model_revision = model_init_kwargs.get("revision")
            model = create_model_from_path(model, **model_init_kwargs)
        elif args.model_init_kwargs is not None:
            logger.warning(
                "You passed `model_init_kwargs` to the self-distillation config, but `model` is already "
                "instantiated. The `model_init_kwargs` will be ignored."
            )

        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        if peft_config is None and getattr(model, "peft_config", None) is not None:
            logger.warning(
                "The provided self-distillation student model already contains a PEFT adapter. "
                "This setup is accepted but not directly supported. In particular, `teacher_model_kind='base'` "
                "may refer to the underlying base weights rather than the exact initially loaded student state "
                "including its adapter. For unambiguous teacher behavior, start from a merged/non-adapter model "
                "or manage separate adapters explicitly."
            )
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
                    "You passed a `PeftModel` instance together with a `peft_config`. Pass either a base "
                    "model with `peft_config`, or a pre-wrapped PEFT model."
                )
        if peft_config is not None or (is_peft_available() and getattr(model, "peft_config", None) is not None):
            pass

        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(
                get_config_model_id(model.config),
                revision=model_revision,
                truncation_side="left",
                padding_side="left",
                trust_remote_code=args.trust_remote_code,
            )

        if isinstance(processing_class, ProcessorMixin):
            self._tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            self._tokenizer = processing_class
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

        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.num_generations_eval = args.num_generations_eval or args.num_generations
        self.num_iterations = args.num_iterations
        self.shuffle_dataset = args.shuffle_dataset
        self.temperature = args.temperature
        self.generate_from_teacher = args.generate_from_teacher
        self.use_vllm = args.use_vllm
        self.chat_template_kwargs = args.chat_template_kwargs or {}
        self._step = 0
        self._buffered_inputs = None
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

        self.generation_kwargs = {
            "max_new_tokens": self.max_completion_length,
            "do_sample": True,
            "pad_token_id": self._tokenizer.pad_token_id,
            "bos_token_id": self._tokenizer.bos_token_id,
            "eos_token_id": self.eos_token_ids,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "repetition_penalty": args.repetition_penalty,
            "cache_implementation": args.cache_implementation,
        }
        if args.generation_kwargs is not None:
            self.generation_kwargs.update(args.generation_kwargs)
        self.generation_config = GenerationConfig(**self.generation_kwargs, disable_compile=True)

        if hasattr(model, "warnings_issued"):
            model.warnings_issued["estimate_tokens"] = True

        # Teacher logprobs from the running vLLM generation server [reuses the synced student weights] instead of a
        # local teacher forward. MVP: `live` teacher only — the generation server holds the current student weights.
        self.use_teacher_server = args.use_teacher_server
        if self.use_teacher_server:
            if not (args.use_vllm and args.vllm_mode == "server"):
                raise ValueError(
                    "`use_teacher_server=True` requires `use_vllm=True` and `vllm_mode='server'`: teacher logprobs are "
                    "served by the same vLLM server used for generation."
                )
            if args.teacher_model_kind != "live":
                raise ValueError(
                    "`use_teacher_server=True` only supports `teacher_model_kind='live'` (the generation server holds "
                    f"the current student weights), got {args.teacher_model_kind!r}."
                )
            if args.distillation_mode not in ("sampled_token", "topk_logits"):
                raise ValueError(
                    "`use_teacher_server=True` only supports `distillation_mode` in {'sampled_token', 'topk_logits'}, "
                    f"got {args.distillation_mode!r}. The server returns the teacher's top-k logprobs, not the full "
                    "vocabulary, so `full_logits` is unavailable. Note `topk_logits` distills over the teacher's own "
                    "top-k support (the server cannot score the student's top-k indices)."
                )
        super().__init__(
            model=model,
            args=args,
            data_collator=identity,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self._last_loaded_step = -1 if self.use_vllm else 0
        if self.use_vllm:
            from trl.generation.vllm_generation import VLLMGeneration

            self.vllm_generation = VLLMGeneration(
                model=self.model,
                accelerator=self.accelerator,
                processing_class=self.processing_class,
                mode=args.vllm_mode,
                server_base_url=args.vllm_server_base_url,
                server_host=args.vllm_server_host,
                server_port=args.vllm_server_port,
                group_port=args.vllm_group_port,
                server_timeout=args.vllm_server_timeout,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_length=args.vllm_max_model_length,
                max_num_seqs=args.per_device_train_batch_size
                * args.vllm_tensor_parallel_size
                * args.steps_per_generation,
                enable_sleep_mode=args.vllm_enable_sleep_mode,
                model_impl=args.vllm_model_impl,
                trust_remote_code=args.trust_remote_code,
                repetition_penalty=args.repetition_penalty,
                temperature=self.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                max_completion_length=self.max_completion_length,
                logprobs=None,
                generation_kwargs=args.generation_kwargs,
            )

        # Per-rank read-only client to the same generation server for teacher scoring [weights are synced there by
        # `VLLMGeneration`; scoring needs no weight-update communicator]. Mirrors the distillation trainer's
        # `teacher_client`.
        self.teacher_client = None
        if self.use_teacher_server:
            from trl.generation.vllm_client import VLLMClient

            base_url = args.vllm_server_base_url or f"http://{args.vllm_server_host}:{args.vllm_server_port}"
            self.teacher_client = VLLMClient(base_url=base_url, connection_timeout=args.vllm_server_timeout)

        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        self._setup_teacher_model()

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "privileged_context"]

    def _dispatch_self_distillation_callback(self, event_name: str, **payload) -> None:
        for callback in self.callback_handler.callbacks:
            callback_fn = getattr(callback, event_name, None)
            if callback_fn is not None:
                callback_fn(
                    args=self.args,
                    state=self.state,
                    control=self.control,
                    model=self.model,
                    processing_class=self.processing_class,
                    **payload,
                )

    def _setup_teacher_model(self) -> None:
        """Prepare teacher state according to the semantic teacher choice.

        Resolve `teacher_model_kind` × PEFT state into the effective teacher:

            - `"live"` (any model):
                Teacher is the student. No divergence, no callback.
            - `"base"` + PEFT model:
                Teacher reuses `self.model`; the base weights are recovered downstream by disabling the adapter via
                `use_adapter` during teacher forward.
            - `"base"` + non-PEFT model:
                Teacher is a frozen deepcopy of the initial student (falls through to the copy branch below).
            - `"ema"` + pure-LoRA training:
                Teacher reuses `self.model`; a dedicated `"teacher"` LoRA adapter is attached and updated by
                `PEFTAdapterEMACallback`. Teacher forward switches to that adapter downstream.
            - `"ema"` (otherwise):
                Teacher is a frozen deepcopy synchronized each step by `SyncTeacherModelCallback`.

        Must be called after `super().__init__` so that `self.callback_handler` is available.
        """

        teacher_model_kind = self.args.teacher_model_kind

        if teacher_model_kind == "live":
            self.teacher_model = self.model
            return

        if teacher_model_kind == "base" and is_peft_model(self.model):
            self.teacher_model = self.model
            return

        if self._use_peft_ema_teacher_adapter():
            # Must run after super().__init__ so self.callback_handler exists.
            self.add_callback(
                PEFTAdapterEMACallback(
                    model=self.model,
                    teacher_adapter_name="teacher",
                    update_rate=self.args.teacher_update_rate,
                    sync_steps=self.args.teacher_sync_steps,
                    accelerator=self.accelerator,
                )
            )
            self.teacher_model = self.model
            return

        if is_peft_model(self.model):
            raise ValueError(
                "`teacher_model_kind='ema'` with a non-pure-LoRA PEFT model is not supported: the separate EMA "
                "teacher cannot be parameter-matched to the PEFT student. Use pure-LoRA training, a non-PEFT model, "
                "or `teacher_model_kind` in {'live', 'base'}."
            )

        # Build the teacher from the model path (like the GRPO/DPO reference model) rather than deep-copying the
        # student: under ZeRO-3 the student params are already sharded, so a deep copy would clone empty shards.
        model_init_kwargs = self.args.model_init_kwargs or {}
        if self.args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
            model_init_kwargs["device_map"] = None
        model_init_kwargs.setdefault("trust_remote_code", self.args.trust_remote_code)
        self.teacher_model = create_model_from_path(get_config_model_id(self.model.config), **model_init_kwargs)
        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()
        if self.is_deepspeed_enabled:
            self.teacher_model = prepare_deepspeed(self.teacher_model, self.accelerator)
        elif self.is_fsdp_enabled:
            self.teacher_model = prepare_fsdp(self.teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(self.teacher_model, evaluation_mode=True)

        if teacher_model_kind == "ema":
            self.add_callback(SyncTeacherModelCallback(teacher_model=self.teacher_model, accelerator=self.accelerator))

    def _use_peft_ema_teacher_adapter(self) -> bool:
        return self.args.teacher_model_kind == "ema" and is_pure_lora_training(self.model, self.accelerator)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset=None) -> Sampler:
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations_eval,
            seed=self.args.seed,
        )

    def training_step(self, model, inputs, num_items_in_batch):
        output = super().training_step(model, inputs, num_items_in_batch)
        self._step += 1
        return output

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if not isinstance(inputs, dict):
            inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None

    def _prepare_inputs(self, generation_batch):
        """Return the per-step training batch, regenerating rollouts and buffering them for reuse in train mode.

        In train mode, rollouts are generated once every `steps_per_generation * num_iterations` steps and split into
        per-step slices reused until the next regeneration. In eval mode, every batch is freshly prepared.
        """
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                buffered_batch = self._prepare_training_batch(generation_batch)
                self._buffered_inputs = split_tensor_dict(buffered_batch, self.args.steps_per_generation)
                self._dispatch_self_distillation_callback(
                    "on_generation_batch_built",
                    generate_every=generate_every,
                    steps_per_generation=self.args.steps_per_generation,
                )
            return self._buffered_inputs[self._step % self.args.steps_per_generation]
        return self._prepare_training_batch(generation_batch)

    def _prepare_training_batch(self, inputs: list[dict[str, Any]]) -> TrainingBatch:
        """Sample student rollouts and construct teacher prompts"""
        batch = self.sample_rollouts(inputs)
        mode = "train" if self.model.training else "eval"
        self._record_completion_metrics(mode, batch)
        prompts = [example["prompt"] for example in inputs]
        privileged_contexts = [example.get("privileged_context") for example in inputs]
        teacher_batch = self.teacher_context_builder.build(
            prompts,
            privileged_contexts,
            batch["completion_ids"],
            batch["completion_mask"],
        )
        batch.update(
            {
                "teacher_input_ids": teacher_batch["teacher_input_ids"],
                "teacher_attention_mask": teacher_batch["teacher_attention_mask"],
            }
        )

        self._dispatch_self_distillation_callback(
            "on_self_distillation_batch_prepared",
            old_per_token_logps=batch.get("old_per_token_logps"),
            prompt_ids=batch["prompt_ids"],
            completion_ids=batch["completion_ids"],
            teacher_input_ids=batch["teacher_input_ids"],
            teacher_attention_mask=batch["teacher_attention_mask"],
        )
        return batch

    def sample_rollouts(self, inputs: list[dict[str, Any]]) -> TrainingBatch:
        """Generate completions for a batch of prompts and assemble the training batch."""
        prompts = [example["prompt"] for example in inputs]
        privileged_contexts = [example.get("privileged_context") for example in inputs]

        student_prompt_ids_list = self._tokenize_prompts(prompts)
        if self.generate_from_teacher:
            generation_prompts = self.teacher_context_builder.select_generation_prompts(prompts, privileged_contexts)
            generation_prompt_ids_list = self._tokenize_prompts(generation_prompts)
        else:
            generation_prompts = prompts
            generation_prompt_ids_list = student_prompt_ids_list

        self._dispatch_self_distillation_callback(
            "on_generation_prompts_selected",
            generation_prompts=generation_prompts,
            generation_prompt_text=None,
        )

        _, completion_ids_list = self._generate(generation_prompt_ids_list)
        device = self.accelerator.device
        prompt_ids = [torch.tensor(ids) for ids in student_prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self._tokenizer.pad_token_id, padding_side="left").to(device=device)
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left").to(device=device)

        completion_ids = [torch.tensor(ids) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self._tokenizer.pad_token_id, padding_side="right").to(
            device=device
        )
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right").to(device=device)

        old_per_token_logps = None
        if not self.generate_from_teacher:
            old_per_token_logps = self._compute_rollout_logps(
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                completion_ids=completion_ids,
                completion_mask=completion_mask,
            )
        batch: TrainingBatch = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "raw_completion_lengths": torch.tensor(
                [len(ids) for ids in completion_ids_list], device=device, dtype=torch.long
            ),
        }
        if old_per_token_logps is not None:
            batch["old_per_token_logps"] = old_per_token_logps
        return batch

    def _get_completion_ids_list(self, batch: TrainingBatch) -> list[list[int]]:
        raw_completion_lengths = batch["raw_completion_lengths"].detach().cpu().tolist()
        return [
            ids[:length].tolist()
            for ids, length in zip(batch["completion_ids"].detach().cpu(), raw_completion_lengths, strict=True)
        ]

    def _tokenize_prompts(self, prompts: list[Any]) -> list[list[int]]:
        if is_conversational({"prompt": prompts[0]}):
            tokenized = self.processing_class.apply_chat_template(
                conversation=prompts,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **self.chat_template_kwargs,
            )
            prompt_ids = tokenized["input_ids"]
        else:
            prompt_ids = self.processing_class(text=prompts)["input_ids"]
        if self.max_prompt_length is not None:
            prompt_ids = [ids[-self.max_prompt_length :] for ids in prompt_ids]
        return prompt_ids

    def _generate(self, prompt_ids: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
        if self.use_vllm:
            return self._generate_vllm(prompt_ids)
        return self._generate_transformers(prompt_ids)

    def _generate_vllm(self, prompt_ids: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
        if self.state.global_step != self._last_loaded_step:
            self.vllm_generation.sync_weights()
            self._last_loaded_step = self.state.global_step

        mode = "train" if self.model.training else "eval"
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        prompt_ids_out, completion_ids_list, _, _ = self.vllm_generation.generate(
            prompts=prompt_ids,
            images=None,
            num_generations=num_generations,
        )
        return prompt_ids_out, completion_ids_list

    def _generate_transformers(self, prompt_ids: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
        device = self.accelerator.device
        prompt_tensors = [torch.tensor(ids) for ids in prompt_ids]
        padded_ids = pad(prompt_tensors, padding_value=self._tokenizer.pad_token_id, padding_side="left").to(
            device=device
        )
        attention_mask = pad([torch.ones_like(t) for t in prompt_tensors], padding_value=0, padding_side="left").to(
            device=device
        )
        generate_inputs: dict[str, torch.Tensor | Any] = {"input_ids": padded_ids, "attention_mask": attention_mask}

        with (
            unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                generation_kwargs=self.generation_kwargs,
            ) as unwrapped_model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
        ):
            prompt_completion_ids = unwrapped_model.generate(
                **generate_inputs, generation_config=self.generation_config
            )

        prompt_length = generate_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        is_eos = torch.isin(completion_ids, torch.tensor(self.eos_token_ids, device=device))
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_idx = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()
        completion_ids_list = [
            c[m].tolist() for c, m in zip(completion_ids.cpu(), completion_mask.bool().cpu(), strict=True)
        ]
        return prompt_ids, completion_ids_list

    def _compute_rollout_logps(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        generate_every = self.args.steps_per_generation * self.num_iterations
        old_per_token_logps = None

        if self.args.gradient_accumulation_steps % generate_every != 0:
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)
            with torch.no_grad():
                logits = self._forward_logits(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                )
                old_per_token_logps = selective_log_softmax(logits, completion_ids)

        return old_per_token_logps

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The SDFTTrainer does not support returning outputs")

        if self.use_teacher_server:
            loss = self._compute_server_distillation_loss(model, inputs)
        else:
            distillation_logits = self._compute_teacher_student_logits(model, self.teacher_model, inputs)
            loss = self._compute_self_distillation_loss(model, inputs, distillation_logits)
        accumulation_scale = self.current_gradient_accumulation_steps if self.model.training else 1.0
        return loss / accumulation_scale

    def _compute_self_distillation_loss(
        self,
        model,
        inputs: TrainingBatch,
        distillation_logits: DistillationLogits,
    ) -> torch.Tensor:
        """Compute the per-token distillation loss and aggregate by normalizing over sequence length.

        Dispatches between three objectives based on `distillation_mode`:

            - `"topk_logits"`: top-k approximation of the divergence, optionally with a tail bucket for the remaining
              probability mass (`distillation_add_tail`).
            - `"full_logits"`: full-vocab divergence.
            - `"sampled_token"`: token-level (reverse-KL) distillation on sampled `completion_ids`.

        When `distillation_is_clip` is set and `old_per_token_logps` are available, the loss is corrected by a clipped
        importance-sampling ratio between the current student and the student at rollout time.
        """
        if distillation_logits.loss_mask.sum() == 0:
            mode = "train" if model.training else "eval"
            self._log_self_distillation_metric(mode, 0.0)
            # Keep the zero loss attached to the student graph so backward produces zero gradients instead of stopping.
            return distillation_logits.student_logits.sum() * 0.0

        if self.args.distillation_mode == "topk_logits":
            if self.args.distillation_topk is None:
                raise ValueError("`distillation_mode='topk_logits'` requires `distillation_topk` to be set.")
            per_token_loss = compute_topk_self_distillation_loss(
                distillation_logits.student_logits,
                distillation_logits.teacher_logits,
                distillation_topk=self.args.distillation_topk,
                distillation_alpha=self.args.distillation_alpha,
                distillation_add_tail=self.args.distillation_add_tail,
            )
        elif self.args.distillation_mode == "full_logits":
            per_token_loss = compute_full_logit_self_distillation_loss(
                distillation_logits.student_logits,
                distillation_logits.teacher_logits,
                distillation_alpha=self.args.distillation_alpha,
            )
        elif self.args.distillation_mode == "sampled_token":
            per_token_loss = compute_sampled_token_self_distillation_loss(
                distillation_logits.student_logits,
                distillation_logits.teacher_logits,
                distillation_logits.completion_ids,
                distillation_alpha=self.args.distillation_alpha,
            )
        else:
            raise ValueError(
                "distillation_mode must be one of: 'sampled_token', 'full_logits', 'topk_logits', "
                f"got {self.args.distillation_mode!r}"
            )

        old_per_token_logps = inputs.get("old_per_token_logps")
        if self.args.distillation_is_clip is not None and old_per_token_logps is not None:
            student_per_token_logps = selective_log_softmax(
                distillation_logits.student_logits,
                distillation_logits.completion_ids,
            )
            per_token_loss = apply_importance_sampling_clipping(
                per_token_loss,
                student_per_token_logps,
                old_per_token_logps,
                self.args.distillation_is_clip,
            )

        loss = (per_token_loss * distillation_logits.loss_mask).sum(-1) / distillation_logits.loss_mask.sum(-1).clamp(
            min=1.0
        )
        loss = loss.mean()

        mode = "train" if model.training else "eval"
        mean_distill_loss = (
            per_token_loss * distillation_logits.loss_mask
        ).sum() / distillation_logits.loss_mask.sum().clamp(min=1.0)
        self._log_self_distillation_metric(
            mode,
            self.accelerator.gather(mean_distill_loss).mean().item(),
        )
        return loss

    def _compute_server_distillation_loss(self, model, inputs: TrainingBatch) -> torch.Tensor:
        """Distillation loss with teacher logprobs served by the vLLM generation server (`teacher_model_kind='live'`).

        The student is forwarded locally (grad) for its logits; the teacher logprobs are fetched from the server (no
        local teacher forward). `sampled_token` distills the realized token (reverse KL); `topk_logits` distills over
        the teacher's own top-k support (the server cannot score the student's top-k indices).
        """
        # Buffered batches are reused across optimizer steps (`num_iterations > 1`), so the server weights can lag the
        # live student; re-sync before scoring.
        if self.state.global_step != self._last_loaded_step:
            self.vllm_generation.sync_weights()
            # Only the main process pushes weights to the server, so hold every rank here until the push completes;
            # otherwise non-main ranks could score against half-updated weights.
            self.accelerator.wait_for_everyone()
            self._last_loaded_step = self.state.global_step

        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        logits_to_keep = completion_ids.size(1)

        # SDFT skips the first few completion tokens in the distillation loss to suppress teacher-prompt artifacts.
        loss_mask = completion_mask
        if self.num_loss_tokens_to_skip > 0:
            token_positions = torch.arange(completion_mask.size(1), device=completion_mask.device).unsqueeze(0)
            loss_mask = completion_mask * (token_positions >= self.num_loss_tokens_to_skip).long()

        student_input_ids = torch.cat([inputs["prompt_ids"], completion_ids], dim=1)
        student_attention_mask = torch.cat([inputs["prompt_mask"], completion_mask], dim=1)
        student_logits = self._forward_logits(
            model=model,
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            logits_to_keep=logits_to_keep,
        )

        mode = "train" if model.training else "eval"
        teacher = self._get_teacher_token_logprobs_from_server(inputs, logits_to_keep)
        # Padding positions come back as `-inf`; the masked-out positions are neutralized to finite values below so the
        # divergence does not leak `+inf` through them (mirrors the distillation trainer).
        required = loss_mask.bool()
        student_per_token_logps = selective_log_softmax(student_logits, completion_ids)

        teacher_per_token_logps = teacher["actual_logprobs"]
        if (required & ~torch.isfinite(teacher_per_token_logps)).any():
            raise ValueError("Teacher server returned no logprob for a required completion token.")
        teacher_per_token_logps = torch.where(
            required, teacher_per_token_logps, torch.zeros_like(teacher_per_token_logps)
        )

        # Diagnostic for disagreement between local student scores and server teacher scores on realized tokens.
        # Sudden jumps can indicate stale server weights or numerical drift.
        abs_diff = ((student_per_token_logps.detach() - teacher_per_token_logps).abs() * loss_mask).sum() / (
            loss_mask.sum().clamp(min=1.0)
        )
        self._metrics[mode]["self_distillation/server_logprob_abs_diff"].append(
            self.accelerator.gather(abs_diff).mean().item()
        )

        if self.args.distillation_mode == "sampled_token":
            # Reverse-KL token-level objective, matching `compute_sampled_token_self_distillation_loss`.
            per_token_loss = (student_per_token_logps - teacher_per_token_logps).detach() * student_per_token_logps
        else:
            teacher_topk_logps = teacher["topk_logprobs"]
            teacher_topk_ids = teacher["topk_token_ids"]
            # Project the student onto the teacher's top-k support, then renormalize (or add a tail bucket) before the
            # divergence, matching `compute_topk_self_distillation_loss` on the teacher's support.
            keep = required.unsqueeze(-1)
            student_topk_logps = torch.gather(
                torch.log_softmax(student_logits, dim=-1), dim=-1, index=teacher_topk_ids
            )
            student_topk_logps = torch.where(keep, student_topk_logps, torch.zeros_like(student_topk_logps))
            teacher_topk_logps = torch.where(keep, teacher_topk_logps, torch.zeros_like(teacher_topk_logps))
            if self.args.distillation_add_tail:
                student_topk_logps = add_tail_bucket(student_topk_logps)
                teacher_topk_logps = add_tail_bucket(teacher_topk_logps)
            else:
                student_topk_logps = student_topk_logps - torch.logsumexp(student_topk_logps, dim=-1, keepdim=True)
                teacher_topk_logps = teacher_topk_logps - torch.logsumexp(teacher_topk_logps, dim=-1, keepdim=True)
            per_token_loss = compute_divergence(student_topk_logps, teacher_topk_logps, self.args.distillation_alpha)

        old_per_token_logps = inputs.get("old_per_token_logps")
        if self.args.distillation_is_clip is not None and old_per_token_logps is not None:
            per_token_loss = apply_importance_sampling_clipping(
                per_token_loss, student_per_token_logps, old_per_token_logps, self.args.distillation_is_clip
            )

        loss = (per_token_loss * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1.0)
        loss = loss.mean()
        mean_distill_loss = (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
        self._log_self_distillation_metric(mode, self.accelerator.gather(mean_distill_loss).mean().item())
        return loss

    def _get_teacher_token_logprobs_from_server(
        self, inputs: TrainingBatch, logits_to_keep: int
    ) -> dict[str, torch.Tensor]:
        """Per-token teacher logprobs from the vLLM server.

        Returns a dict with `actual_logprobs` (`(B, T)`, the realized-token logprob for reverse KL) and `topk_logprobs`
        / `topk_token_ids` (`(B, T, K)`, the teacher's top-k support for top-k logit distillation). Completions are
        right-padded, so server values fill the leading positions of each row and the trailing padding keeps the `-inf`
        / `0` sentinels (neutralized by the caller).
        """
        import numpy as np

        teacher_input_ids = inputs["teacher_input_ids"]
        teacher_attention_mask = inputs["teacher_attention_mask"]
        device = teacher_input_ids.device
        batch_size = teacher_input_ids.size(0)
        prompt_attention_mask = teacher_attention_mask[:, : teacher_attention_mask.size(1) - logits_to_keep]
        if self.args.distillation_mode == "topk_logits" and self.args.distillation_topk is None:
            raise ValueError("`distillation_mode='topk_logits'` requires `distillation_topk` to be set.")
        top_logprobs = self.args.distillation_topk if self.args.distillation_mode == "topk_logits" else 1

        sequences, prompt_lengths, _ = build_teacher_request_inputs(
            teacher_input_ids, teacher_attention_mask, prompt_attention_mask
        )
        result = self.teacher_client.get_sequence_logprobs(
            sequences=sequences,
            prompt_lengths=prompt_lengths,
            top_logprobs=top_logprobs,
            temperature=self.temperature,
        )

        actual = np.full((batch_size, logits_to_keep), float("-inf"), dtype=np.float32)
        topk = np.full((batch_size, logits_to_keep, top_logprobs), float("-inf"), dtype=np.float32)
        topk_ids = np.zeros((batch_size, logits_to_keep, top_logprobs), dtype=np.int64)
        for i in range(batch_size):
            seq_actual = result["actual_logprobs"][i]
            n = min(len(seq_actual), logits_to_keep)
            if n:
                actual[i, :n] = np.array(seq_actual, dtype=np.float32)[:n, 0]
                topk[i, :n] = np.array(result["logprobs"][i], dtype=np.float32)[:n]
                topk_ids[i, :n] = np.array(result["logprob_token_ids"][i], dtype=np.int64)[:n]
        return {
            "actual_logprobs": torch.from_numpy(actual).to(device),
            "topk_logprobs": torch.from_numpy(topk).to(device),
            "topk_token_ids": torch.from_numpy(topk_ids).to(device),
        }

    def _compute_teacher_student_logits(
        self,
        model,
        teacher_model,
        inputs: TrainingBatch,
    ) -> DistillationLogits:
        """Compute the per-token logits of the student and teacher over the completion tokens.

        The student is forwarded on its own input (original prompt plus the sampled completion) while the teacher is
        forwarded on its input (prompt, privileged context, and the same completion). Both sets of logits are aligned
        to the completion tokens so they can be compared position-by-position in the distillation loss.

        The teacher forward runs under `torch.no_grad()` and the context resolved by
        `_get_teacher_context_for_self_distillation`, which routes it to the correct weights.
        """
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        logits_to_keep = completion_ids.size(1)

        # SDFT skips the first few completion tokens in the distillation loss to suppress teacher-prompt artifacts.
        loss_mask = completion_mask
        if self.num_loss_tokens_to_skip > 0:
            token_positions = torch.arange(completion_mask.size(1), device=completion_mask.device).unsqueeze(0)
            loss_mask = completion_mask * (token_positions >= self.num_loss_tokens_to_skip).long()

        student_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        student_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        student_logits = self._forward_logits(
            model=model,
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            logits_to_keep=logits_to_keep,
        )

        with torch.no_grad(), self._get_teacher_context_for_self_distillation():
            teacher_logits = self._forward_logits(
                model=teacher_model,
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
                logits_to_keep=logits_to_keep,
            )

        return DistillationLogits(
            completion_ids=completion_ids,
            loss_mask=loss_mask,
            student_logits=student_logits,
            teacher_logits=teacher_logits,
        )

    def _forward_logits(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor:
        """Forward the model and return temperature-scaled logits aligned to the completion tokens."""
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if "logits_to_keep" in self.model_kwarg_keys:
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        logits = model(**model_inputs).logits
        logits = logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :]
        return logits / self.temperature

    def _get_teacher_context_for_self_distillation(self):
        """Return the context manager that routes the teacher forward to the correct weights.

        For non-PEFT models this is a no-op. For PEFT models:

            - `teacher_model_kind == "base"`: disable the student adapter so the teacher forward uses the base weights.
            - `teacher_model_kind == "ema"` under pure-LoRA training: switch to the `"teacher"` LoRA adapter.
            - otherwise: no-op; the teacher is a separate deepcopy.
        """
        teacher_model_kind = self.args.teacher_model_kind
        if not is_peft_model(self.model):
            return nullcontext()

        target_model = self.accelerator.unwrap_model(self.teacher_model)

        if teacher_model_kind == "base":
            return use_adapter(target_model, adapter_name=None)
        if teacher_model_kind == "ema" and self._use_peft_ema_teacher_adapter():
            return use_adapter(target_model, adapter_name="teacher")
        return nullcontext()

    def _record_completion_metrics(self, mode: str, batch: TrainingBatch) -> None:
        device = self.accelerator.device
        completion_ids_list = self._get_completion_ids_list(batch)
        agg_completion_lengths = self.accelerator.gather(batch["raw_completion_lengths"])
        # Fail clearly if the generation backend returned no completions (avoids a cryptic min() error below).
        if agg_completion_lengths.numel() == 0:
            raise RuntimeError(
                "No completions were generated. This usually means the generation backend failed to return any "
                "results; see the generation logs above for the underlying error."
            )
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        eos_and_pad = [*self.eos_token_ids, self._tokenizer.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

    def _log_self_distillation_metric(self, mode: str, value: float) -> None:
        metric_prefix = self._name.lower().replace(" ", "_")
        self._metrics[mode]["self_distillation/distillation_loss"].append(value)
        self._metrics[mode][f"{metric_prefix}/distillation_loss"].append(value)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {k: sum(v) / len(v) for k, v in self._metrics[mode].items() if v}
        if mode == "eval":
            metrics = {f"eval_{k}": v for k, v in metrics.items()}
        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()
class UnslothSDFTTrainer(_UnslothSDFTTrainer):
    """
    Trainer for SDFT-style on-policy self-distillation with explicit teacher prompts.
    """
    def __init__(
        self,
        model,
        args = None,
        train_dataset = None,
        eval_dataset = None,
        processing_class = None,
        callbacks = None,
        peft_config = None,
        **kwargs
    ):
        if args is None: args = UnslothSDFTConfig()
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
        PatchRLStatistics('sdft_trainer', other_metrics)
        
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
            args = args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            processing_class = processing_class,
            callbacks = callbacks,
            peft_config = peft_config,**kwargs)
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

