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
from trl.trainer.sft_trainer import (Any, AutoProcessor, BaseTunerLayer, BitsAndBytesConfig, Callable, DataCollator, DataCollatorForLanguageModeling, DataCollatorForVisionLanguageModeling, Dataset, DatasetDict, EvalPrediction, FLASH_ATTENTION_VARIANTS, IterableDataset, IterableDatasetDict, Path, PeftConfig, PeftModel, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, SFTConfig, SFTTrainer, TrainerCallback, TrainingArguments, Version, _BaseTrainer, _is_package_version_below, _tokenize, accelerate, add_fused_lm_head, apply_chat_template, clone_chat_template, contextlib, create_model_from_path, dataclass, defaultdict, enable_selective_activation_checkpointing, get_act_offloading_ctx_manager, get_config_model_id, get_peft_model, get_training_chat_template, has_generation_markers, is_chat_template_stop_token_trained, is_conversational, is_peft_available, is_peft_model, json, logger, os, pack_dataset, pad, prepare_multimodal_messages, set_seed, torch, transformers, warnings, Any, AutoProcessor, BaseTunerLayer, BitsAndBytesConfig, Callable, DataCollator, DataCollatorForLanguageModeling, DataCollatorForVisionLanguageModeling, Dataset, DatasetDict, EvalPrediction, FLASH_ATTENTION_VARIANTS, IterableDataset, IterableDatasetDict, PeftConfig, PeftModel, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, SFTConfig, SFTTrainer, TrainerCallback, TrainingArguments, Version, accelerate, add_fused_lm_head, clone_chat_template, contextlib, create_model_from_path, defaultdict, enable_selective_activation_checkpointing, get_act_offloading_ctx_manager, get_config_model_id, get_peft_model, get_training_chat_template, has_generation_markers, is_chat_template_stop_token_trained, is_conversational, is_peft_available, is_peft_model, logger, os, pad, set_seed, torch, transformers, warnings, Callable, DataCollator, DataCollatorForLanguageModeling, Dataset, IterableDataset, apply_chat_template, is_conversational, json, os, pack_dataset, pad, prepare_multimodal_messages, transformers, warnings, DataCollator, DataCollatorForLanguageModeling, Dataset, os, PeftModel, PreTrainedModel, is_peft_available, logger, os, torch, add_fused_lm_head, os, torch)


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
class UnslothSFTConfig(SFTConfig):
    """
    
    Configuration class for the [`SFTTrainer`].

    This class includes only the parameters that are specific to SFT training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model

        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`SFTTrainer`] is provided as a string. The `revision` value is also used when loading the
            processing class.
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to
            [`~transformers.AutoModelForCausalLM.from_pretrained`] and [`~transformers.AutoProcessor.from_pretrained`].
        router_aux_loss_coef (`float`, *optional*):
            Coefficient of the load-balancing auxiliary loss for Mixture-of-Experts (MoE) models, added to the training
            loss with this weight. When not set, the value declared by the model config is used. Set to `0.0` to
            disable it. Fails when used with a non-MoE model.
        chat_template_path (`str`, *optional*):
            If specified, sets the model's chat template. This can either be the path to a tokenizer (local directory
            or Hugging Face Hub model) or a direct path to a Jinja template file. When using a Jinja file, you must
            ensure that any special tokens referenced in the template are added to the tokenizer and that the model's
            embedding layer is resized accordingly.

        > Parameters that control the data preprocessing

        dataset_text_field (`str`, *optional*, defaults to `"text"`):
            Name of the column that contains text data in the dataset.
        dataset_kwargs (`dict[str, Any]`, *optional*):
            Dictionary of optional keyword arguments for the dataset preparation. The only supported key is
            `skip_prepare_dataset`. When the dataset contains images, `skip_prepare_dataset` is automatically treated
            as `True` regardless of the provided value, since preprocessing is done on the fly.
        dataset_num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.
        eos_token (`str`, *optional*):
            Token used to indicate the end of a turn or sequence. If `None`, it defaults to
            `processing_class.eos_token`.
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from the left or
            right depending on `truncation_mode`. If `None`, no truncation is applied. When packing is enabled, this
            value sets the sequence length.
        truncation_mode (`str`, *optional*, defaults to `"keep_start"`):
            Truncation mode to use when the sequence exceeds `max_length`. The only supported value is `"keep_start"`.
            The `"keep_end"` value is deprecated and will be removed in v2.0.0.
        shuffle_dataset (`bool`, *optional*, defaults to `False`):
            Whether to shuffle the dataset.
        packing (`bool`, *optional*, defaults to `False`):
            Whether to group multiple sequences into fixed-length blocks to improve computational efficiency and reduce
            padding. Uses `max_length` to define sequence length.
        packing_strategy (`str`, *optional*, defaults to `"bfd"`):
            Strategy for packing sequences. Can be `"bfd"` (best-fit decreasing, truncates overflow), `"bfd_split"`
            (best-fit decreasing, splits overflow sequences), or `"wrapped"` (aggressive, cuts mid-sequence).
        padding_free (`bool`, *optional*, defaults to `False`):
            Whether to perform forward passes without padding by flattening all sequences in the batch into a single
            continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this is only
            supported with the FlashAttention 2 or 3, which can efficiently handle the flattened batch structure. When
            packing is enabled with strategy `"bfd"`, padding-free is enabled, regardless of the value of this
            parameter.
        pad_to_multiple_of (`int`, *optional*):
            If set, the sequences will be padded to a multiple of this value.
        eval_packing (`bool`, *optional*):
            Whether to pack the eval dataset. If `None`, uses the same value as `packing`.

        > Parameters that control the training

        completion_only_loss (`bool`, *optional*):
            Whether to compute loss only on the completion part of the sequence. If set to `True`, loss is computed
            only on the completion, which is supported only for [prompt-completion](#prompt-completion) datasets. If
            `False`, loss is computed on the entire sequence. If `None` (default), the behavior depends on the dataset:
            loss is computed on the completion for [prompt-completion](#prompt-completion) datasets, and on the full
            sequence for [language modeling](#language-modeling) datasets.
        assistant_only_loss (`bool`, *optional*, defaults to `False`):
            Whether to compute loss only on the assistant part of the sequence. If set to `True`, loss is computed only
            on the assistant responses, which is supported only for [conversational](#conversational) datasets. If
            `False`, loss is computed on the entire sequence.
        loss_type (`str`, *optional*, defaults to `"nll"`):
            Type of loss to use. Possible values are:

            - `"nll"`: standard negative log-likelihood.
            - `"dft"`: Dynamic Fine-Tuning, as described in [this paper](https://huggingface.co/papers/2508.05629).
            - `"chunked_nll"`: deprecated alias of `"nll"`, which computes the loss the same way. It will be removed in
              v2.0.0.

        activation_offloading (`bool`, *optional*, defaults to `False`):
            Whether to offload the activations to the CPU.

        > Deprecated parameters

        pad_token:

            <Deprecated version="1.1.0">

            Parameter `pad_token` is deprecated and will be removed in version v2.0.0. Set `tokenizer.pad_token`
            directly and pass it as `processing_class` to the trainer instead.

            </Deprecated>

    > [!NOTE]
    > These parameters have default values different from [`~transformers.TrainingArguments`]:
    > - `logging_steps`: Defaults to `10` instead of `500`.
    > - `gradient_checkpointing`: Defaults to `True` instead of `False`.
    > - `bf16`: Defaults to `True` if `fp16` is not set, instead of `False`.
    > - `learning_rate`: Defaults to `2e-5` instead of `5e-5`.
    
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
    context_parallel_size : Optional[int] = field(
        default = 1,
        metadata = {'help': 'Ranks per context parallel group. 1 disables it.'},
    )
    context_parallel_backend : Optional[str] = field(
        default = 'ring',
        metadata = {'help': 'Context parallel attention backend: ring or ulysses (keeps packing).'},
    )
    context_parallel_rotate_method : Optional[str] = field(
        default = 'allgather',
        metadata = {'help': 'Ring KV rotation: allgather or alltoall.'},
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
        vllm_sampling_params = None,
        unsloth_num_chunks = -1,
        unsloth_logit_chunk_multiplier = None,
        unsloth_grpo_mini_batch = None,
        max_seq_length = None,context_parallel_size = 1,context_parallel_backend = 'ring',context_parallel_rotate_method = 'allgather',
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
            dataset_num_proc = _unsloth_get_dataset_num_proc(dataset_num_proc, serial_as_none = False)
        if os.environ.get('UNSLOTH_ENABLE_FLEX_ATTENTION', '0') == '1':
            from unsloth_zoo.flex_attention import HAS_FLEX_ATTENTION
            if HAS_FLEX_ATTENTION and pad_to_multiple_of is None:
                from unsloth_zoo.flex_attention import FLEX_ATTENTION_BLOCK_SIZE
                pad_to_multiple_of = FLEX_ATTENTION_BLOCK_SIZE
        
        if eval_steps is not None and eval_strategy != 'steps':
            print(f'Unsloth: `eval_steps = {eval_steps}` is ignored because `eval_strategy` is {getattr(eval_strategy, "value", eval_strategy)!r}. Set `eval_strategy = "steps"` to evaluate every `eval_steps` steps.')
        
        
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
            pad_token = pad_token,**kwargs)
        super().__init__(**_unsloth_filter_config_init_kwargs(SFTConfig, _unsloth_config_arguments, mirrored_from = __class__))
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
        self.context_parallel_size = context_parallel_size
        self.context_parallel_backend = context_parallel_backend
        self.context_parallel_rotate_method = context_parallel_rotate_method
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

class _UnslothSFTTrainer(_BaseTrainer):
    """"""

    _tag_names = ["trl", "sft"]
    _name = "SFT"

    def __init__(
        self,
        model: "str | PreTrainedModel | PeftModel",
        args: SFTConfig | TrainingArguments | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset
        | IterableDataset
        | DatasetDict
        | IterableDatasetDict
        | dict[str, Dataset | IterableDataset]
        | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        compute_loss_func: Callable | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        quantization_config: "BitsAndBytesConfig | None" = None,
        peft_config: "PeftConfig | None" = None,
        formatting_func: Callable[[dict], str] | None = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = SFTConfig(f"{model_name}-SFT")
        elif isinstance(args, TrainingArguments) and not isinstance(args, SFTConfig):
            dict_args = args.to_dict()
            dict_args["hub_token"] = args.hub_token  # to_dict hides the hub_token
            if Version(transformers.__version__) < Version("5.0.0"):
                dict_args.pop("push_to_hub_token", None)
            args = UnslothSFTConfig(**dict_args)

        if train_dataset is None:
            raise ValueError("`train_dataset` is required")
        elif isinstance(train_dataset, IterableDataset):
            # IterableDataset requires dispatch_batches=False because Accelerate's dispatch mode may try to concatenate
            # batches from multiple processes, leading to mismatch errors.
            if args.accelerator_config.dispatch_batches is True:
                logger.warning(
                    "You are using an `IterableDataset` for training with `dispatch_batches=True`. `dispatch_batches` "
                    "is forced to `False` when using an `IterableDataset`. To remove this warning, unset "
                    "`dispatch_batches` in `SFTConfig` or set it to `False`."
                )
            args.accelerator_config.dispatch_batches = False
        elif not isinstance(train_dataset, (Dataset, list, tuple)):
            raise TypeError(
                f"`train_dataset` must be a `Dataset` or `IterableDataset`, got `{type(train_dataset).__name__}`."
            )

        # Model
        # PEFT initializes the adapter weights randomly, so set_seed must be done before creating the model to ensure
        # reproducibility.
        set_seed(args.seed)
        if isinstance(model, str):
            model_init_kwargs = dict(args.model_init_kwargs or {})  # copy to avoid mutating model_init_kwargs
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
            model_revision = None
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `SFTConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )
            if quantization_config is not None:
                logger.warning(
                    "You passed `quantization_config` to the trainer, but your model is already instantiated. The "
                    "`quantization_config` will be ignored."
                )
        # Non-quantized models do not have the `is_loaded_in_{8,4}bit` attributes, whereas quantized models do
        _is_quantized_model = getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(
                get_config_model_id(model.config), revision=model_revision, trust_remote_code=args.trust_remote_code
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

        if args.eos_token is not None:
            if args.eos_token not in self._tokenizer.get_vocab():
                raise ValueError(
                    f"The specified `eos_token` ('{args.eos_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `eos_token` exists "
                    "in the vocabulary before using it as an EOS token."
                )
            self._tokenizer.eos_token = args.eos_token
            # The model must agree with the tokenizer on the eos token from construction, so mirror it onto the model
            # configs. The generation config may hold several eos tokens, any of which halts generation, so the new
            # one is added to the existing ones instead of replacing them.
            model.config.get_text_config().eos_token_id = self._tokenizer.eos_token_id
            eos_token_ids = model.generation_config.eos_token_id
            if eos_token_ids is None:
                eos_token_ids = []
            elif isinstance(eos_token_ids, int):
                eos_token_ids = [eos_token_ids]
            if self._tokenizer.eos_token_id not in eos_token_ids:
                model.generation_config.eos_token_id = [self._tokenizer.eos_token_id, *eos_token_ids]

        if args.chat_template_path is not None:
            if os.path.isfile(args.chat_template_path) and args.chat_template_path.endswith((".jinja", ".j2")):
                with open(args.chat_template_path, encoding="utf-8") as chat_template_file:
                    processing_class.chat_template = chat_template_file.read()
                added_tokens = []
            else:
                model, processing_class, added_tokens = clone_chat_template(
                    model, processing_class, args.chat_template_path
                )
        else:
            added_tokens = []

        # Vision dataset detection
        dataset_sample = next(iter(train_dataset))
        self._is_vision_dataset = "image" in dataset_sample or "images" in dataset_sample
        # Unsloth: override _is_vlm for VLM models that pass a bare tokenizer
        if not self._is_vlm and self._is_vision_dataset:
            _m = model
            if hasattr(_m, "model"): _m = _m.model
            if hasattr(getattr(_m, "config", None), "vision_config") or\
               _m.__class__.__name__.endswith("ForConditionalGeneration"):
                self._is_vlm = True
        if self._is_vision_dataset and not self._is_vlm:
            raise ValueError(
                "The dataset appears to be vision-related (contains 'image' or 'images' keys), but the provided "
                "model does not seem to be a vision-language model. Please check your model and dataset."
            )

        if self._is_vision_dataset and args.packing:
            raise ValueError(
                "Packing is not supported for vision datasets. Please set `packing=False` in the SFTConfig."
            )
        if self._is_vision_dataset and args.padding_free:
            raise ValueError(
                "Padding-free training is yet not supported for vision datasets. Please set `padding_free=False` in "
                "the `SFTConfig`."
            )
        if (
            self._is_vision_dataset
            and args.assistant_only_loss
            and Version(transformers.__version__) < Version("5.18.0.dev0")
        ):
            raise ValueError(
                "Assistant-only loss for vision datasets requires transformers>=5.18.0. Please upgrade transformers "
                "or set `assistant_only_loss=False` in the `SFTConfig`."
            )
        if self._is_vision_dataset and args.max_length is not None and args.truncation_mode == "keep_end":
            raise ValueError(
                "truncation_mode='keep_end' is not supported for vision datasets. Image tokens reside "
                "inside the prompt portion of the sequence; depending on the example, keep_end may silently "
                "drop them, causing pixel_values to be forwarded to the model with no corresponding visual "
                "tokens in input_ids. Use truncation_mode='keep_start' (the default) or set max_length=None."
            )

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
            if added_tokens:
                # Ensure that the added tokens are trainable
                if peft_config.trainable_token_indices is None:
                    peft_config.trainable_token_indices = {"embed_tokens": added_tokens}
                elif "embed_tokens" not in peft_config.trainable_token_indices:
                    peft_config.trainable_token_indices["embed_tokens"] = added_tokens
                else:
                    peft_config.trainable_token_indices["embed_tokens"].extend(added_tokens)
                # Ensure that the lm_head is trainable
                if peft_config.modules_to_save is None or "lm_head" not in peft_config.modules_to_save:
                    logger.warning(
                        "Cloning chat template added new tokens to the tokenizer, but 'lm_head' is not in PEFT's "
                        "`modules_to_save`. As a result, the model may not learn to generate outputs with these new "
                        "tokens, leading to degraded generation quality. To fix this, add "
                        "`modules_to_save=['lm_head']` to your PEFT configuration."
                    )

                    if peft_config.modules_to_save is None:
                        peft_config.modules_to_save = ["lm_head"]
                    else:
                        peft_config.modules_to_save.append("lm_head")
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
            if (args.gradient_checkpointing_kwargs or {}).get("selective"):
                raise ValueError(
                    "`gradient_checkpointing_kwargs={'selective': True}` is not supported with PEFT + DeepSpeed "
                    "ZeRO-3, which requires reentrant gradient checkpointing while SAC requires non-reentrant "
                    "checkpointing."
                )
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

        # Data collator
        # BFD packing requires padding-free mode; otherwise, the collator outputs padded attention masks, causing
        # FlashAttention to ignore position_ids and recompute them incorrectly from the padded attention mask.
        self.padding_free = args.padding_free or (args.packing and args.packing_strategy in {"bfd", "bfd_split"})
        # A hub kernel can be requested with a revision and/or a kernel name [`repo_id@revision:kernel_name`], while
        # the variants above are bare repo ids, so compare against the repo id alone.
        attn_implementation = model.config._attn_implementation.split("@")[0].split(":")[0]
        use_flash_attention = attn_implementation in FLASH_ATTENTION_VARIANTS
        if self.padding_free:
            if data_collator is not None:
                raise ValueError("Passing a custom data collator is not supported when using padding-free.")
            if args.packing and args.packing_strategy == "wrapped":
                logger.warning(
                    "You are passing `padding_free=True` with the 'wrapped' packing strategy, which is not "
                    "recommended. Please refer to the documentation to understand why this is not recommended."
                )
            if not use_flash_attention:
                logger.warning(
                    "Padding-free training is enabled, but the attention implementation is not set to a supported "
                    "Flash Attention variant. Padding-free training flattens batches into a single sequence, and only "
                    "the following implementations are known to reliably support this: "
                    f"{', '.join(sorted(FLASH_ATTENTION_VARIANTS))}. Using other implementations may lead to "
                    "unexpected behavior. To ensure compatibility, set `attn_implementation` in the model "
                    "configuration to one of these supported options or verify that your attention mechanism can "
                    "handle flattened sequences."
                )
        if args.assistant_only_loss and not is_conversational(dataset_sample):
            raise ValueError(
                "You set `assistant_only_loss=True`, but the dataset is not conversational. This option is only "
                "supported for conversational datasets."
            )

        # When assistant_only_loss is enabled, swap in a training chat template with {% generation %} markers
        # if the current template doesn't already have them.
        if args.assistant_only_loss and not has_generation_markers(processing_class.chat_template):
            try:
                self.chat_template = get_training_chat_template(processing_class)
            except ValueError:
                if getattr(self, '_is_vision_dataset', False) or (getattr(args, 'dataset_kwargs', None) or {}).get('skip_prepare_dataset'): raise
                self.chat_template = None
                self._unsloth_assistant_mask_fallback = True
                print('Unsloth: TRL has no training chat template for this tokenizer, so assistant_only_loss masks non-assistant tokens with train_on_responses_only markers.')
        else:
            self.chat_template = None

        # A template can define generation markers and still attribute the assistant's end-of-turn token to the next
        # message, leaving it out of the assistant mask so the model is never trained to stop.
        if args.assistant_only_loss and not getattr(self, '_unsloth_assistant_mask_fallback', False) and not is_chat_template_stop_token_trained(
            processing_class, chat_template=self.chat_template
        ):
            logger.warning(
                "The chat template does not include the assistant turn's end-of-turn token in the loss mask; "
                "the model may not learn to stop. The training loss still looks healthy, so this usually only "
                "surfaces at inference. Either set `assistant_only_loss=False` to train on the full sequence, "
                "or edit the chat template so the end-of-turn token falls inside "
                "`{% generation %}...{% endgeneration %}`."
            )

        # Decide whether to use completion-only loss: if not specified, then it is set to True if the dataset format
        # is prompt-completion, and False if the dataset format is language modeling.
        if args.completion_only_loss is None:
            self.completion_only_loss = "prompt" in dataset_sample and "completion" in dataset_sample
        else:
            self.completion_only_loss = args.completion_only_loss

        if data_collator is None:
            # Get the pad token: if not provided, use the one from the processing class or the eos token
            # if the processing class does not have a pad token.
            pad_token = args.pad_token or self._tokenizer.pad_token or self._tokenizer.eos_token
            if pad_token not in self._tokenizer.get_vocab():
                raise ValueError(
                    f"The specified `pad_token` ('{pad_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `pad_token` exists "
                    "in the vocabulary before using it as a padding token."
                )
            self._tokenizer.pad_token = pad_token
            # The model must agree with the tokenizer on the pad token from construction, so mirror it onto the model
            # configs.
            model.config.get_text_config().pad_token_id = self._tokenizer.pad_token_id
            model.generation_config.pad_token_id = self._tokenizer.pad_token_id
            if self._is_vision_dataset:
                data_collator = DataCollatorForVisionLanguageModeling(
                    processor=processing_class,
                    max_length=args.max_length,
                    completion_only_loss=self.completion_only_loss,
                    pad_to_multiple_of=args.pad_to_multiple_of,
                    dataset_text_field=args.dataset_text_field,
                    assistant_only_loss=args.assistant_only_loss,
                    chat_template=self.chat_template,
                )
            else:
                data_collator = DataCollatorForLanguageModeling(
                    pad_token_id=self._tokenizer.pad_token_id,
                    padding_free=self.padding_free,
                    pad_to_multiple_of=args.pad_to_multiple_of,
                )

        if args.packing and args.packing_strategy in {"bfd", "bfd_split"} and not use_flash_attention:
            logger.warning(
                "You are using packing, but the attention implementation is not set to a supported Flash Attention "
                "variant. Packing gathers multiple samples into a single sequence, and only the following "
                f"implementations are known to reliably support this: {', '.join(sorted(FLASH_ATTENTION_VARIANTS))}. "
                "Using other implementations may lead to cross-contamination between samples. To avoid this, either "
                "disable packing by setting `packing=False`, or set `attn_implementation` in the model configuration "
                "to one of these supported options."
            )

        # Dataset
        if self.padding_free and not args.packing and args.max_length is not None and not self._is_vision_dataset:
            raise ValueError(
                "When `padding_free=True` without packing, `max_length` is not enforced. Either enable packing "
                "(e.g., `packing=True, packing_strategy='bfd'`), provide already truncated inputs, or set "
                "`max_length=None`."
            )
        # Skip dataset preparation if `skip_prepare_dataset=True` in `dataset_kwargs`, or if it's a VLM, where
        # preprocessing [e.g., image-to-pixel conversion] is too costly and done on the fly instead.
        self._skip_prepare_dataset = (
            args.dataset_kwargs is not None
            and args.dataset_kwargs.get("skip_prepare_dataset", False)
            or self._is_vision_dataset
        )
        # Kept on the instance so that `evaluate` can preprocess freshly-passed eval datasets the same way.
        self._formatting_func = formatting_func
        eval_datasets = (
            eval_dataset if isinstance(eval_dataset, dict) else {"eval": eval_dataset} if eval_dataset else {}
        )
        self._reject_skip_prepare_without_labels({"train": train_dataset, **eval_datasets}, data_collator)
        if not self._skip_prepare_dataset:
            if self.completion_only_loss and formatting_func:
                raise ValueError(
                    "A formatting function was provided while `completion_only_loss=True`, which is incompatible. "
                    "Using a formatter converts the dataset to a language modeling type, conflicting with "
                    "completion-only loss. To resolve this, apply your formatting function before passing the "
                    "dataset, or disable `completion_only_loss` in `SFTConfig`."
                )
            self._unsloth_model_ref = model
            train_dataset = self._prepare_dataset(
                train_dataset, processing_class, args, args.packing, formatting_func, "train"
            )
            if eval_dataset is not None:
                packing = args.packing if args.eval_packing is None else args.eval_packing
                if isinstance(eval_dataset, dict):
                    eval_dataset = {
                        key: self._prepare_dataset(dataset, processing_class, args, packing, formatting_func, key)
                        for key, dataset in eval_dataset.items()
                    }
                else:
                    eval_dataset = self._prepare_dataset(
                        eval_dataset, processing_class, args, packing, formatting_func, "eval"
                    )

        # Loss function
        if args.loss_type not in ("nll", "dft"):
            raise ValueError(f"Invalid `loss_type` {args.loss_type} passed. Supported values are 'nll' and 'dft'.")
        if args.loss_type == "dft" and compute_loss_func is not None:
            raise ValueError(
                "You passed a `compute_loss_func` together with `loss_type='dft'` to the `SFTTrainer`. When using "
                "`loss_type='dft'`, the loss function is internally set to the DFT loss, so passing a "
                "`compute_loss_func` is not allowed."
            )
        if compute_loss_func is not None:
            warnings.warn(
                "`compute_loss_func` is deprecated and will be removed in v2.0.0. Subclass `SFTTrainer` and override "
                "`compute_loss` instead. Expect slower training meanwhile: the loss needs the full logits, so each "
                "step runs an extra forward pass.",
                FutureWarning,
                stacklevel=2,
            )
        if args.label_smoothing_factor > 0 and args.loss_type == "nll":
            warnings.warn(
                "`label_smoothing_factor` is deprecated in `SFTTrainer` and will be removed in v2.0.0. Label smoothing "
                "needs the full logits, so the loss is computed from a forward pass that does not use the fused LM "
                "head. Expect slower training meanwhile: each step runs an extra forward pass.",
                FutureWarning,
                stacklevel=2,
            )
        if is_peft_model(model) and isinstance(model.get_output_embeddings(), BaseTunerLayer):
            # The log-probabilities are computed by multiplying the hidden states by `lm_head.weight` directly, so an
            # adapter on the LM head would be ignored and never trained.
            raise ValueError(
                "Applying a PEFT adapter to `lm_head` is not supported: the log-probabilities are computed from "
                "`lm_head.weight` directly, so the adapter would never be trained. Remove `'lm_head'` from your "
                "`target_modules`."
            )

        # Transformers explicitly set use_reentrant=True in the past to silence a PyTorch warning, but the default was
        # never updated once PyTorch switched to recommending use_reentrant=False. Until that change lands upstream
        # [see https://github.com/huggingface/transformers/pull/43203] and is released [most likely in 5.0.0], we
        # default to the recommended non-reentrant behavior here, while preserving any user-provided value.
        if args.gradient_checkpointing and Version(transformers.__version__) < Version("5.0.0"):
            args.gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
            args.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

        # `selective` is a TRL-only key: when set, the SAC wrapper strips it before it reaches
        # `torch.utils.checkpoint`, otherwise it is dropped here [a missing key already means SAC is off]
        if (args.gradient_checkpointing_kwargs or {}).get("selective"):
            if not args.gradient_checkpointing:
                raise ValueError(
                    "`gradient_checkpointing_kwargs={'selective': True}` requires `gradient_checkpointing=True`."
                )
            if (args.gradient_checkpointing_kwargs or {}).get("use_reentrant"):
                raise ValueError(
                    "`gradient_checkpointing_kwargs={'selective': True}` requires non-reentrant gradient "
                    "checkpointing. Set `use_reentrant` to `False`, or leave it unset."
                )
            # Wrap before `super[].__init__[]` so the context function is set when the Trainer enables checkpointing.
            enable_selective_activation_checkpointing(model)
        elif args.gradient_checkpointing_kwargs:
            args.gradient_checkpointing_kwargs.pop("selective", None)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
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

        # Liger's fused linear cross-entropy replaces `model.forward` when training starts, which would drop the fused
        # LM head, so only its layer kernels are applied
        if args.use_liger_kernel:
            warnings.warn(
                "`use_liger_kernel=True` is deprecated and will be removed in v2.0.0. Use the Hub kernels instead, "
                'with `model_init_kwargs={"use_kernels": True}`.',
                FutureWarning,
                stacklevel=2,
            )
            liger_kernel_config = args.liger_kernel_config or {}
            if liger_kernel_config.get("fused_linear_cross_entropy"):
                raise ValueError(
                    '`liger_kernel_config={"fused_linear_cross_entropy": True}` is not supported: it replaces the '
                    "model's forward, which this trainer patches with a fused LM head."
                )
            args.liger_kernel_config = {**liger_kernel_config, "fused_linear_cross_entropy": False}

        # Compute the per-token log-probabilities in chunks, without materializing the full logits
        # `is_top1` feeds the `mean_token_accuracy` metric
        add_fused_lm_head(
            self.model.get_base_model() if is_peft_model(self.model) else self.model,
            outputs=("log_probs", "entropy", "is_top1"),
        )

        # Context parallelism can only express full causal attention: the per-layer attention mask is dropped
        # and replaced by `is_causal=True`. Packed sequences rely on a block-diagonal mask to keep documents
        # from attending to each other, so a packed batch would silently train with documents attending across
        # their boundaries. Read the parallelism config off the accelerator rather than off `args`: when
        # context parallelism is configured through an accelerate YAML, `args.parallelism_config` stays None.
        if not _is_package_version_below("accelerate", "1.10.1"):
            parallelism_config = self.accelerator.parallelism_config
            if (
                parallelism_config is not None
                and parallelism_config.cp_enabled
                and (args.packing or args.eval_packing)
            ):
                raise ValueError(
                    "Packing is not compatible with context parallelism (`cp_size > 1`). Packing relies on a "
                    "block-diagonal attention mask to keep packed documents separate, and context parallelism "
                    "drops that mask, so documents would attend across their boundaries. Set `packing=False` "
                    "and `eval_packing=False`, or disable context parallelism."
                )

        # Sequence parallelism [Ulysses/ALST] shards batches along the sequence dimension and requires
        # `position_ids` in every batch to preserve each token's global position. Sequence parallelism was added in
        # accelerate 1.12.0.
        if (
            Version(accelerate.__version__) >= Version("1.12.0")
            and self.accelerator.parallelism_config is not None
            and self.accelerator.parallelism_config.sp_enabled
            and isinstance(self.data_collator, DataCollatorForLanguageModeling)
        ):
            self.data_collator.return_position_ids = True

        # Under DeepSpeed sequence parallelism [Ulysses/ALST], `Trainer` reduces the model's own loss across ranks,
        # which is the fused LM head's negative log-likelihood
        if (
            Version(accelerate.__version__) >= Version("1.12.0")
            and self.accelerator.parallelism_config is not None
            and self.accelerator.parallelism_config.sp_backend == "deepspeed"
            and self.accelerator.parallelism_config.sp_enabled
            and (args.loss_type == "dft" or self.compute_loss_func is not None or args.label_smoothing_factor > 0)
        ):
            raise ValueError(
                "`loss_type='dft'`, `compute_loss_func` and `label_smoothing_factor` are not supported with DeepSpeed "
                "sequence parallelism (Ulysses/ALST): the loss is the model's own negative log-likelihood, reduced "
                "across ranks by `Trainer`."
            )

        # Initialize activation offloading context
        if self.args.activation_offloading:
            self.maybe_activation_offload_context = get_act_offloading_ctx_manager(model=self.model)
        else:
            self.maybe_activation_offload_context = contextlib.nullcontext()

        # MoE load-balancing auxiliary loss. `output_router_logits` in the config means the model returns its router
        # logits; `router_aux_loss_coef` is the architecture's own coefficient, 0.0 when the config declares none.
        text_config = model.config.get_text_config()
        coef = args.router_aux_loss_coef
        self.router_aux_loss_coef = getattr(text_config, "router_aux_loss_coef", 0.0) if coef is None else coef
        if coef and not hasattr(text_config, "output_router_logits"):
            raise ValueError(
                f"`router_aux_loss_coef` is set to {coef} but {type(model).__name__} is not a Mixture-of-Experts model "
                f"that returns its router logits, so there is no auxiliary loss to weight."
            )
        self.aux_loss_enabled = hasattr(text_config, "output_router_logits") and self.router_aux_loss_coef != 0.0
        if hasattr(text_config, "router_aux_loss_coef"):
            # The native and chunked forwards add the aux loss from the model config, so keep the config in sync with
            # the coef: enable it [and propagate the coef] when non-zero, disable it otherwise. This overrides any
            # `output_router_logits` the model was loaded with, so `router_aux_loss_coef=0.0` reliably turns it off.
            text_config.output_router_logits = self.aux_loss_enabled
            text_config.router_aux_loss_coef = self.router_aux_loss_coef

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

        # Tensor parallel ranks are given the same batch, so a token count gathered across all processes repeats
        # every token once per rank in the group. Context and sequence parallelism shard the batch before the loss is
        # computed, so they need no such correction. `ParallelismConfig.tp_size` requires accelerate 1.10.0.
        if Version(accelerate.__version__) >= Version("1.10.0") and self.accelerator.parallelism_config is not None:
            self._tp_size = self.accelerator.parallelism_config.tp_size
        else:
            self._tp_size = 1

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class,
        args,
        packing: bool,
        formatting_func: Optional[Callable[[dict], str]],
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        import inspect as _inspect
        try:
            _unsloth_pack_has_strategy = "strategy" in _inspect.signature(pack_dataset).parameters
        except Exception:
            _unsloth_pack_has_strategy = True
        _unsloth_wrapped_packing = packing and (
            getattr(args, "packing_strategy", None) in ("wrapped", "bfd_split", "bfd-requeue")
            or not _unsloth_pack_has_strategy
        )
        # All Unsloth Zoo code licensed under LGPLv3
        try:
            if isinstance(dataset, ConstantLengthDataset): return dataset
        except:
            pass
    
        map_kwargs = {}
        use_desc = isinstance(dataset, Dataset)
        is_vlm = hasattr(processing_class, "tokenizer")
        tokenizer = processing_class
        if is_vlm: tokenizer = processing_class.tokenizer
    
        # Detect whether the model's module needs token_type_ids when training
        import sys as _sys
        _needs_token_type_ids = False
        # Split to avoid compiler substring match on masking_utils names
        _ccm = 'create_' + 'causal_mask_mapping'
        _model = getattr(self, '_unsloth_model_ref', None) or getattr(self, 'model', None)
        if _model is not None:
            for _m in (_model, getattr(_model, 'model', None)):
                if _m is None: continue
                _mod = _sys.modules.get(type(_m).__module__)
                if _mod is not None and hasattr(_mod, _ccm):
                    _needs_token_type_ids = True
                    break
    
        if not _needs_token_type_ids:
            # Fallback: model not yet available, check processor class MRO
            for _base in type(processing_class).__mro__:
                _base_mod = getattr(_base, '__module__', '')
                if 'transformers.models.' in _base_mod:
                    _modeling_mod = _base_mod.replace('.processing_', '.modeling_')
                    _mod = _sys.modules.get(_modeling_mod)
                    if _mod is not None and hasattr(_mod, _ccm):
                        _needs_token_type_ids = True
                        break
        if _needs_token_type_ids and hasattr(args, 'remove_unused_columns'):
            args.remove_unused_columns = False
    
        # Get max length
        max_seq_length = getattr(args, "max_length", 0) or 0
        if max_seq_length == 0: max_seq_length = getattr(args, "max_seq_length", 0)
        if max_seq_length == 0: max_seq_length = getattr(self, "max_seq_length", 0)
        if max_seq_length == 0: max_seq_length = getattr(self, "max_seq", 0)
        if max_seq_length == 0: raise RuntimeError("Unsloth: max_seq_length is 0! Please specify one!")
        dataset_text_field = getattr(args, "dataset_text_field", "text")
        do_truncation = max_seq_length != 0
        do_formatting_func = False
        do_tokenize = True
        do_prompt_completion = False
        do_messages = False
    
        # Get correct column names
        column_names = set(next(iter(dataset)).keys())
        used_column_names = ["input_ids"]
        if "attention_mask" in column_names:
            used_column_names.append("attention_mask")
        if _needs_token_type_ids:
            used_column_names.append("token_type_ids")
    
        # Skip tokenization if already tokenized; just set the data collator
        from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
        if "labels" in column_names:
            # Most likely forgot data collator!
            if is_vlm and not hasattr(tokenizer, "pad"):
                raise RuntimeError(f"Unsloth: {processing_class.__class__} does not have .pad!")
            self.data_collator = DataCollatorForSeq2Seq(tokenizer)
            used_column_names.append("labels")
            do_tokenize = False
        elif "input_ids" in column_names:
            if is_vlm and not hasattr(tokenizer, "pad"):
                raise RuntimeError(f"Unsloth: {processing_class.__class__} does not have .pad!")
            mask_columns = [x for x in ("completion_mask", "assistant_masks") if x in column_names]
            if mask_columns:
                used_column_names += mask_columns
            else:
                self.data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)
            do_tokenize = False
        elif "prompt" in column_names and "completion" in column_names:
            # Prompt/completion dataset (used with completion_only_loss).
            # TRL's __init__ already set self.data_collator for completion_only_loss
            # before calling us -- we must NOT overwrite it here.
            do_prompt_completion = True
            used_column_names.append("completion_mask")
        elif (dataset_text_field not in column_names or getattr(args, "assistant_only_loss", False)) and\
            formatting_func is None and "messages" in column_names and\
            isinstance(_first_messages := next(iter(dataset))["messages"], list) and\
            len(_first_messages) != 0 and isinstance(_first_messages[0], dict) and\
            "role" in _first_messages[0]:
            # Text columns carry no assistant masks, so assistant_only_loss prefers "messages" even if text exists.
            do_messages = True
        elif dataset_text_field not in column_names:
            do_formatting_func = True
            if formatting_func is None:
                raise RuntimeError("Unsloth: You must specify a `formatting_func`")
        pass
        if do_tokenize and not do_messages and not do_prompt_completion and getattr(args, "assistant_only_loss", False):
            raise ValueError(
                "Unsloth: assistant_only_loss needs a conversational `messages` dataset and no formatting_func. "
                "Or use `train_on_responses_only` on your formatted text."
            )
    
        if do_tokenize:
            if do_formatting_func:
                test_text = formatting_func(next(iter(dataset)))
                if not isinstance(test_text, list):
                    raise ValueError(
                        "Unsloth: The `formatting_func` should return a list of processed strings."
                    )
                test_text = test_text[0]
            elif do_prompt_completion:
                _first_ex = next(iter(dataset))
                try:
                    from trl import is_conversational as _sft_is_conversational
                except ImportError:
                    def _sft_is_conversational(example):
                        for key in ("prompt", "completion", "messages"):
                            val = example.get(key)
                            if isinstance(val, list) and val and isinstance(val[0], dict):
                                if "role" in val[0] and "content" in val[0]:
                                    return True
                        return False
                _is_conv = _sft_is_conversational(_first_ex)
                if not _is_conv:
                    if getattr(args, "assistant_only_loss", False):
                        raise ValueError(
                            "Unsloth: assistant_only_loss needs conversational prompt/completion messages, not plain strings."
                        )
                    test_text = _first_ex["prompt"]
                else:
                    test_text = None  # chat template handles BOS
            elif do_messages:
                test_text = None  # chat template handles BOS
            else:
                # No [0] on a str: that is the first CHARACTER, so startswith(bos_token)
                # below could never match. Only unwrap when the field really is a list.
                test_text = next(iter(dataset))[dataset_text_field]
                if isinstance(test_text, (list, tuple)):
                    test_text = test_text[0] if len(test_text) != 0 else None
    
            chat_template = getattr(processing_class, 'chat_template', '')
            if chat_template == '' and is_vlm:
                chat_template = getattr(tokenizer, 'chat_template', '')
            if chat_template is None:
                chat_template = ''
    
            # Detect double BOS so we can drop the duplicate
            add_special_tokens = True
            bos_token_1 = getattr(processing_class, 'bos_token', None)
            bos_token_2 = getattr(tokenizer, 'bos_token', None)
            bos_token = bos_token_1 or bos_token_2
    
            if bos_token is not None:
                if (test_text is not None and test_text.startswith(bos_token)) or bos_token in chat_template:
                    add_special_tokens = False
                    print("Unsloth: We found double BOS tokens - we shall remove one automatically.")
            pass
    
            def _tokenize(example):
                return tokenizer(
                    example[dataset_text_field] if not do_formatting_func else formatting_func(example),
                    truncation = do_truncation and not _unsloth_wrapped_packing,
                    max_length = max_seq_length,
                    return_token_type_ids = _needs_token_type_ids,
                    add_special_tokens = add_special_tokens,
                )
            pass
    
            # Worker count via the shared policy; spelled exactly as the edit unsloth applies here, so unsloth sees it as done.
            if not isinstance(dataset, IterableDataset):
                try:
                    from unsloth_zoo.dataset_num_proc import get_dataset_num_proc as _unsloth_get_dataset_num_proc
                except ImportError:
                    from unsloth.dataset_num_proc import get_dataset_num_proc as _unsloth_get_dataset_num_proc
                map_kwargs["num_proc"] = _unsloth_get_dataset_num_proc(
                    getattr(args, "dataset_num_proc", None)
                )
            else:
                map_kwargs["batch_size"] = _iterable_batch_size(dataset)
    
            if do_messages or do_prompt_completion:
                assistant_only_loss = bool(getattr(args, "assistant_only_loss", False))
                training_chat_template = getattr(self, "chat_template", None)
                # No {% generation %} markers (or no TRL training template): assistant masks would be all zeros,
                # so mask with train_on_responses_only markers instead.
                import re as _re
                def _template_text(template):
                    # Named template sets are picked per call: check them all.
                    if isinstance(template, dict):
                        return "\n".join(v for v in template.values() if isinstance(v, str))
                    return template or ""
                marker_masks = assistant_only_loss and (
                    getattr(self, "_unsloth_assistant_mask_fallback", False) or
                    _re.search(r"\{%-?\s*generation\s*-?%\}", _template_text(training_chat_template or chat_template)) is None
                )
                response_labels = None
                if marker_masks:
                    from unsloth_zoo.dataset_utils import train_on_responses_only as _response_labels
                    marker_kwargs = {}
                    # Harmony (gpt-oss): auto-detection would keep only the final channel, not analysis/tool calls.
                    if "<|channel|>" in _template_text(getattr(tokenizer, "chat_template", None)) and\
                        not hasattr(tokenizer, "_unsloth_output_part"):
                        marker_kwargs = dict(
                            instruction_part = "<|start|>user<|message|>",
                            response_part = "<|start|>assistant",
                        )
                    response_labels = _response_labels(
                        None, tokenizer = processing_class, return_function = True, **marker_kwargs,
                    )
                if assistant_only_loss:
                    used_column_names.append("assistant_masks")
    
            if do_prompt_completion:
                _eos_token = getattr(tokenizer, 'eos_token', None)
    
                def _tokenize_pc(example):
                    if _is_conv:
                        template_kwargs = dict(example.get("chat_template_kwargs") or {})
                        if training_chat_template is not None:
                            template_kwargs["chat_template"] = training_chat_template
                        prompt_ids = processing_class.apply_chat_template(
                            example["prompt"], tokenize=True,
                            add_generation_prompt=True, return_dict=False,
                            tools=example.get("tools"),
                            **template_kwargs,
                        )
                        if prompt_ids and isinstance(prompt_ids[0], list):
                            prompt_ids = prompt_ids[0]
                        pc_processed = processing_class.apply_chat_template(
                            example["prompt"] + example["completion"],
                            return_dict=True, tokenize=True,
                            return_assistant_tokens_mask=assistant_only_loss and not marker_masks,
                            tools=example.get("tools"),
                            **template_kwargs,
                        )
                        if isinstance(pc_processed.get("input_ids", [None])[0], list):
                            pc_processed = {k: v[0] for k, v in pc_processed.items()}
                        pc_ids = pc_processed["input_ids"]
                    else:
                        _completion = example["completion"]
                        if _eos_token and not _completion.endswith(_eos_token):
                            _completion = _completion + _eos_token
                        prompt_ids = tokenizer(
                            example["prompt"], add_special_tokens=add_special_tokens,
                        )["input_ids"]
                        pc_ids = tokenizer(
                            example["prompt"] + _completion,
                            add_special_tokens=add_special_tokens,
                        )["input_ids"]
                    n_prompt = min(len(prompt_ids), len(pc_ids))
                    completion_mask = [0] * n_prompt + [1] * (len(pc_ids) - n_prompt)
                    result = {"input_ids": pc_ids, "completion_mask": completion_mask}
                    if assistant_only_loss:
                        if marker_masks:
                            labels = response_labels({"input_ids": [pc_ids]})["labels"][0]
                            result["assistant_masks"] = [int(x != -100) for x in labels]
                        else:
                            result["assistant_masks"] = list(pc_processed["assistant_masks"])
                    if do_truncation and not _unsloth_wrapped_packing and max_seq_length > 0:
                        result = {k: v[:max_seq_length] for k, v in result.items()}
                    if _needs_token_type_ids:
                        result["token_type_ids"] = [0] * len(result["input_ids"])
                    return result
    
                if use_desc:
                    map_kwargs["desc"] = 'Unsloth: Tokenizing ["prompt"+"completion"]'
                import warnings as _w
                try:
                    from unsloth_zoo.dataset_num_proc import map_failure_diagnostics as _unsloth_map_diagnostics
                except ImportError:
                    from unsloth.dataset_num_proc import map_failure_diagnostics as _unsloth_map_diagnostics
                with _w.catch_warnings(), _unsloth_map_diagnostics(map_kwargs.get("num_proc")):
                    _w.filterwarnings("ignore", message=".*couldn't be hashed properly.*")
                    dataset = dataset.map(
                        _tokenize_pc, batched=False,
                        remove_columns=list(column_names), **map_kwargs,
                    )
                if assistant_only_loss:
                    # Prompt-history assistant turns are no target while completion_mask applies.
                    pc_completion_only = getattr(self, "completion_only_loss", None)
                    if pc_completion_only is None:
                        pc_completion_only = getattr(args, "completion_only_loss", None)
                    if pc_completion_only is None or pc_completion_only:
                        dataset = dataset.filter(
                            lambda row: any(a and c for a, c in zip(row["assistant_masks"], row["completion_mask"]))
                        )
                    else:
                        dataset = dataset.filter(lambda row: any(row["assistant_masks"]))
            elif do_messages:
                truncate_messages = do_truncation and isinstance(max_seq_length, int) and max_seq_length > 0 and\
                    not locals().get("_unsloth_wrapped_packing", False)
                message_cut = slice(-max_seq_length, None) if truncate_messages and\
                    getattr(args, "truncation_mode", "keep_start") == "keep_end" else slice(None, max_seq_length)
                def _tokenize_messages(example):
                    template_kwargs = dict(example.get("chat_template_kwargs") or {})
                    if training_chat_template is not None:
                        template_kwargs["chat_template"] = training_chat_template
                    # Arrow fills missing keys with None, which `is defined` template checks then index.
                    messages = [
                        {k: v for k, v in message.items() if v is not None}
                        if isinstance(message, dict) else message
                        for message in example["messages"]
                    ]
                    if is_vlm:
                        # Processor templates index content parts, as TRL's prepare_multimodal_messages builds them.
                        messages = [
                            {**message, "content": [{"type": "text", "text": message["content"]}]}
                            if isinstance(message, dict) and isinstance(message.get("content"), str) else message
                            for message in messages
                        ]
                    tools = example.get("tools")
                    if isinstance(tools, str):
                        import json as _json
                        tools = _json.loads(tools)
                    processed = processing_class.apply_chat_template(
                        messages, tools = tools,
                        tokenize = True, return_dict = True,
                        return_assistant_tokens_mask = assistant_only_loss and not marker_masks,
                        **template_kwargs,
                    )
                    input_ids = processed["input_ids"]
                    if input_ids and isinstance(input_ids[0], list):
                        input_ids = input_ids[0]
                    result = {"input_ids": list(input_ids)}
                    if assistant_only_loss:
                        if marker_masks:
                            labels = response_labels({"input_ids": [result["input_ids"]]})["labels"][0]
                            result["assistant_masks"] = [int(x != -100) for x in labels]
                        else:
                            masks = processed["assistant_masks"]
                            if masks and isinstance(masks[0], list):
                                masks = masks[0]
                            result["assistant_masks"] = list(masks)
                    if truncate_messages:
                        result = {k: v[message_cut] for k, v in result.items()}
                    if _needs_token_type_ids:
                        result["token_type_ids"] = [0] * len(result["input_ids"])
                    return result
    
                if use_desc: map_kwargs["desc"] = 'Unsloth: Tokenizing ["messages"]'
                import warnings as _w
                with _w.catch_warnings():
                    _w.filterwarnings("ignore", message = ".*couldn't be hashed properly.*")
                    dataset = dataset.map(
                        _tokenize_messages, batched=False,
                        remove_columns=list(column_names), **map_kwargs,
                    )
                if assistant_only_loss:
                    # Truncation can drop every assistant token; such a row has no target.
                    dataset = dataset.filter(lambda row: any(row["assistant_masks"]))
            else:
                if use_desc: map_kwargs["desc"] = f'Unsloth: Tokenizing ["{dataset_text_field}"]'
                import warnings as _w
                try:
                    from unsloth_zoo.dataset_num_proc import map_failure_diagnostics as _unsloth_map_diagnostics
                except ImportError:
                    from unsloth.dataset_num_proc import map_failure_diagnostics as _unsloth_map_diagnostics
                with _w.catch_warnings(), _unsloth_map_diagnostics(map_kwargs.get("num_proc")):
                    _w.filterwarnings("ignore", message=".*couldn't be hashed properly.*")
                    dataset = dataset.map(_tokenize, batched = True, remove_columns = list(column_names), **map_kwargs)
    
            # VLMs need .pad; switch the data collator
            # transformers' collator rebuilds labels from input_ids, dropping assistant masks: keep TRL's for messages.
            if is_vlm and not hasattr(processing_class, "pad") and not do_prompt_completion and not do_messages:
                data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)
                self.data_collator = data_collator
            pass
        pass
    
        # Imported here: this function's source is copied into the compiled trainer module.
        from unsloth_zoo.dataset_utils import trl_collator_applies_masks
        column_names = set(next(iter(dataset)).keys())
        if "labels" not in column_names and not trl_collator_applies_masks():
            completion_only_loss = getattr(self, "completion_only_loss", None)
            if completion_only_loss is None:
                completion_only_loss = getattr(args, "completion_only_loss", None)
            if completion_only_loss is None:
                completion_only_loss = do_prompt_completion
            mask_columns = [x for x in ("completion_mask", "assistant_masks") if x in column_names]
            if not completion_only_loss and "completion_mask" in mask_columns:
                mask_columns.remove("completion_mask")
            if mask_columns:
                def _build_labels(example):
                    masks = [example[x] for x in mask_columns]
                    return {"labels": [t if all(m) else -100 for t, *m in zip(example["input_ids"], *masks)]}
                label_kwargs = {}
                if isinstance(dataset, Dataset):
                    label_kwargs["num_proc"] = map_kwargs.get("num_proc", None)
                    label_kwargs["desc"] = f"Unsloth: Building labels for {dataset_name} dataset"
                dataset = dataset.map(_build_labels, remove_columns = mask_columns, **label_kwargs)
                used_column_names = [x for x in used_column_names if x not in mask_columns] + ["labels"]
        pass
        if packing:
            # Use TRL's pack_dataset if available
            try:
                # A presence probe, not a use: TRL exports pack_dataset only on some
                # versions, and the bare name is what the except below is for.
                pack_dataset  # noqa: F821
            except:
                print("Unsloth: Hugging Face's packing is currently buggy - we're disabling it for now!")
                return dataset
    
            if max_seq_length == 0:
                raise ValueError("When packing is enabled, `max_seq_length` can't be `None`.")
    
            if use_desc: map_kwargs["desc"] = f"Unsloth: Packing {dataset_name} dataset"
            _pack_kwargs = {"map_kwargs": map_kwargs}
            if _unsloth_pack_has_strategy:
                _pack_kwargs["strategy"] = getattr(args, "packing_strategy", "bfd")
            dataset = pack_dataset(
                dataset.select_columns(used_column_names),
                max_seq_length,
                **_pack_kwargs,
            )
        pass
        return dataset
    
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids",
        # "attention_mask" and "labels"). Dataset preparation also produces a "seq_lengths" column (for packing /
        # padding-free), so we override the default signature columns to keep it alongside the model inputs.
        if self._signature_columns is None:
            if self._is_vision_dataset:
                self._signature_columns = ["messages", "prompt", "completion", "image", "images"]
            else:
                self._signature_columns = ["input_ids", "labels", "seq_lengths"]

    def _reject_skip_prepare_without_labels(self, datasets: dict[str, Dataset], data_collator) -> None:
        # This guard may look defensive, but it covers a behavior change introduced when label building moved from
        # the collator to dataset preparation: the collator used to consume the mask columns directly, so a
        # skipped-preparation dataset carrying masks trained correctly. Now labels are built during preparation, which
        # is skipped here, and the collator ignores the mask columns. Without a "labels" column, such a dataset would
        # silently optimize the loss over the full sequence, so we fail loudly instead. Checked both at init and in
        # `evaluate`, since a dataset passed directly to `evaluate` also skips preparation.
        if not (
            self._skip_prepare_dataset
            and not self._is_vision_dataset
            and isinstance(data_collator, DataCollatorForLanguageModeling)
        ):
            return
        for name, dataset in datasets.items():
            cols = _unsloth_dataset_column_names(dataset)
            if "labels" not in cols and ("completion_mask" in cols or "assistant_masks" in cols):
                raise ValueError(
                    f"The {name} dataset has mask columns but no 'labels', and `skip_prepare_dataset=True` skips "
                    "label building, so it would train on the full sequence. Add a 'labels' column (-100 for "
                    "non-loss tokens) or drop `skip_prepare_dataset`."
                )

    def evaluate(
        self,
        eval_dataset: Dataset
        | IterableDataset
        | DatasetDict
        | IterableDatasetDict
        | dict[str, Dataset | IterableDataset]
        | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        # When a dataset is passed directly to `evaluate` (e.g. a held-out test set), preprocess it the same way
        # `__init__` does, so that `evaluate` accepts the same dataset types as the trainer (language modeling,
        # prompt-completion, etc.). `_prepare_dataset` is idempotent: it skips datasets that are already tokenized. A
        # `str` selects a dataset that was already prepared at init time, so it's left untouched.
        if not self._skip_prepare_dataset and eval_dataset is not None and not isinstance(eval_dataset, str):
            packing = self.args.packing if self.args.eval_packing is None else self.args.eval_packing
            if isinstance(eval_dataset, dict):
                eval_dataset = {
                    key: self._prepare_dataset(
                        dataset, self.processing_class, self.args, packing, self._formatting_func, key
                    )
                    for key, dataset in eval_dataset.items()
                }
            else:
                eval_dataset = self._prepare_dataset(
                    eval_dataset, self.processing_class, self.args, packing, self._formatting_func, "eval"
                )
        eval_datasets = (
            eval_dataset
            if isinstance(eval_dataset, dict)
            else {"eval": eval_dataset}
            if eval_dataset is not None and not isinstance(eval_dataset, str)
            else {}
        )
        self._reject_skip_prepare_without_labels(eval_datasets, self.data_collator)
        return super().evaluate(
            eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs = False,
        num_items_in_batch = None,
    ):
        # TRL >= 1.15 computes loss_type="dft" only in the compute_loss this replaces (else it trains as NLL):
        # run it on TRL's fused LM head. Older TRL routes DFT through compute_loss_func.
        if (
            getattr(self.args, "loss_type", None) == "dft"
            and self.compute_loss_func is None
            and getattr(self, "label_smoother", None) is None
        ):
            import trl.trainer.utils as _trl_utils
            if hasattr(_trl_utils, "add_fused_lm_head"):
                # TRL's fused head projects with lm_head.weight directly, skipping QAT fake quantization.
                try:
                    head = model.get_output_embeddings()
                except Exception:
                    head = None
                try:
                    from unsloth.models.llama import lm_head_needs_forward as _needs_forward
                except Exception:
                    _needs_forward = lambda module: False
                if head is not None and (
                    _needs_forward(head)
                    or any(
                        getattr(m, "activation_fake_quantizer", None) is not None
                        or getattr(m, "weight_fake_quantizer", None) is not None
                        for m in head.modules()
                    )
                ):
                    raise NotImplementedError(
                        "Unsloth: loss_type = 'dft' runs on TRL's fused LM head, which skips the QAT fake "
                        "quantization of lm_head. Leave lm_head out of QAT or use loss_type = 'nll'."
                    )
                inputs = {**inputs, "use_cache": False}
                outputs = model(**inputs, fused_lm_head = True)
                if getattr(outputs, "log_probs", None) is None:
                    raise RuntimeError(
                        "Unsloth: loss_type = 'dft' needs TRL's fused LM head, but the model returned no "
                        "per-token log-probabilities. Use loss_type = 'nll'."
                    )
                per_token_loss = -outputs.log_probs * outputs.log_probs.exp().detach()
                # Trainer divides by gradient accumulation itself when the model does not take num_items_in_batch.
                if not getattr(self, "model_accepts_loss_kwargs", True):
                    num_items_in_batch = None
                num_tokens = (
                    outputs.label_mask.sum() if num_items_in_batch is None else num_items_in_batch
                )
                loss = per_token_loss.sum() / torch.as_tensor(num_tokens).clamp(min = 1)
                if self.args.average_tokens_across_devices and num_items_in_batch is not None:
                    # As TRL / Trainer: tensor parallel ranks share one batch, so they do not count.
                    tp_size = (
                        self.get_tp_size() if callable(getattr(self, "get_tp_size", None)) else 1
                    )
                    loss = loss * (self.accelerator.num_processes // max(int(tp_size or 1), 1))
                aux_loss = getattr(outputs, "aux_loss", None)
                if getattr(self, "aux_loss_enabled", False) and aux_loss is not None:
                    steps = (
                        self.current_gradient_accumulation_steps
                        if num_items_in_batch is not None
                        else 1
                    )
                    loss = loss + self.router_aux_loss_coef * aux_loss / steps
                if return_outputs:
                    # prediction_step / compute_metrics read logits, which the fused head never builds (as TRL does).
                    with torch.no_grad():
                        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
                return (loss, outputs) if return_outputs else loss
        outputs = super().compute_loss(
            model,
            inputs,
            return_outputs = return_outputs,
            num_items_in_batch = num_items_in_batch,
        )
        return outputs

    # During eval, Trainer calls prediction_step, which asks `compute_loss` for the outputs even when only the loss is
    # needed. Ask for them only when the logits are, since they take an extra forward pass.
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: list[str] | None = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            if prediction_loss_only:
                loss = self.compute_loss(model, inputs, return_outputs=False)
                logits, labels = None, None
            else:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                logits, labels = outputs.logits, inputs.get("labels")
        return loss, logits, labels

    # Override training step to add activation offloading context.
    def training_step(self, *args, **kwargs):
        with self.maybe_activation_offload_context:
            return super().training_step(*args, **kwargs)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs.update(metrics)
        super().log(logs, start_time)
        self._metrics[mode].clear()

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
class UnslothSFTTrainer(_UnslothSFTTrainer):
    """
    
    Trainer for Supervised Fine-Tuning (SFT) method.

    This class is a wrapper around the [`~transformers.Trainer`] class and inherits all of its attributes and methods.

    Example:

    ```python
    >>> from trl import SFTTrainer
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")

    >>> trainer = SFTTrainer(
    ...     model="Qwen/Qwen2.5-0.5B-Instruct",
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
        args ([`SFTConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Function to use to form a batch from a list of elements of the processed `train_dataset` or `eval_dataset`.
            Will default to [`~trainer.sft_trainer.DataCollatorForLanguageModeling`], or to
            [`~trainer.sft_trainer.DataCollatorForVisionLanguageModeling`] if the dataset contains images.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. This trainer supports both [language modeling](#language-modeling) type and
            [prompt-completion](#prompt-completion) type. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).

            The trainer also supports pre-tokenized datasets, recognized by a required `input_ids` column. An optional
            `labels` column (`-100` on tokens excluded from the loss) is used as is if present; otherwise labels are
            built from the optional `assistant_masks` / `completion_mask` columns (which are folded in then dropped,
            `completion_mask` only when `completion_only_loss=True`), or default to a copy of `input_ids`. Sequences
            are truncated to `max_length` during preparation. With `skip_prepare_dataset=True`, preparation is skipped
            and the collator is expected to handle the dataset as is.

            When `train_dataset` is an [`~datasets.IterableDataset`] (e.g. a streaming dataset), `max_steps` must be
            set in the training arguments, since its length cannot be inferred and the total number of training steps
            is required to bound the training loop and configure the learning rate scheduler.
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`], [`~datasets.DatasetDict`], [`~datasets.IterableDatasetDict`] or `dict[str, Dataset | IterableDataset]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. If `None`, the processing class is loaded from the model's name
            with [`~transformers.AutoProcessor.from_pretrained`]. A padding token, `tokenizer.pad_token`, must be set.
            If the processing class has not set a padding token, `tokenizer.eos_token` will be used as the default.
        compute_loss_func (`Callable`, *optional*):
            A function that accepts the model outputs, the labels, and the number of items in the entire accumulated
            batch (batch_size * gradient_accumulation_steps) and returns the loss. This is deprecated and will be
            removed in v2.0.0: subclass [`SFTTrainer`] and override `compute_loss` instead.
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a
            [`~transformers.EvalPrediction`] and return a dictionary string to metric values. When passing
            [`SFTConfig`] with `batch_eval_metrics` set to `True`, your `compute_metrics` function must take a boolean
            `compute_result` argument. This will be triggered after the last eval batch to signal that the function
            needs to calculate and return the global summary statistics rather than accumulating the batch-level
            statistics. The logits it receives are recomputed with an extra forward pass. This is deprecated and will
            be removed in v2.0.0.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of `AdamW` on your
            model and a scheduler given by [`~transformers.get_linear_schedule_with_warmup`] controlled by `args`.
        optimizer_cls_and_kwargs (`tuple[Type[torch.optim.Optimizer], Dict[str, Any]]`, *optional*):
            A tuple containing the optimizer class and keyword arguments to use. Overrides `optim` and `optim_args` in
            `args`. Incompatible with the `optimizers` argument.

            Unlike `optimizers`, this argument avoids the need to place model parameters on the correct devices before
            initializing the Trainer.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.

            Note that the labels (second parameter) will be `None` if the dataset does not have them.
        quantization_config ([`~transformers.BitsAndBytesConfig`], *optional*):
            Quantization configuration used when loading the model from a model identifier. Combine with `peft_config`
            for QLoRA training. Ignored if the model is already instantiated.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        formatting_func (`Callable`, *optional*):
            Formatting function applied to the dataset before tokenization. Applying the formatting function explicitly
            converts the dataset into a [language modeling](#language-modeling) type.
    
    """
    def __init__(
        self,
        model,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        processing_class = None,
        compute_loss_func = None,
        compute_metrics = None,
        callbacks = None,
        optimizer_cls_and_kwargs = None,
        preprocess_logits_for_metrics = None,
        quantization_config = None,
        peft_config = None,
        formatting_func = None,
        **kwargs
    ):
        if args is None: args = UnslothSFTConfig()
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
        _unsloth_explicit_max_length = None
        try:
            import dataclasses as _unsloth_dc
            _unsloth_cfg_cls = type(args)
            while '_unsloth_patched_rl_config' in _unsloth_cfg_cls.__dict__ or _unsloth_cfg_cls.__name__.startswith('Unsloth'):
                _unsloth_cfg_cls = _unsloth_cfg_cls.__bases__[0]
            _unsloth_default_max_length = None
            for _unsloth_field in _unsloth_dc.fields(_unsloth_cfg_cls):
                if _unsloth_field.name == 'max_length': _unsloth_default_max_length = _unsloth_field.default
            _unsloth_given_max_length = getattr(args, 'max_length', None)
            if (_unsloth_given_max_length or 0) > 0 and _unsloth_given_max_length != _unsloth_default_max_length:
                _unsloth_explicit_max_length = _unsloth_given_max_length
        except Exception:
            _unsloth_explicit_max_length = None
        if 'max_seq_length' not in locals() and not hasattr(args, 'max_seq_length'):
            pass
        else:
            model_max_seq_length = getattr(model, 'max_seq_length', None)
            args_max_seq_length  = getattr(args,  'max_seq_length', None)
            if args_max_seq_length is None and model_max_seq_length is not None:
                max_seq_length = min(model.max_seq_length, _unsloth_explicit_max_length) if _unsloth_explicit_max_length else model.max_seq_length
                if hasattr(args, 'max_seq_length'): args.max_seq_length = max_seq_length
            elif args_max_seq_length is not None and model_max_seq_length is not None:
                if args_max_seq_length > model_max_seq_length:
                    print('Unsloth: You set `max_seq_length` as ' + str(args_max_seq_length) + ' but '
                           'the maximum the model supports is ' + str(model_max_seq_length) + '. We shall reduce it.')
                    args.max_seq_length = model_max_seq_length
        if 'max_length' not in locals() and not hasattr(args, 'max_length'):
            pass
        else:
            if hasattr(args, 'max_seq_length') and args.max_seq_length is not None and args.max_seq_length > 0:
                if hasattr(args, 'max_length'):
                    args.max_length = args.max_seq_length
                    max_length = args.max_length
            else:
                model_max_length = getattr(model, 'max_seq_length', None)
                if model_max_length is None: model_max_length = getattr(model, 'max_length', None)
                if model_max_length is not None:
                    args.max_length = min(_unsloth_explicit_max_length, model_max_length) if _unsloth_explicit_max_length else model_max_length
                    max_length = args.max_length
                elif hasattr(args, 'max_length') and args.max_length is not None:
                    max_length = args.max_length
                    # if we are here, then we are in a weird case where max_length is set but max_seq_length is not set
                    setattr(model, 'max_seq_length', max_length)
                else:
                    print('Unsloth: We did not find `max_seq_length` or `max_length` in the model or args. We will set it to 1024.')
                    args.max_length = 1024
        if getattr(args, 'padding_free', False) is True and not getattr(args, 'packing', False) and getattr(args, 'max_length', None) is not None:
            _unsloth_prep_truncates = True
            _unsloth_skip_prepare = False
            try:
                _unsloth_dataset_kwargs = getattr(args, 'dataset_kwargs', None)
                if _unsloth_dataset_kwargs is not None and _unsloth_dataset_kwargs.get('skip_prepare_dataset', False):
                    _unsloth_prep_truncates = False
                    _unsloth_skip_prepare = True
            except Exception:
                pass
            def _unsloth_is_transformed(_ds):
                _f = getattr(_ds, 'format', None)
                _f = _f.get('type') if isinstance(_f, dict) else None
                return (_f or getattr(_ds, '_format_type', None)) == 'custom'
            try:
                _unsloth_transformed = _unsloth_is_transformed(train_dataset)
                _unsloth_columns = None if _unsloth_transformed else getattr(train_dataset, 'column_names', None)
                if _unsloth_columns is None and train_dataset is not None:
                    _unsloth_probe_cols = iter(train_dataset)
                    if _unsloth_probe_cols is train_dataset or _unsloth_probe_cols is iter(train_dataset):
                        _unsloth_prep_truncates = False
                    else:
                        _unsloth_first_row = next(_unsloth_probe_cols, None)
                        if isinstance(_unsloth_first_row, dict): _unsloth_columns = list(_unsloth_first_row.keys())
                if _unsloth_columns is None and not _unsloth_transformed:
                    _unsloth_columns = getattr(train_dataset, 'column_names', None)
                    if isinstance(_unsloth_columns, dict):
                        _unsloth_columns = [_c for _v in _unsloth_columns.values() for _c in (_v or [])]
                if _unsloth_columns is not None and ('input_ids' in _unsloth_columns or 'labels' in _unsloth_columns):
                    _unsloth_prep_truncates = False
            except Exception:
                _unsloth_prep_truncates = False
            def _unsloth_truncatable(_ds):
                if _ds is None or not hasattr(_ds, 'map'): return False
                try:
                    if str((getattr(_ds, 'format', None) or {}).get('type', '')).lower() == 'custom':
                        return False
                except Exception:
                    return False
                try:
                    _cols = getattr(_ds, 'column_names', None)
                    if isinstance(_cols, dict):
                        _cols = [_c for _v in _cols.values() for _c in (_v or [])]
                    if 'seq_lengths' in (_cols or ()): return False
                    return bool(_cols) and 'input_ids' in _cols
                except Exception:
                    return False
            _unsloth_cap = args.max_length
            _unsloth_truncation_mode = getattr(args, 'truncation_mode', 'keep_start') or 'keep_start'
            _unsloth_keep_end = _unsloth_truncation_mode == 'keep_end'
            _unsloth_known_mode = _unsloth_truncation_mode in ('keep_start', 'keep_end')
            _unsloth_slice = slice(-_unsloth_cap, None) if _unsloth_keep_end else slice(None, _unsloth_cap)
            _unsloth_eval_packing = getattr(args, 'packing', False) if getattr(args, 'eval_packing', None) is None else getattr(args, 'eval_packing')
            _unsloth_completion_only = getattr(args, 'completion_only_loss', None)
            if _unsloth_completion_only is None:
                try:
                    _unsloth_fmt = getattr(train_dataset, 'format', None)
                    _unsloth_fmt = _unsloth_fmt if isinstance(_unsloth_fmt, dict) else {}
                    _unsloth_shown = _unsloth_fmt.get('columns')
                    if _unsloth_fmt.get('output_all_columns') or not _unsloth_shown:
                        _unsloth_shown = getattr(train_dataset, 'column_names', None)
                    _unsloth_train_sample = {} if _unsloth_is_transformed(train_dataset) else dict.fromkeys(
                        _unsloth_shown or [])
                    if not _unsloth_train_sample:
                        _unsloth_probe = iter(train_dataset)
                        if _unsloth_probe is train_dataset or _unsloth_probe is iter(train_dataset):
                            _unsloth_train_sample = {}
                        else:
                            _unsloth_train_sample = next(_unsloth_probe, None) or {}
                except Exception:
                    _unsloth_train_sample = {}
                _unsloth_completion_only = ('prompt' in _unsloth_train_sample and 'completion' in _unsloth_train_sample)
            args._unsloth_completion_only_loss = _unsloth_completion_only
            _UNSLOTH_SCAN_ROWS = 1024
            def _unsloth_within_cap(_ds):
                if _ds is None: return True
                _unsloth_own = getattr(_ds, '__dict__', None)
                if isinstance(_unsloth_own, dict):
                    _unsloth_claim = _unsloth_own.get('_unsloth_truncated_to')
                    if isinstance(_unsloth_claim, int) and not isinstance(_unsloth_claim, bool):
                        return _unsloth_claim <= _unsloth_cap
                try:
                    try:    _n = len(_ds)
                    except Exception: _n = None
                    _unsloth_rows = iter(_ds)
                    if _unsloth_rows is iter(_ds): return False
                    _seen = 0
                    for _row in _unsloth_rows:
                        if 'input_ids' not in _row: return True
                        if len(_row['input_ids']) > _unsloth_cap: return False
                        _seen += 1
                        if _n is None and _seen >= _UNSLOTH_SCAN_ROWS: return False
                except Exception:
                    return False
                return True
            def _unsloth_splits_within_cap(_ev):
                _splits = list(_ev.values()) if isinstance(_ev, dict) else [_ev]
                return all(_unsloth_within_cap(_s) for _s in _splits)
            if not _unsloth_skip_prepare:
                _unsloth_map_kw = {}
                try:
                    try:
                        from unsloth_zoo.dataset_num_proc import get_dataset_num_proc as _unsloth_get_nproc
                    except ImportError:
                        from unsloth.dataset_num_proc import get_dataset_num_proc as _unsloth_get_nproc
                    _unsloth_nproc = _unsloth_get_nproc(getattr(args, 'dataset_num_proc', None))
                except Exception:
                    _unsloth_nproc = None
                if _unsloth_nproc: _unsloth_map_kw['num_proc'] = _unsloth_nproc
                def _unsloth_is_sequence_column(_col):
                    try:
                        if len(_col) == 0: return False
                        _first = _col[0]
                    except Exception:
                        return False
                    if isinstance(_first, (str, bytes)): return False
                    try:    len(_first)
                    except Exception: return False
                    return True
                def _unsloth_cut_value(_v, _r):
                    try:
                        if len(_v) != len(_r): return _v
                    except Exception: return _v
                    return _v[_unsloth_slice]
                def _unsloth_truncate_rows(_batch):
                    _ids = _batch.get('input_ids')
                    _out = {}
                    for _k, _col in _batch.items():
                        if not _unsloth_is_sequence_column(_col):
                            _out[_k] = _col
                        elif _ids is None:
                            _out[_k] = [_v[_unsloth_slice] for _v in _col]
                        else:
                            _out[_k] = [_unsloth_cut_value(_v, _r) for _v, _r in zip(_col, _ids)]
                    return _out
                def _unsloth_is_stream(_ds):
                    try:    return not hasattr(_ds, '__len__')
                    except Exception: return True
                def _unsloth_pretokenized(_ds):
                    try:
                        _cols = None if _unsloth_is_transformed(_ds) else getattr(_ds, 'column_names', None)
                        if isinstance(_cols, dict):
                            _cols = [_c for _v in _cols.values() for _c in (_v or [])]
                        if _cols is not None: return 'input_ids' in _cols
                        _probe = iter(_ds)
                        if _probe is _ds or _probe is iter(_ds): return True
                        _row = next(_probe, None)
                    except Exception: return True
                    return isinstance(_row, dict) and 'input_ids' in _row
                def _unsloth_rank_first():
                    try:
                        from accelerate import PartialState
                        return PartialState().main_process_first()
                    except Exception:
                        import contextlib
                        return contextlib.nullcontext()
                def _unsloth_cap_split(_ds):
                    with _unsloth_rank_first():
                        return _unsloth_cap_one(_ds)
                def _unsloth_cap_one(_ds):
                    if _ds is None: return _ds, True
                    if not _unsloth_truncatable(_ds): return _ds, not _unsloth_pretokenized(_ds)
                    _kw = {} if _unsloth_is_stream(_ds) else _unsloth_map_kw
                    _new = _ds.map(_unsloth_truncate_rows, batched = True, **_kw)
                    _unsloth_cols = getattr(_new, 'column_names', None) or ()
                    _unsloth_masks = []
                    if _unsloth_completion_only and 'completion_mask' in _unsloth_cols:
                        _unsloth_masks.append('completion_mask')
                    if 'assistant_masks' in _unsloth_cols:
                        _unsloth_masks.append('assistant_masks')
                    try:
                        _unsloth_supervision = (['labels'] if 'labels' in _unsloth_cols else []) + _unsloth_masks
                        if _unsloth_supervision:
                            _new = _new.filter(lambda _e, _c = tuple(_unsloth_supervision): any(all((_x != -100) if _n == 'labels' else _x for _n, _x in zip(_c, _v)) for _v in zip(*[_e[_n] for _n in _c])), **_kw)
                    except Exception:
                        pass
                    try:
                        if _unsloth_supervision and len(_new) == 0: _unsloth_emptied.append(1)
                    except TypeError:
                        pass
                    return _new, (True if _unsloth_is_stream(_new) else _unsloth_within_cap(_new))
                _unsloth_emptied = []
                _unsloth_orig_train = train_dataset
                _unsloth_orig_eval = eval_dataset if 'eval_dataset' in locals() else None
                try:
                    _unsloth_capped = _unsloth_known_mode
                    if not _unsloth_known_mode:
                        print('Unsloth: `truncation_mode = ' + str(_unsloth_truncation_mode) + '` is not one of keep_start / keep_end, so `max_length` is not being enforced here.')
                    if _unsloth_known_mode and not _unsloth_prep_truncates:
                        train_dataset, _unsloth_split_ok = _unsloth_cap_split(train_dataset)
                        _unsloth_capped = _unsloth_capped and _unsloth_split_ok
                    if _unsloth_eval_packing or not _unsloth_known_mode:
                        _unsloth_capped = False
                    elif 'eval_dataset' in locals() and eval_dataset is not None:
                        if isinstance(eval_dataset, dict):
                            _unsloth_new_eval = {}
                            for _k, _v in eval_dataset.items():
                                _unsloth_new_eval[_k], _unsloth_split_ok = _unsloth_cap_split(_v)
                                _unsloth_capped = _unsloth_capped and _unsloth_split_ok
                            eval_dataset = _unsloth_new_eval
                        else:
                            eval_dataset, _unsloth_split_ok = _unsloth_cap_split(eval_dataset)
                            _unsloth_capped = _unsloth_capped and _unsloth_split_ok
                    _unsloth_prep_truncates = _unsloth_capped
                    if not _unsloth_capped:
                        print('Unsloth: `max_length` cannot be enforced for every split here, so padding-free batching is being turned off instead.')
                except Exception as _unsloth_truncate_error:
                    train_dataset = _unsloth_orig_train
                    if 'eval_dataset' in locals(): eval_dataset = _unsloth_orig_eval
                    _unsloth_prep_truncates = False
                    print('Unsloth: could not truncate the pre-tokenized dataset to `max_length` (' + str(_unsloth_truncate_error) + ').')
                if _unsloth_emptied:
                    raise ValueError('Unsloth: truncating to `max_length = ' + str(args.max_length) + '` left every row with no supervised token, so there is nothing to train on. The supervised part of your rows starts past that length: raise `max_length`, or set `truncation_mode = "keep_end"` if the completion sits at the end of each row.')
            if not _unsloth_prep_truncates and not _unsloth_eval_packing:
                def _unsloth_attests(_ds):
                    if _ds is None: return True
                    _own = getattr(_ds, '__dict__', None)
                    if not isinstance(_own, dict): return False
                    _claim = _own.get('_unsloth_truncated_to')
                    if not isinstance(_claim, int) or isinstance(_claim, bool): return False
                    return _claim <= _unsloth_cap
                _unsloth_attest_eval = eval_dataset if 'eval_dataset' in locals() else None
                _unsloth_attest_splits = list(_unsloth_attest_eval.values()) if isinstance(_unsloth_attest_eval, dict) else [_unsloth_attest_eval]
                if _unsloth_attests(train_dataset) and all(_unsloth_attests(_s) for _s in _unsloth_attest_splits):
                    _unsloth_prep_truncates = True
            if _unsloth_prep_truncates:
                args.max_seq_length = args.max_length
                args.max_length = None
                max_length = None
            else:
                _unsloth_scan_eval = None if _unsloth_eval_packing else (eval_dataset if 'eval_dataset' in locals() else None)
                if not (_unsloth_within_cap(train_dataset) and _unsloth_splits_within_cap(_unsloth_scan_eval)):
                    raise ValueError('Unsloth: `max_length = ' + str(args.max_length) + '` cannot be enforced. Your dataset already carries `input_ids` and holds rows longer than that, and nothing downstream truncates pre-tokenized rows. Truncate it yourself before passing it in, or drop `max_length`.')
                print('Unsloth: Turning padding-free batching off, since your dataset is already tokenized and cannot be truncated here. Padding-free batching cannot enforce a `max_length` of ' + str(args.max_length) + '.')
                args.padding_free = False
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
        PatchRLStatistics('sft_trainer', other_metrics)
        IGNORED_TOKENIZER_NAMES = os.environ.get('UNSLOTH_IGNORED_TOKENIZER_NAMES', '').split('\n')
        from unsloth_zoo.tokenizer_utils import fix_untrained_tokens
        from unsloth_zoo.training_utils  import fix_zero_training_loss
        if 'tokenizer' not in locals(): tokenizer = processing_class
        fix_untrained_tokens(model, tokenizer, train_dataset, IGNORED_TOKENIZER_NAMES, eps = 1e-16)
        fix_zero_training_loss(model, tokenizer, train_dataset)
        
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
            data_collator = data_collator,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            processing_class = processing_class,
            compute_loss_func = compute_loss_func,
            compute_metrics = compute_metrics,
            callbacks = callbacks,
            optimizer_cls_and_kwargs = optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics = preprocess_logits_for_metrics,
            quantization_config = quantization_config,
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
        if hasattr(self, 'aux_loss_enabled') and hasattr(getattr(self, 'model', None), 'modules'):
            for _module in self.model.modules():
                _config = getattr(_module, 'config', None)
                if 'router_aux_loss_coef' in vars(_module) and hasattr(_config, 'get_text_config'):
                    _coef = getattr(_config.get_text_config(), 'router_aux_loss_coef', None)
                    if _coef is not None:
                        _module.router_aux_loss_coef = _coef
        pass
        
pass


if hasattr(logger, "addFilter"):
    import logging
    class HideLoggingMessage(logging.Filter):
        def __init__(self, text): self.text = text
        def filter(self, x): return not (self.text in x.getMessage())
    pass
    logger.addFilter(HideLoggingMessage("`use_cache=True`"))

