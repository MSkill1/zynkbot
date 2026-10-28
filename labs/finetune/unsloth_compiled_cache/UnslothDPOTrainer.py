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
from trl.trainer.dpo_trainer import (Any, AutoProcessor, BaseTunerLayer, BitsAndBytesConfig, Callable, DPOConfig, DPOTrainer, DataCollator, DataCollatorForPreference, DataCollatorForVisionPreference, DataLoader, Dataset, DatasetDict, EvalPrediction, F, Hasher, IterableDataset, IterableDatasetDict, LoraConfig, Path, PeftConfig, PeftModel, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, SyncRefModelCallback, TrainerCallback, Version, _BaseTrainer, _tokenize, accelerate, add_fused_lm_head, apply_chat_template, broadcast_object_list, common_prefix_length, concatenate_datasets, contextlib, create_model_from_path, dataclass, defaultdict, disable_dropout_in_model, disable_gradient_checkpointing, extract_prompt, get_act_offloading_ctx_manager, get_config_model_id, get_peft_model, global_then_local_main_first, hash_module, is_conversational, is_peft_available, is_peft_model, json, logger, math, os, pad, peft, prepare_deepspeed, prepare_fsdp, set_seed, textwrap, torch, tqdm, transformers, use_adapter, warnings, AutoProcessor, BaseTunerLayer, BitsAndBytesConfig, Callable, DPOConfig, DPOTrainer, DataCollator, DataCollatorForPreference, DataCollatorForVisionPreference, Dataset, DatasetDict, EvalPrediction, F, IterableDataset, IterableDatasetDict, LoraConfig, PeftConfig, PeftModel, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, SyncRefModelCallback, TrainerCallback, Version, accelerate, add_fused_lm_head, contextlib, create_model_from_path, defaultdict, disable_dropout_in_model, get_act_offloading_ctx_manager, get_config_model_id, get_peft_model, hash_module, is_peft_available, is_peft_model, logger, os, pad, peft, prepare_deepspeed, prepare_fsdp, set_seed, torch, transformers, warnings, F, PeftModel, PreTrainedModel, is_peft_available, logger, os, peft, torch)


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
def dpo_trainer_vision_process_row(
    features,
    processing_class,
    max_prompt_length = None,
    max_completion_length = None,
    add_special_tokens = True,
    is_chat = False,
):
    text = features.get("prompt", "")
    images = features.get("images")
    processor, tokenizer = processing_class, processing_class.tokenizer
    processed_features = processor(
        images = images,
        text = text,
        add_special_tokens = False,
    )

    prompt_input_ids = processed_features["input_ids"][0]
    chosen_input_ids = tokenizer(features["chosen"], add_special_tokens = False)["input_ids"]
    rejected_input_ids = tokenizer(features["rejected"], add_special_tokens = False)["input_ids"]

    if add_special_tokens:
        if tokenizer.bos_token_id is not None:
            prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
        if tokenizer.eos_token_id is not None:
            prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
    if not is_chat and tokenizer.eos_token_id is not None:
        chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

    if max_prompt_length is not None:
        prompt_input_ids = prompt_input_ids[-max_prompt_length:]
    if max_completion_length is not None:
        chosen_input_ids = chosen_input_ids[:max_completion_length]
        rejected_input_ids = rejected_input_ids[:max_completion_length]

    output = {
        "prompt_input_ids": prompt_input_ids,
        "chosen_input_ids": chosen_input_ids,
        "rejected_input_ids": rejected_input_ids,
    }
    if "pixel_values" in processed_features:
        output["pixel_values"] = processed_features["pixel_values"][0]
    if "pixel_attention_mask" in processed_features:
        output["pixel_attention_mask"] = processed_features["pixel_attention_mask"][0]
    if "image_sizes" in processed_features:
        output["image_sizes"] = processed_features["image_sizes"][0]
    if "token_type_ids" in processed_features:
        token_type_ids = processed_features["token_type_ids"][0]
        if max_prompt_length is not None:
            token_type_ids = token_type_ids[-max_prompt_length:]
        output["token_type_ids"] = token_type_ids
    if "pixel_position_ids" in processed_features:
        output["pixel_position_ids"] = processed_features["pixel_position_ids"][0]
    if "image_position_ids" in processed_features:
        output["image_position_ids"] = processed_features["image_position_ids"][0]
    if "mm_token_type_ids" in processed_features:
        mm_token_type_ids = processed_features["mm_token_type_ids"][0]
        if max_prompt_length is not None:
            mm_token_type_ids = mm_token_type_ids[-max_prompt_length:]
        output["mm_token_type_ids"] = mm_token_type_ids

    return output

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

def _unsloth_pref_returns_hidden_states(model, tensor, lm_head):
    """``_unsloth_grpo_returns_hidden_states`` for the preference routes, which need a zoo that writes the hidden-state
    markers: a square head with no signal is a forward that ignored the flag, so its output is real logits."""
    if tensor.shape[-1] != lm_head.shape[1]:
        return False
    if lm_head.shape[0] != lm_head.shape[1]:
        return True
    return _unsloth_grpo_hidden_states_signal(model) is True

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

def _unsloth_pref_logit_transforms(model):
    """``(multiply, divide, softcap)`` for the chunked kernel, from the GKD resolver (one scale, 0.0 = off)."""
    scale, softcap = _unsloth_gkd_logit_transforms(model)
    return (0.0 if scale == 1.0 else scale), 0.0, softcap

def _unsloth_stock_logps_hook(
    self,
    method,
    hooks = ("get_batch_logps",),
):
    """False when a subclass overrides one of ``hooks`` below the class whose ``method`` is the rewritten one: TRL's
    body would call the override (or it reads the full logits the chunked path replaces with stand-ins)."""
    owner = None
    for klass in type(self).__mro__:
        # The rewritten body calls a chunked or fused-pass helper; a subclass method wrapping it via super() does not.
        names = getattr(getattr(vars(klass).get(method), "__code__", None), "co_names", ())
        if any(
            name.startswith("_unsloth_") and ("chunked" in name or "fused_pass" in name)
            for name in names
        ):
            owner = klass
            break
    if owner is None:
        return True
    for name in hooks:
        hook = next((klass for klass in type(self).__mro__ if name in vars(klass)), None)
        if hook is not None and hook not in owner.__mro__:
            return False
    return True

def _unsloth_dpo_chunked_head(
    self,
    model,
    concatenated_batch,
    method = "concatenated_forward",
):
    """``(unwrapped_model, head, (multiply, divide, softcap))`` when concatenated_forward can score from hidden
    states, else ``None`` (TRL's own logits path runs). Decided before the forward, which then skips the logits."""
    reason = None
    if _unsloth_masked_logprobs is None:
        reason = "unsloth_zoo has no chunked_logprob"
    elif os.environ.get("UNSLOTH_FUSED_LINEAR", "1") == "0":
        reason = "UNSLOTH_FUSED_LINEAR=0"
    elif os.environ.get("UNSLOTH_DPO_CHUNKED", "1").lower() in ("0", "false", "no", "off"):
        reason = "UNSLOTH_DPO_CHUNKED=0"
    elif getattr(self, "is_encoder_decoder", False):
        reason = "encoder-decoder"
    elif getattr(self, "aux_loss_enabled", False):
        reason = "router aux loss needs the stock forward"
    elif not _unsloth_stock_logps_hook(
        self,
        method,
        ("get_batch_logps",)
        if method == "concatenated_forward"
        else ("get_batch_logps", "get_batch_loss_metrics", method),
    ):
        # KTO hands its callers logits stand-ins; DPO's outputs are exact, so a DPO wrapper is fine.
        reason = "get_batch_logps / get_batch_loss_metrics overridden"
    elif getattr(self, "padding_free", False):
        reason = "padding_free"
    elif getattr(self, "use_weighting", False):
        reason = "use_weighting needs full-vocabulary log-probs"
    elif getattr(self, "is_fsdp_enabled", False) or getattr(self, "is_deepspeed_enabled", False):
        reason = "FSDP / DeepSpeed"
    elif any(
        k in concatenated_batch
        for k in ("pixel_values", "pixel_attention_mask", "image_sizes", "token_type_ids")
        + tuple(_DPO_VISION_KEYS)
    ):
        reason = "vision inputs"
    if reason is None:
        try:
            unwrapped = self.accelerator.unwrap_model(model)
        except Exception:
            unwrapped = None
            reason = "unwrap_model"
    if reason is None:
        head = _unsloth_gkd_dense_head(unwrapped)
        if head is None:
            reason = "output head is not a dense nn.Linear"
        # The kernel reads weight / bias directly: a hooked head (accelerate, forward / backward hooks) is not
        # F.linear(h, W, b), and its backward hooks would never see the gradient.
        elif (
            hasattr(head, "_hf_hook")
            or "forward" in vars(head)
            or head._forward_hooks
            or head._forward_pre_hooks
            or head._backward_hooks
            or getattr(head, "_backward_pre_hooks", None)
        ):
            reason = "output head has hooks or a patched forward"
    if reason is None:
        for config in _unsloth_text_configs(_unsloth_get_model_config(unwrapped)):
            if getattr(config, "model_type", None) == "minicpm3":
                reason = "minicpm3 scales hidden states before the head"
    if reason is None and getattr(model, "find_unused_parameters", False):
        if any(p is not None and p.requires_grad for p in (head.weight, head.bias)):
            reason = "DDP find_unused_parameters with a trainable head"
    if reason is not None:
        try:
            fallbacks = self.__dict__.setdefault("_unsloth_dpo_chunked_fallbacks", {})
            fallbacks[reason] = fallbacks.get(reason, 0) + 1
        except Exception:
            pass
        return None
    return unwrapped, head, _unsloth_pref_logit_transforms(unwrapped)

def _unsloth_dpo_model_forward(model, input_ids, model_kwargs, want_hidden_states):
    if not want_hidden_states:
        return model(input_ids, **model_kwargs)
    prior = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    try:
        return model(input_ids, **model_kwargs)
    finally:
        if prior is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = prior

def _unsloth_dpo_returns_hidden_states(self, chunked, logits):
    if chunked is None:
        return False
    unwrapped, head, _ = chunked
    hidden = _unsloth_pref_returns_hidden_states(unwrapped, logits, head.weight)
    if not hidden:
        fallbacks = self.__dict__.setdefault("_unsloth_dpo_chunked_fallbacks", {})
        reason = "forward returned logits, not hidden states"
        fallbacks[reason] = fallbacks.get(reason, 0) + 1
    return hidden

def _unsloth_dpo_hidden_logps(self, chunked, hidden_states, labels, loss_mask):
    """``(per_token_logps [B, S], mean_logits [B, S, 1])`` where TRL computes ``selective_log_softmax(logits, labels)``
    and later reads ``logits[mask].mean()``. Rows outside ``loss_mask`` are 0 and never reach the head; the mean over
    a masked row's vocab equals TRL's element mean, so the stand-in keeps the ``logits/*`` metrics exact."""
    _, head, (multiply, divide, softcap) = chunked
    per_token_logps, mean_logits = _unsloth_masked_logprobs(
        hidden_states.to(head.weight.device),
        head.weight,
        labels,
        mask = loss_mask,
        logit_scale_multiply = multiply,
        logit_scale_divide = divide,
        logit_softcapping = softcap,
        lm_head_bias = head.bias,
        return_mean_logits = True,
    )
    try:
        self._unsloth_dpo_chunked_calls = getattr(self, "_unsloth_dpo_chunked_calls", 0) + 1
    except Exception:
        pass
    return per_token_logps.to(labels.device), mean_logits.unsqueeze(-1).to(labels.device)

def _unsloth_dpo_nll(chosen_logps, chosen_labels):
    """``F.cross_entropy(chosen_logits, chosen_labels, ignore_index=0)`` from the per-token log-probs at the same
    positions (TRL zeroes masked labels, so 0 is its ignore index)."""
    keep = chosen_labels != 0
    return -(chosen_logps * keep).sum() / keep.sum()

def _unsloth_dpo_forward(model, model_kwargs, want_hidden_states):
    if not want_hidden_states:
        return model(**model_kwargs)
    prior = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    try:
        return model(**model_kwargs)
    finally:
        if prior is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = prior

def _unsloth_dpo_loss_head(
    self,
    model,
    inputs,
    return_outputs = False,
):
    """``_unsloth_dpo_chunked_head`` for TRL 0.29 - 1.14's ``_compute_loss``, which also needs the fused linear op
    (entropy and argmax match) and must not hand hidden states back as ``outputs.logits``."""
    reason = None
    if _unsloth_fused_linear_logprobs is None:
        reason = "unsloth_zoo has no fused_linear_logprobs"
    elif os.environ.get("UNSLOTH_FUSED_LINEAR", "1") == "0":
        reason = "UNSLOTH_FUSED_LINEAR=0"
    elif return_outputs:
        reason = "return_outputs needs the logits"
    if reason is not None:
        try:
            fallbacks = self.__dict__.setdefault("_unsloth_dpo_chunked_fallbacks", {})
            fallbacks[reason] = fallbacks.get(reason, 0) + 1
        except Exception:
            pass
        return None
    return _unsloth_dpo_chunked_head(self, model, inputs, "_compute_loss")

def _unsloth_dpo_loss_stats(
    self,
    chunked,
    hidden_states,
    labels,
    mask,
    ref = False,
):
    """TRL's per-token statistics of ``logits = head(hidden_states)`` at completion positions (``mask``), the rest 0:
    ``(log_probs, entropy, mean_logits [.., 1], is_top1)``, or the log-probs alone for ``ref``. ``mean_logits`` stands
    in for the logits: ``x[mask].mean(-1)`` on it is TRL's vocab mean. No temperature: TRL's DPO scores at 1.0."""
    _, head, (multiply, divide, softcap) = chunked
    # TRL projects the reference logits inside its no_grad forward: a trainable (or shared) head gets no gradient.
    with torch.set_grad_enabled(torch.is_grad_enabled() and not ref):
        out = _unsloth_fused_linear_logprobs(
            hidden_states,
            head.weight,
            labels,
            bias = head.bias,
            mask = mask,
            logit_scale_multiply = multiply,
            logit_scale_divide = divide,
            logit_softcapping = softcap,
            return_entropy = not ref,
            return_mean_logits = not ref,
            return_is_top1 = not ref,
            entropy_requires_grad = False,
        )
    try:
        self._unsloth_dpo_chunked_calls = getattr(self, "_unsloth_dpo_chunked_calls", 0) + 1
    except Exception:
        pass
    if ref:
        return out
    logps, entropy, mean_logits, is_top1 = out
    return logps, entropy, mean_logits.unsqueeze(-1), is_top1

try:
    from unsloth_zoo.device_map_planner import detect_logit_transforms
except Exception:
    detect_logit_transforms = None
try:
    from unsloth_zoo.fused_losses.chunked_logprob import (
        unsloth_masked_logprobs as _unsloth_masked_logprobs,
    )
except Exception:
    _unsloth_masked_logprobs = None
try:
    from unsloth_zoo.fused_losses.fused_linear import (
        fused_linear_logprobs as _unsloth_fused_linear_logprobs,
    )
except Exception:
    _unsloth_fused_linear_logprobs = None
_DPO_VISION_KEYS = ('pixel_position_ids', 'image_position_ids', 'mm_token_type_ids')

def _unsloth_fused_pass_head(self, model, inputs, return_outputs, method):
    """``_unsloth_dpo_chunked_head`` for a rewritten pass that also needs the fused linear op, keeps TRL's path for
    the kill switch, ``return_outputs`` (the caller reads ``outputs.logits``) and TRL's own liger / chunked head."""
    reason = None
    if _unsloth_fused_linear_logprobs is None:
        reason = "unsloth_zoo has no fused_linear_logprobs"
    elif os.environ.get("UNSLOTH_FUSED_LINEAR", "1") == "0":
        reason = "UNSLOTH_FUSED_LINEAR=0"
    elif return_outputs:
        reason = "return_outputs needs the logits"
    elif getattr(self, "use_liger_kernel", False):
        reason = "use_liger_kernel runs TRL's own chunked head"
    if reason is not None:
        try:
            fallbacks = self.__dict__.setdefault("_unsloth_dpo_chunked_fallbacks", {})
            fallbacks[reason] = fallbacks.get(reason, 0) + 1
        except Exception:
            pass
        return None
    return _unsloth_dpo_chunked_head(self, model, inputs, method)

def _unsloth_fused_pass_logps(
    self,
    chunked,
    hidden_states,
    labels,
    mask,
    stats = False,
    grad = False,
):
    """TRL's ``selective_log_softmax(head(hidden_states), labels)`` with rows outside ``mask`` zeroed (TRL's
    ``row_mask`` or its ``logps[mask == 0] = 0.0``). ``stats`` also returns the entropy (no grad, as TRL computes it
    from detached logits) and the per-position vocab mean ``[.., 1]``: ``x[mask].mean(-1)`` on it is TRL's mean of
    the logits row. A shorter ``hidden_states`` (the model truncated input_ids) is scored over the common length and
    zero padded. Only the policy pass (``grad``) keeps a graph: TRL scores the reference / KL logits it built under
    no_grad, so a trainable head must get no gradient from them."""
    _, head, (multiply, divide, softcap) = chunked
    width = labels.shape[1]
    n = min(hidden_states.shape[1], width)
    with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
        out = _unsloth_fused_linear_logprobs(
            hidden_states[:, :n],
            head.weight,
            labels[:, :n],
            bias = head.bias,
            mask = mask[:, :n].bool(),
            logit_scale_multiply = multiply,
            logit_scale_divide = divide,
            logit_softcapping = softcap,
            return_entropy = stats,
            return_mean_logits = stats,
            entropy_requires_grad = False,
        )
    try:
        self._unsloth_dpo_chunked_calls = getattr(self, "_unsloth_dpo_chunked_calls", 0) + 1
    except Exception:
        pass
    out = out if isinstance(out, tuple) else (out,)
    out = tuple(torch.nn.functional.pad(x, (0, width - n)) if n != width else x for x in out)
    if not stats:
        return out[0].to(labels.device)
    logps, entropy, mean_logits = out
    return (
        logps.to(labels.device),
        entropy.to(labels.device),
        mean_logits.unsqueeze(-1).to(labels.device),
    )

def _unsloth_fused_pass_vocab_sums(self, chunked, hidden_states):
    """``[B, S, 1]`` per-position vocab sums of ``head(hidden_states)`` (no grad): TRL 1.6 - 1.8 KTO logs
    ``logits.index_select(0, idx).nansum()``, a sum over every position and vocab entry, which this keeps exact."""
    _, head, (multiply, divide, softcap) = chunked
    with torch.no_grad():
        hidden_states = hidden_states.detach()
        _, mean_logits = _unsloth_fused_linear_logprobs(
            hidden_states,
            head.weight,
            torch.zeros(hidden_states.shape[:-1], dtype = torch.long, device = hidden_states.device),
            bias = head.bias,
            logit_scale_multiply = multiply,
            logit_scale_divide = divide,
            logit_softcapping = softcap,
            return_mean_logits = True,
        )
    return (mean_logits * head.weight.shape[0]).unsqueeze(-1).to(hidden_states.device)
@dataclass
class UnslothDPOConfig(DPOConfig):
    """
    
    Configuration class for the [`DPOTrainer`].

    This class includes only the parameters that are specific to DPO training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model

        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`DPOTrainer`] is provided as a string. The `revision` value is also used when loading the
            processing class.
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to
            [`~transformers.AutoModelForCausalLM.from_pretrained`] and [`~transformers.AutoProcessor.from_pretrained`].
        router_aux_loss_coef (`float`, *optional*):
            Coefficient of the load-balancing auxiliary loss for Mixture-of-Experts (MoE) models, added to the training
            loss with this weight. When not set, the value declared by the model config is used. Set to `0.0` to
            disable it. Fails when used with a non-MoE model.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model and reference model.

        > Parameters that control the data preprocessing

        dataset_num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from the left or
            right depending on the `truncation_mode`. If `None`, no truncation is applied.
        truncation_mode (`str`, *optional*, defaults to `"keep_start"`):
            Truncation mode to use when the sequence exceeds `max_length`. The only supported value is `"keep_start"`.
            The `"keep_end"` value is deprecated and will be removed in v2.0.0.
        padding_free (`bool`, *optional*, defaults to `False`):
            Whether to perform forward passes without padding by flattening all sequences in the batch into a single
            continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this is only
            supported with the FlashAttention 2 or 3, which can efficiently handle the flattened batch structure.
            Temporarily unavailable: since the DPO refactor, setting it to `True` warns and falls back to standard
            padding. It is planned to return in a future update.
        pad_to_multiple_of (`int`, *optional*):
            If set, the sequences will be padded to a multiple of this value.
        precompute_ref_log_probs (`bool`, *optional*, defaults to `False`):
            Whether to precompute the reference model log probabilities for the entire training dataset before
            training. This allows to save memory during training, as the reference model does not need to be kept in
            memory.
        precompute_ref_batch_size (`int`, *optional*):
            Batch size to use when precomputing reference model log probabilities. This can be set higher than the
            training batch size to speed up preprocessing. If `None`, defaults to `per_device_train_batch_size` for
            training and `per_device_eval_batch_size` for evaluation.

        > Parameters that control the training

        loss_type (`list[str]`, *optional*, defaults to `["sigmoid"]`):
            Type of loss to use. Possible values are: `'sigmoid'`, `'hinge'`, `'ipo'`, `'exo_pair'`, `'nca_pair'`,
            `'robust'`, `'bco_pair'`, `'sppo_hard'`, `'aot'`, `'aot_unpaired'`, `'apo_zero'`, `'apo_down'`,
            `'discopop'`, `'sft'`, `'sigmoid_norm'`. If multiple loss types are provided, they will be combined using
            the weights specified in `loss_weights`.
        loss_weights (`list[float]`, *optional*):
            List of loss weights for multi-loss combinations. Used when combining multiple loss types. Example: `[0.8,
            0.2, 1.0]` for MPO. If not provided, defaults to equal weights (`1.0`) for all loss types.
        ld_alpha (`float`, *optional*):
            α parameter from the LD-DPO paper, which controls the weighting of the verbose token log-probabilities in
            responses. If `None`, no weighting is applied to the verbose part, and the loss is equivalent to the
            standard DPO loss. Must be in [0.0, 1.0]: `ld_alpha=1.0` applies no weighting, and `ld_alpha=0.0` masks
            tokens beyond shared lengths.
        f_divergence_type (`str`, *optional*, defaults to `"reverse_kl"`):
            f-divergence regularizer between policy and reference (f-DPO paper). Possible values are: `reverse_kl`
            (default), `forward_kl`, `js_divergence`, `alpha_divergence`. Only the loss types built on the
            chosen-rejected reward difference support a non-default value: `'sigmoid'`, `'sigmoid_norm'`, `'hinge'`,
            `'ipo'`, `'exo_pair'`, `'robust'`, `'discopop'`, `'sft'`. The other loss types have no valid f-divergence
            generalization and are rejected with a non-default value.
        f_alpha_divergence_coef (`float`, *optional*, defaults to `0.5`):
            α coefficient for the α-divergence u^-α regularizer, used only when `f_divergence_type='alpha_divergence'`.
        label_smoothing (`float`, *optional*, defaults to `0.0`):
            Label smoothing parameter used in Robust DPO and EXO. In Robust DPO, it is interpreted as the probability
            that a preference label is flipped and must lie in [0.0, 0.5); a typical value recommended by the Robust
            DPO paper is 0.1. In EXO, it corresponds to the ε label smoothing parameter, for which the paper recommends
            a typical value of 1e-3.
        beta (`float`, *optional*, defaults to `0.1`):
            Parameter controlling the deviation from the reference model. Higher β means less deviation from the
            reference model. For the IPO loss (`loss_type='ipo'`), this value is the regularization parameter denoted
            by τ in the [paper](https://huggingface.co/papers/2310.12036).
        use_weighting (`bool`, *optional*, defaults to `False`):
            Whether to apply [WPO](https://huggingface.co/papers/2406.11827)-style weighting to preference pairs using
            the policy's length-normalized sequence probabilities.
        discopop_tau (`float`, *optional*, defaults to `0.05`):
            τ/temperature parameter from the DiscoPOP paper, which controls the shape of the log-ratio modulated loss
            when using `loss_type='discopop'`. The paper recommends the default value `discopop_tau=0.05`.
        activation_offloading (`bool`, *optional*, defaults to `False`):
            Whether to offload the activations to the CPU.
        sync_ref_model (`bool`, *optional*, defaults to `False`):
            Whether to synchronize the reference model with the active model every `ref_model_sync_steps` steps, using
            the `ref_model_mixup_alpha` parameter. This synchronization originates from the
            [TR-DPO](https://huggingface.co/papers/2404.09656) paper. `sync_ref_model=True` is not yet compatible with
            PEFT or `precompute_ref_log_probs=True`.
        ref_model_mixup_alpha (`float`, *optional*, defaults to `0.6`):
            α parameter from the TR-DPO paper, which controls the mix between the current policy and the previous
            reference policy during updates. The reference policy is updated according to the equation: `π_ref = α *
            π_θ + (1 - α) * π_ref_prev`. To use this parameter, you must set `sync_ref_model=True`.
        ref_model_sync_steps (`int`, *optional*, defaults to `512`):
            τ parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which determines how
            frequently the current policy is synchronized with the reference policy. To use this parameter, you must
            set `sync_ref_model=True`.

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
    max_seq_length : Optional[int] = field(
        default = None,
        metadata = {'help': 'Maximum sequence length to truncate to.'},
    )
    context_parallel_size : Optional[int] = field(
        default = 1,
        metadata = {'help': 'Ranks per context parallel group (training forwards only). 1 disables it.'},
    )
    context_parallel_backend : Optional[str] = field(
        default = 'ulysses',
        metadata = {'help': 'Context parallel attention backend; preference and RL trainers support ulysses.'},
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
        disable_dropout = True,
        dataset_num_proc = None,
        max_length = 1024,
        truncation_mode = 'keep_start',
        padding_free = None,
        pad_to_multiple_of = None,
        precompute_ref_log_probs = False,
        precompute_ref_batch_size = None,
        loss_weights = None,
        ld_alpha = None,
        f_divergence_type = 'reverse_kl',
        f_alpha_divergence_coef = 0.5,
        label_smoothing = 0.0,
        beta = 0.1,
        use_weighting = False,
        discopop_tau = 0.05,
        activation_offloading = False,
        sync_ref_model = False,
        ref_model_mixup_alpha = 0.6,
        ref_model_sync_steps = 512,
        pad_token = None,
        vllm_sampling_params = None,
        unsloth_num_chunks = -1,
        unsloth_logit_chunk_multiplier = None,
        unsloth_grpo_mini_batch = None,
        max_seq_length = None,context_parallel_size = 1,context_parallel_backend = 'ulysses',
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
            disable_dropout = disable_dropout,
            dataset_num_proc = dataset_num_proc,
            max_length = max_length,
            truncation_mode = truncation_mode,
            padding_free = padding_free,
            pad_to_multiple_of = pad_to_multiple_of,
            precompute_ref_log_probs = precompute_ref_log_probs,
            precompute_ref_batch_size = precompute_ref_batch_size,
            loss_weights = loss_weights,
            ld_alpha = ld_alpha,
            f_divergence_type = f_divergence_type,
            f_alpha_divergence_coef = f_alpha_divergence_coef,
            label_smoothing = label_smoothing,
            beta = beta,
            use_weighting = use_weighting,
            discopop_tau = discopop_tau,
            activation_offloading = activation_offloading,
            sync_ref_model = sync_ref_model,
            ref_model_mixup_alpha = ref_model_mixup_alpha,
            ref_model_sync_steps = ref_model_sync_steps,
            pad_token = pad_token,**kwargs)
        super().__init__(**_unsloth_filter_config_init_kwargs(DPOConfig, _unsloth_config_arguments, mirrored_from = __class__))
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

class _UnslothDPOTrainer(_BaseTrainer):
    """"""

    loss_is_scaled_for_ga = False

    _tag_names = ["trl", "dpo"]
    _name = "DPO"
    _paper = {
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "id": "2305.18290",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @inproceedings{rafailov2023direct,
                title        = {{Direct Preference Optimization: Your Language Model is Secretly a Reward Model}},
                author       = {Rafael Rafailov and Archit Sharma and Eric Mitchell and Christopher D. Manning and Stefano Ermon and Chelsea Finn},
                year         = 2023,
                booktitle    = {Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023},
                url          = {http://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html},
                editor       = {Alice Oh and Tristan Naumann and Amir Globerson and Kate Saenko and Moritz Hardt and Sergey Levine},
            }"""),
    }

    def __init__(
        self,
        model: "str | PreTrainedModel | PeftModel",
        ref_model: PreTrainedModel | None = None,
        args: DPOConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset
        | IterableDataset
        | DatasetDict
        | IterableDatasetDict
        | dict[str, Dataset | IterableDataset]
        | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        quantization_config: "BitsAndBytesConfig | None" = None,
        peft_config: "PeftConfig | None" = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = DPOConfig(f"{model_name}-DPO")

        if train_dataset is None:
            raise ValueError("`train_dataset` is required")
        elif isinstance(train_dataset, IterableDataset):
            # IterableDataset requires dispatch_batches=False because Accelerate's dispatch mode may try to concatenate
            # batches from multiple processes, leading to mismatch errors.
            if args.accelerator_config.dispatch_batches is True:
                logger.warning(
                    "You are using an `IterableDataset` for training with `dispatch_batches=True`. `dispatch_batches` "
                    "is forced to `False` when using an `IterableDataset`. To remove this warning, unset "
                    "`dispatch_batches` in `DPOConfig` or set it to `False`."
                )
            args.accelerator_config.dispatch_batches = False
        elif not isinstance(train_dataset, Dataset):
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
                    "You passed `model_init_kwargs` to the `DPOConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )
            if quantization_config is not None:
                raise ValueError(
                    "You passed `quantization_config` to the trainer, but your model is already instantiated. "
                    "Quantization can only be applied when the model is loaded from a model identifier (`str`). "
                    "Either pass the model as a model identifier, or omit `quantization_config`."
                )
        # Non-quantized models do not have the `is_loaded_in_{8,4}bit` attributes, whereas quantized models do
        _is_quantized_model = getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)
        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. In most cases you should omit `ref_model` and "
                "we'll initialize it to a copy of `model` for you."
            )

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

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # The model must agree with the tokenizer on the pad token from construction, so mirror it onto the model
        # configs.
        model.config.get_text_config().pad_token_id = self._tokenizer.pad_token_id
        model.generation_config.pad_token_id = self._tokenizer.pad_token_id

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

        elif is_peft_model(model) and ref_model is None:
            # If the model is a PEFT model with a pretrained adapter, we need to create a "ref" adapter that is a copy
            # of the "default" adapter, so that we can use it as the reference model during DPO training. Before PEFT
            # 0.20.0, only one adapter per model was supported when the LoRA config uses `target_parameters` [see
            # peft#3340, fixed in peft#3350], so in that case we skip the "ref" adapter and compute the reference log
            # probs with adapters disabled, i.e. with the base model. The fix only allows adapters targeting the same
            # parameters, which holds here since the "ref" adapter reuses the "default" config. `target_parameters`
            # itself was only added in PEFT 0.17.0, so the version check is bounded on both sides.
            default_config = model.peft_config["default"]
            if (
                isinstance(default_config, LoraConfig)
                and Version("0.17.0") <= Version(peft.__version__) < Version("0.20.0")
                and default_config.target_parameters
            ):
                logger.warning(
                    "PEFT<0.20.0 can't add a frozen reference adapter alongside one that uses `target_parameters` "
                    "(peft#3340], so the reference log probs are computed from the base model [adapters disabled]. "
                    "Upgrade to `peft>=0.20.0` to train against a copy of your adapter instead. If you wrapped the "
                    "model only to apply LoRA, pass a `peft_config` to the trainer instead; if you wrapped it "
                    "deliberately (pretrained adapter or custom init), note that the base model matches your adapter "
                    "only when it's freshly zero-initialized. If it is, this warning is safe to ignore."
                )
            else:
                model.add_adapter("ref", default_config)
                for name, param in model.named_parameters():
                    if ".default." in name:
                        ref_name = name.replace(".default.", ".ref.")
                        ref_param = model.get_parameter(ref_name)
                        ref_param.data.copy_(param.data)

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

        # Data collator
        self.padding_free = args.padding_free
        if self.padding_free:
            logger.warning(
                "`padding_free=True` is temporarily unavailable after a refactor and is currently disabled. Falling "
                "back to standard padding (`padding_free=False`). This feature is planned to return in a future "
                "update; for now, please set `padding_free=False` explicitly."
            )
            self.padding_free = False
        dataset_sample = next(iter(train_dataset))
        self._is_vision_dataset = "image" in dataset_sample or "images" in dataset_sample
        if self._is_vision_dataset and not self._is_vlm:
            raise ValueError(
                "The dataset appears to be vision-related (contains 'image' or 'images' keys), but the provided "
                "model does not seem to be a vision-language model. Please check your model and dataset."
            )
        if self._is_vision_dataset and args.max_length is not None and args.truncation_mode == "keep_end":
            raise ValueError(
                "truncation_mode='keep_end' is not supported for vision-language models. Image tokens reside "
                "inside the prompt portion of the sequence; depending on the example, keep_end may silently "
                "drop them, causing pixel_values to be forwarded to the model with no corresponding visual "
                "tokens in input_ids. Use truncation_mode='keep_start' (the default) or set max_length=None."
            )
        if self._is_vision_dataset and args.precompute_ref_log_probs:
            raise ValueError(
                "`precompute_ref_log_probs=True` is not supported for vision datasets. For vision-language "
                "models, all data processing is performed on the fly rather than upfront, and running a full "
                "forward pass of the reference model over the entire dataset is not supported for large "
                "multimodal models. Set `precompute_ref_log_probs=False`."
            )
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
                data_collator = DataCollatorForVisionPreference(
                    processor=processing_class,
                    max_length=args.max_length,
                    pad_to_multiple_of=args.pad_to_multiple_of,
                )
            else:
                data_collator = DataCollatorForPreference(
                    pad_token_id=self._tokenizer.pad_token_id,
                    max_length=args.max_length,
                    truncation_mode=args.truncation_mode,
                    pad_to_multiple_of=args.pad_to_multiple_of,
                )

        # Training arguments
        self.beta = args.beta
        self.precompute_ref_logps = args.precompute_ref_log_probs
        self.loss_types = args.loss_type  # args.loss_type is already a list
        self.loss_weights = args.loss_weights or [1.0] * len(self.loss_types)
        self.ld_alpha = args.ld_alpha
        self.f_divergence_type = args.f_divergence_type
        self.f_alpha_divergence_coef = args.f_alpha_divergence_coef
        # The f-DPO reparameterization leaves a β·log Z[x] term that only cancels in the chosen-rejected reward
        # difference, so the other losses have no valid f-divergence generalization.
        f_divergence_loss_types = {"sigmoid", "sigmoid_norm", "hinge", "ipo", "exo_pair", "robust", "discopop", "sft"}
        if self.f_divergence_type != "reverse_kl" and not set(self.loss_types) <= f_divergence_loss_types:
            raise ValueError(
                f"`f_divergence_type='{self.f_divergence_type}'` is only supported for the following loss types: "
                f"{sorted(f_divergence_loss_types)}. You provided {self.loss_types}. Use the default "
                "`f_divergence_type='reverse_kl'` with these losses."
            )
        self.label_smoothing = args.label_smoothing
        self.use_weighting = args.use_weighting
        if self.use_weighting and any(loss_type in {"aot", "aot_unpaired"} for loss_type in self.loss_types):
            raise NotImplementedError(
                "WPO-style weighting is not implemented for 'aot' or 'aot_unpaired' because those losses sort "
                "samples, which would misalign per-pair weights."
            )
        if "robust" in self.loss_types and not (0.0 <= self.label_smoothing < 0.5):
            logger.warning(
                "The `label_smoothing` parameter should lie in [0.0, 0.5) for the 'robust' loss. You provided "
                f"{self.label_smoothing}."
            )
        if "exo_pair" in self.loss_types and self.label_smoothing == 0.0:
            raise ValueError(
                "Label smoothing must be greater than 0.0 when using 'exo_pair' loss. The EXO paper recommends a "
                "value of 1e-3."
            )

        if is_peft_model(model) and isinstance(model.get_output_embeddings(), BaseTunerLayer):
            # The log-probabilities are computed by multiplying the hidden states by `lm_head.weight` directly, so an
            # adapter on the LM head would be ignored and never trained.
            raise ValueError(
                "Applying a PEFT adapter to `lm_head` is not supported: the log-probabilities are computed from "
                "`lm_head.weight` directly, so the adapter would never be trained. Remove `'lm_head'` from your "
                "`target_modules`."
            )

        # Dataset
        # Skip dataset preparation if it's a VLM, where preprocessing [e.g., image-to-pixel conversion] is too costly
        # and done on the fly instead.
        skip_prepare_dataset = self._is_vision_dataset
        if not skip_prepare_dataset:
            train_dataset = self._prepare_dataset(train_dataset, processing_class, args, "train")
            if eval_dataset is not None:
                if isinstance(eval_dataset, dict):
                    eval_dataset = {
                        key: self._prepare_dataset(dataset, processing_class, args, key)
                        for key, dataset in eval_dataset.items()
                    }
                else:
                    eval_dataset = self._prepare_dataset(eval_dataset, processing_class, args, "eval")

        # Transformers explicitly set use_reentrant=True in the past to silence a PyTorch warning, but the default was
        # never updated once PyTorch switched to recommending use_reentrant=False. Until that change lands upstream
        # [see https://github.com/huggingface/transformers/pull/43203] and is released [most likely in 5.0.0], we
        # default to the recommended non-reentrant behavior here, while preserving any user-provided value.
        if args.gradient_checkpointing and Version(transformers.__version__) < Version("5.0.0"):
            args.gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
            args.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
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

        # Reference model
        if ref_model is None:
            if is_peft_model(self.model) or (args.precompute_ref_log_probs and not self.is_fsdp_enabled):
                # If PEFT is used, the reference model is not needed since the adapter can be disabled to revert to the
                # initial model. If precompute_ref_log_probs is True, the reference model does not need to be kept in
                # memory during training. Under FSDP, the policy isn't sharded until `train`, so the precompute still
                # loads a reference model, freed once it is done.
                self.ref_model = None
            else:
                ref_model_init_kwargs = dict(args.model_init_kwargs or {})  # copy to avoid mutating model_init_kwargs
                if quantization_config is not None:
                    ref_model_init_kwargs["quantization_config"] = quantization_config
                # Distributed training requires device_map=None ["auto" fails]
                if self.args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                    ref_model_init_kwargs["device_map"] = None
                ref_model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
                ref_model_path = get_config_model_id(self.model.config)
                self.ref_model = create_model_from_path(ref_model_path, **ref_model_init_kwargs)
        else:
            self.ref_model = ref_model

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

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
        # `mean_logits` and `is_top1` feed the `logits/*` and `mean_token_accuracy` metrics, `log_sum_sq_probs` the WPO
        # weights
        outputs = ("log_probs", "entropy", "mean_logits", "is_top1")
        if self.use_weighting:
            outputs += ("log_sum_sq_probs",)
        add_fused_lm_head(self.model.get_base_model() if is_peft_model(self.model) else self.model, outputs=outputs)
        if self.ref_model is not None:
            add_fused_lm_head(self.ref_model)

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

        # Hash before the reference model is sharded, since DTensor parameters can't be hashed
        self._precompute_model_hash = (
            hash_module(self.ref_model or self.model) if args.precompute_ref_log_probs else None
        )

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            if is_peft_model(self.model):
                raise NotImplementedError(
                    "You passed `sync_ref_model=True` while using a PEFT model, which is currently not supported. "
                    "With PEFT, DPOTrainer does not keep a separate reference model in memory; instead, it recovers "
                    "reference behavior by temporarily disabling the adapter. As a result, there is no standalone "
                    "`ref_model` instance to synchronize. Use `sync_ref_model=False`, or opt for full fine-tuning if "
                    "you need a synced reference model. If you need `sync_ref_model` to work with PEFT, please open a "
                    "feature request at https://github.com/huggingface/trl/issues."
                )
            if args.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `sync_ref_model=True` together with `precompute_ref_log_probs=True`. "
                    "`precompute_ref_log_probs=True` assumes a fixed reference model, but with `sync_ref_model=True` "
                    "the reference model is periodically updated during training, making any precomputed reference "
                    "log-probs stale. Set `precompute_ref_log_probs=False` or disable `sync_ref_model`."
                )
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        # Reference forwards during precompute reuse a single DeepSpeed inference engine [see `_precompute_ref_logps`].
        self._precompute_engine = None
        if args.precompute_ref_log_probs:
            self.train_dataset = self._precompute_ref_logps(
                self.train_dataset,
                "train",
                self.args.precompute_ref_batch_size or self.args.per_device_train_batch_size,
            )
            if self.eval_dataset is not None:
                if isinstance(self.eval_dataset, dict):
                    self.eval_dataset = {
                        name: self._precompute_ref_logps(
                            dataset, name, self.args.precompute_ref_batch_size or self.args.per_device_eval_batch_size
                        )
                        for name, dataset in self.eval_dataset.items()
                    }
                else:
                    self.eval_dataset = self._precompute_ref_logps(
                        self.eval_dataset,
                        "eval",
                        self.args.precompute_ref_batch_size or self.args.per_device_eval_batch_size,
                    )
            if self.is_fsdp_enabled and ref_model is None:
                self.ref_model = None

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        args: DPOConfig,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with global_then_local_main_first():
            # Extract the prompt if needed
            first_example = next(iter(dataset))
            if "prompt" not in first_example:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Extracting prompt from {dataset_name} dataset"
                dataset = dataset.map(extract_prompt, **map_kwargs)

            # Add EOS token if needed: non-conversational only
            first_example = next(iter(dataset))
            if not is_conversational(first_example):
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Adding EOS to {dataset_name} dataset"

                def add_eos(example, eos_token):
                    if not example["chosen"].endswith(eos_token):
                        example["chosen"] = example["chosen"] + eos_token
                    if not example["rejected"].endswith(eos_token):
                        example["rejected"] = example["rejected"] + eos_token
                    return example

                dataset = dataset.map(add_eos, fn_kwargs={"eos_token": self._tokenizer.eos_token}, **map_kwargs)

            # Tokenize the dataset
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            def tokenize_fn(example, processing_class):
                tools = example.get("tools")
                tools = json.loads(tools) if isinstance(tools, str) else tools
                apply_chat_template_kwargs = {"tools": tools, **example.get("chat_template_kwargs", {})}
                output = {}
                if is_conversational(example):
                    prompt_ids = _tokenize(
                        processing_class,
                        example["prompt"],
                        add_generation_prompt=True,
                        **apply_chat_template_kwargs,
                    )["input_ids"]
                    prompt_chosen_ids = _tokenize(
                        processing_class,
                        example["prompt"] + example["chosen"],
                        **apply_chat_template_kwargs,
                    )["input_ids"]
                    prompt_rejected_ids = _tokenize(
                        processing_class,
                        example["prompt"] + example["rejected"],
                        **apply_chat_template_kwargs,
                    )["input_ids"]
                else:
                    prompt_ids = _tokenize(processing_class, example["prompt"])["input_ids"]
                    prompt_chosen_ids = _tokenize(processing_class, example["prompt"] + example["chosen"])["input_ids"]
                    prompt_rejected_ids = _tokenize(processing_class, example["prompt"] + example["rejected"])[
                        "input_ids"
                    ]

                # The completion starts where the tokenized prompt and prompt+completion diverge, which is not always
                # after the prompt (see `common_prefix_length`)
                prompt_len = min(
                    common_prefix_length(prompt_ids, prompt_chosen_ids),
                    common_prefix_length(prompt_ids, prompt_rejected_ids),
                )
                if prompt_len < len(prompt_ids):
                    logger.warning_once(
                        "The tokenized prompt is not a prefix of the tokenized prompt+completion, usually because the "
                        "chat template renders the prompt alone differently or because tokens merge across the "
                        "boundary. The completion starts where they diverge, and this end of the prompt is left out "
                        "of the training sequence: "
                        f"{processing_class.decode(prompt_ids[prompt_len:])!r}"
                        ". The model is trained on a context that differs from the one it sees at inference. To avoid "
                        "it, use a chat template that renders the prompt the same way in both cases, or end the "
                        "prompt on a token boundary. This warning is shown once, but it likely applies to every "
                        "example in the dataset."
                    )

                output["prompt_ids"] = prompt_ids[:prompt_len]
                output["chosen_ids"] = prompt_chosen_ids[prompt_len:]
                output["rejected_ids"] = prompt_rejected_ids[prompt_len:]
                return output

            dataset = dataset.map(
                tokenize_fn,
                fn_kwargs={"processing_class": processing_class},
                **map_kwargs,
            )

            # Drop examples whose prompt alone fills `max_length`: with `keep_start` truncation the collator would
            # remove every completion token, leaving no learning signal. `keep_end` keeps the completion end, so
            # nothing is dropped there.
            if args.max_length is not None and args.truncation_mode == "keep_start":
                if isinstance(dataset, Dataset):  # `IterableDataset.filter` does not support `desc`
                    map_kwargs["desc"] = f"Dropping fully truncated examples from {dataset_name} dataset"
                dataset = dataset.filter(lambda example: len(example["prompt_ids"]) < args.max_length, **map_kwargs)

        return dataset

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        # and "attention_mask").
        if self._signature_columns is None:
            if self._is_vision_dataset:
                self._signature_columns = [
                    "prompt",
                    "chosen",
                    "rejected",
                    "image",
                    "images",
                    "tools",
                    "chat_template_kwargs",
                ]
            else:
                self._signature_columns = [
                    "prompt_ids",
                    "chosen_ids",
                    "rejected_ids",
                    "ref_chosen_logps",
                    "ref_rejected_logps",
                ]

    def _precompute_ref_logps(self, dataset: Dataset, name: str, batch_size: int) -> Dataset:
        if isinstance(dataset, IterableDataset):
            raise ValueError(
                "`precompute_ref_log_probs=True` is not supported with IterableDataset. Please use a map-style "
                "Dataset or set `precompute_ref_log_probs=False`."
            )
        model_hash = self._precompute_model_hash
        # Both inputs are rank-dependent under distributed training (ZeRO-3 shards the model), so broadcast rank 0's
        # value so all ranks share one cache file.
        fingerprint = [Hasher.hash((dataset._fingerprint, model_hash))]
        broadcast_object_list(fingerprint, from_process=0)
        fingerprint = fingerprint[0]
        cache_file = dataset._get_cache_file_path(fingerprint)

        if os.path.exists(cache_file):
            return concatenate_datasets([dataset, Dataset.from_file(cache_file)], axis=1)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=False,
        )
        data_loader = self.accelerator.prepare(dataloader)

        # This runs before the parent class prepares the model in `train`, so with DeepSpeed the parameters are still
        # on CPU (ZeRO-1/2) and sharded (ZeRO-3). Wrap the model in an inference engine to place and gather them. Build
        # it once and reuse it across precompute passes (train, eval, and later `evaluate` calls)
        if self.ref_model is None and self.is_deepspeed_enabled:
            if self._precompute_engine is None:
                self._precompute_engine = prepare_deepspeed(self.model, self.accelerator)
            model = self._precompute_engine
        else:
            model = self.ref_model or self.model

        ref_chosen_logps = []
        ref_rejected_logps = []
        for padded_batch in tqdm(
            iterable=data_loader,
            desc=f"Computing reference log probs for {name} dataset",
            disable=bool(os.environ.get("TQDM_DISABLE", "")),
        ):
            ref_chosen_logp, ref_rejected_logp = self.compute_ref_log_probs(model, padded_batch)
            ref_chosen_logp, ref_rejected_logp = self.accelerator.gather_for_metrics(
                (ref_chosen_logp, ref_rejected_logp)
            )
            ref_chosen_logps.append(ref_chosen_logp.cpu())
            ref_rejected_logps.append(ref_rejected_logp.cpu())

        ref_chosen_logps = torch.cat(ref_chosen_logps)
        ref_rejected_logps = torch.cat(ref_rejected_logps)

        if self.accelerator.is_main_process:

            def add_ref_logps(batch, indices):
                return {
                    "ref_chosen_logps": ref_chosen_logps[indices],
                    "ref_rejected_logps": ref_rejected_logps[indices],
                }

            dataset.map(
                add_ref_logps,
                with_indices=True,
                batched=True,
                remove_columns=dataset.column_names,
                new_fingerprint=fingerprint,
                cache_file_name=cache_file,
                desc=f"Caching reference log probs for {name} dataset",
            )
        self.accelerator.wait_for_everyone()

        return concatenate_datasets([dataset, Dataset.from_file(cache_file)], axis=1)

    def compute_ref_log_probs(self, model, inputs):
        """Computes reference log probabilities for a single padded batch."""
        device = self.accelerator.device
        _non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
        model_kwargs = {k: v for k, v in inputs.items() if k not in _non_model_keys}
        model_kwargs["use_cache"] = False
        labels = inputs["input_ids"].masked_fill(inputs["completion_mask"] == 0, -100)

        adapter_context = contextlib.nullcontext()
        if self.ref_model is None and is_peft_model(self.model):
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            adapter_context = use_adapter(
                unwrapped_model, adapter_name="ref" if "ref" in unwrapped_model.peft_config else None
            )

        with (
            torch.no_grad(),
            disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs),
            adapter_context,
            self.accelerator.autocast(),
        ):
            ref_per_token_logps = model(**model_kwargs, labels=labels, fused_lm_head=True).log_probs

        shift_completion_mask = inputs["completion_mask"][..., 1:]
        # Prompt-learning PEFT prepends virtual tokens to the outputs; keep the positions of the real tokens
        ref_per_token_logps = ref_per_token_logps[:, -shift_completion_mask.size(1) :]
        if self.ld_alpha is None:
            ref_logps = ref_per_token_logps.sum(dim=1)
        else:
            comp_pos = shift_completion_mask.cumsum(dim=1)
            comp_lens = shift_completion_mask.sum(dim=1).long()
            chosen_lens, rejected_lens = comp_lens.chunk(2, dim=0)
            shared_lens = torch.minimum(chosen_lens, rejected_lens)
            shared_lens = torch.cat([shared_lens, shared_lens], dim=0).to(device)
            shared_mask = (comp_pos > 0) & (comp_pos <= shared_lens.unsqueeze(1))
            tail_mask = comp_pos > shared_lens.unsqueeze(1)
            shared_logps = (ref_per_token_logps * shared_mask).sum(dim=1)
            tail_logps = (ref_per_token_logps * tail_mask).sum(dim=1)
            ref_logps = shared_logps + self.ld_alpha * tail_logps

        return ref_logps.chunk(2, dim=0)

    def _compute_loss(self, model, inputs, return_outputs):
        mode = "train" if self.model.training else "eval"
        device = self.accelerator.device

        _non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
        model_kwargs = {k: v for k, v in inputs.items() if k not in _non_model_keys}
        model_kwargs["use_cache"] = False
        # MoE models: request router logits so the model returns `outputs.aux_loss`. VLM wrappers honor this only
        # as a forward kwarg (not from the model config), so it must be passed here.
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True
        completion_mask = inputs["completion_mask"]
        shift_completion_mask = completion_mask[..., 1:]
        labels = inputs["input_ids"].masked_fill(completion_mask == 0, -100)
        outputs = model(**model_kwargs, labels=labels, fused_lm_head=True)
        # Prompt-learning PEFT prepends virtual tokens to the outputs; keep the positions of the real tokens
        seq_len = shift_completion_mask.size(1)
        per_token_logps = outputs.log_probs[:, -seq_len:]
        if self.ld_alpha is None:
            logps = per_token_logps.sum(dim=1)  # sum over sequence length
        else:
            comp_pos = shift_completion_mask.cumsum(dim=1)
            comp_lens = shift_completion_mask.sum(dim=1).long()
            chosen_lens, rejected_lens = comp_lens.chunk(2, dim=0)
            shared_lens = torch.minimum(chosen_lens, rejected_lens)
            shared_lens = torch.cat([shared_lens, shared_lens], dim=0).to(device)
            shared_mask = (comp_pos > 0) & (comp_pos <= shared_lens.unsqueeze(1))  # shared: 1 <= pos <= shared_len
            tail_mask = comp_pos > shared_lens.unsqueeze(1)  # tail: pos > shared_len
            shared_logps = (per_token_logps * shared_mask).sum(dim=1)
            tail_logps = (per_token_logps * tail_mask).sum(dim=1)
            logps = shared_logps + self.ld_alpha * tail_logps
        chosen_logps, rejected_logps = logps.chunk(2, dim=0)  # batch is [chosen, rejected]

        if self.precompute_ref_logps:
            ref_chosen_logps, ref_rejected_logps = inputs["ref_chosen_logps"], inputs["ref_rejected_logps"]
        else:
            # The reference forward only needs logits for log-probs. Drop `output_router_logits` so the frozen
            # reference model does not materialize router logits and compute a discarded MoE aux loss.
            ref_model_kwargs = {k: v for k, v in model_kwargs.items() if k != "output_router_logits"}
            # When gradient checkpointing is enabled with use_reentrant=True (non default), calling the model inside a
            # torch.no_grad() block triggers a harmless PyTorch warning ("None of the inputs have requires_grad=True").
            # Temporarily disable checkpointing to avoid this warning during inference.
            with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
                if is_peft_model(self.model) and self.ref_model is None:
                    # When training a PEFT adapter, how we obtain the reference depends on the setup:
                    # - New adapter: disabling adapters yields the base model.
                    # - Re-training an existing adapter: an initial copy is loaded under the name "ref".
                    ref_model_unwrapped = self.accelerator.unwrap_model(self.model)
                    with use_adapter(
                        ref_model_unwrapped, adapter_name="ref" if "ref" in ref_model_unwrapped.peft_config else None
                    ):
                        ref_outputs = self.model(**ref_model_kwargs, labels=labels, fused_lm_head=True)
                else:
                    ref_outputs = self.ref_model(**ref_model_kwargs, labels=labels, fused_lm_head=True)
            ref_per_token_logps = ref_outputs.log_probs[:, -seq_len:]
            if self.ld_alpha is None:
                ref_logps = ref_per_token_logps.sum(dim=1)  # sum over sequence length
            else:
                # reuse comp_pos/shared_mask/tail_mask computed above (they depend only on completion_mask)
                ref_shared_logps = (ref_per_token_logps * shared_mask).sum(dim=1)
                ref_tail_logps = (ref_per_token_logps * tail_mask).sum(dim=1)
                ref_logps = ref_shared_logps + self.ld_alpha * ref_tail_logps
            ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)  # batch is [chosen, rejected]

        # Get the log ratios for the chosen and rejected responses
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps

        if self.f_divergence_type == "reverse_kl":  # standard DPO
            chosen_scores = chosen_logratios
            rejected_scores = rejected_logratios
        elif self.f_divergence_type == "forward_kl":
            # f'(t) = 1 - 1/t
            chosen_scores = 1 - torch.exp(-chosen_logratios)
            rejected_scores = 1 - torch.exp(-rejected_logratios)
        elif self.f_divergence_type == "js_divergence":
            # f'(t) = log(2t/(t+1)) = log 2 + logsigmoid(log t)
            chosen_scores = math.log(2) + F.logsigmoid(chosen_logratios)
            rejected_scores = math.log(2) + F.logsigmoid(rejected_logratios)
        elif self.f_divergence_type == "alpha_divergence":
            # alpha-divergence: f'(t) = (t^(α-1) - 1)/(α-1)
            if abs(self.f_alpha_divergence_coef - 1.0) < 1e-6:  # limit case f'(t) -> log(t), fall back to reverse_kl
                chosen_scores = chosen_logratios
                rejected_scores = rejected_logratios
            else:
                coef = 1.0 / (self.f_alpha_divergence_coef - 1.0)
                t_chosen = (self.f_alpha_divergence_coef - 1.0) * chosen_logratios
                t_rejected = (self.f_alpha_divergence_coef - 1.0) * rejected_logratios
                dtype = t_chosen.dtype
                # Clamp max so exp(.) stays representable after casting back
                clamp_max = {torch.float16: 11.0, torch.bfloat16: 80.0, torch.float32: 80.0}[dtype]
                t_chosen_float = torch.clamp(t_chosen.float(), max=clamp_max)
                t_rejected_float = torch.clamp(t_rejected.float(), max=clamp_max)
                chosen_scores = (torch.exp(t_chosen_float) - 1.0).to(dtype) * coef
                rejected_scores = (torch.exp(t_rejected_float) - 1.0).to(dtype) * coef
        else:
            raise ValueError(f"Unknown f_divergence_type: {self.f_divergence_type}")

        delta_score = chosen_scores - rejected_scores

        loss = 0.0
        for loss_type, loss_weight in zip(self.loss_types, self.loss_weights, strict=True):
            if loss_type == "sigmoid":
                per_sequence_loss = -F.logsigmoid(self.beta * delta_score)

            elif loss_type == "hinge":
                per_sequence_loss = torch.relu(1 - self.beta * delta_score)

            elif loss_type == "ipo":
                # IPO uses sequence-level log-prob differences; in code these are token-summed over the completion,
                # which makes the squared loss scale with completion length. We therefore normalize by the number of
                # completion tokens (average per token) to make β/loss comparable across variable lengths. This length
                # normalization is not explicitly discussed in the IPO paper; we confirmed this choice with the IPO
                # authors, and the results reported in the paper correspond to this normalized form.
                chosen_mask, rejected_mask = completion_mask.chunk(2, dim=0)
                chosen_avg_score = chosen_scores / chosen_mask.sum(dim=1).clamp(min=1.0)
                rejected_avg_score = rejected_scores / rejected_mask.sum(dim=1).clamp(min=1.0)
                ipo_delta = chosen_avg_score - rejected_avg_score
                # (Eq. 17) of the paper where beta is the regularization parameter for the IPO loss, denoted by τ.
                per_sequence_loss = (ipo_delta - 1 / (2 * self.beta)) ** 2

            elif loss_type == "exo_pair":
                # Implements EXO-pref from the paper https://huggingface.co/papers/2402.00856 (Eq. 16)
                # Minimize KL(p_fθ || p_rh) for K=2; p_fθ = softmax(βπ * (log πθ − log π_ref)) over {chosen, rejected}
                # p_rh = [(1−ε), ε]; expanded KL gives the weighted logsigmoid form below
                epsilon = torch.tensor(self.label_smoothing, device=device)
                qw = torch.sigmoid(self.beta * delta_score)
                log_qw = F.logsigmoid(self.beta * delta_score)
                log_pw = torch.log1p(-epsilon)
                ql = torch.sigmoid(-self.beta * delta_score)
                log_ql = F.logsigmoid(-self.beta * delta_score)
                log_pl = torch.log(epsilon)
                per_sequence_loss = qw * (log_qw - log_pw) + ql * (log_ql - log_pl)

            elif loss_type == "nca_pair":
                chosen_rewards = self.beta * chosen_logratios
                rejected_rewards = self.beta * rejected_logratios
                per_sequence_loss = (
                    -F.logsigmoid(chosen_rewards)
                    - 0.5 * F.logsigmoid(-chosen_rewards)
                    - 0.5 * F.logsigmoid(-rejected_rewards)
                )

            elif loss_type == "robust":
                clean_loss_term = -(1 - self.label_smoothing) * F.logsigmoid(self.beta * delta_score)
                flipped_loss_term = -self.label_smoothing * F.logsigmoid(-self.beta * delta_score)
                per_sequence_loss = (clean_loss_term - flipped_loss_term) / (1 - 2 * self.label_smoothing)

            elif loss_type == "bco_pair":
                chosen_rewards = self.beta * chosen_logratios
                rejected_rewards = self.beta * rejected_logratios
                per_sequence_loss = -F.logsigmoid(chosen_rewards) - F.logsigmoid(-rejected_rewards)

            elif loss_type == "sppo_hard":
                # In the paper (https://huggingface.co/papers/2405.00675), SPPO employs a soft probability approach,
                # estimated using the PairRM score. The probability calculation is conducted outside of the trainer
                # class. The version described here is the hard probability version, where P in Equation (4.7) of
                # Algorithm 1 is set to 1 for the winner and 0 for the loser.
                winner_margin_error = (chosen_logratios - 0.5 / self.beta) ** 2
                loser_margin_error = (rejected_logratios + 0.5 / self.beta) ** 2
                per_sequence_loss = winner_margin_error + loser_margin_error

            elif loss_type == "aot":
                logratios = chosen_logps - rejected_logps
                ref_logratios = ref_chosen_logps - ref_rejected_logps
                logratios_sorted, _ = torch.sort(logratios, dim=0)
                ref_logratios_sorted, _ = torch.sort(ref_logratios, dim=0)
                delta = logratios_sorted - ref_logratios_sorted
                per_sequence_loss = (
                    -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * delta) * self.label_smoothing
                )

            elif loss_type == "aot_unpaired":
                chosen_logratios_sorted, _ = torch.sort(chosen_logratios, dim=0)
                rejected_logratios_sorted, _ = torch.sort(rejected_logratios, dim=0)
                delta = chosen_logratios_sorted - rejected_logratios_sorted
                per_sequence_loss = (
                    -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * delta) * self.label_smoothing
                )

            elif loss_type == "apo_zero":
                # Eqn (7) of the APO paper (https://huggingface.co/papers/2408.06266)
                # Use this loss when you believe the chosen outputs are better than your model's default output
                # Increase chosen likelihood and decrease rejected likelihood
                losses_chosen = 1 - torch.sigmoid(self.beta * chosen_logratios)
                losses_rejected = torch.sigmoid(self.beta * rejected_logratios)
                per_sequence_loss = losses_chosen + losses_rejected

            elif loss_type == "apo_down":
                # Eqn (8) of the APO paper (https://huggingface.co/papers/2408.06266)
                # Use this loss when you believe the chosen outputs are worse than your model's default output.
                # Decrease chosen likelihood and decrease rejected likelihood more
                losses_chosen = torch.sigmoid(self.beta * chosen_logratios)
                losses_rejected = 1 - torch.sigmoid(self.beta * (chosen_logratios - rejected_logratios))
                per_sequence_loss = losses_chosen + losses_rejected

            elif loss_type == "discopop":
                # Eqn (5) of the DiscoPOP paper (https://huggingface.co/papers/2406.08414)
                logits = delta_score * self.beta
                # Modulate the mixing coefficient based on the log ratio magnitudes
                log_ratio_modulation = torch.sigmoid(logits / self.args.discopop_tau)
                logistic_component = -F.logsigmoid(logits)
                exp_component = torch.exp(-logits)
                # Blend between logistic and exponential component based on log ratio modulation
                per_sequence_loss = (
                    logistic_component * (1 - log_ratio_modulation) + exp_component * log_ratio_modulation
                )

            elif loss_type == "sft":
                chosen_mask, _ = shift_completion_mask.chunk(2, dim=0)
                chosen_per_token_logps, _ = per_token_logps.chunk(2, dim=0)
                batch_loss = -chosen_per_token_logps[chosen_mask.bool()].mean()
                batch_size = chosen_per_token_logps.size(0)
                # Implementation convenience: expand the scalar SFT loss to a per-sequence tensor so it matches the
                # shape of other losses; only the mean is used, so this is a no-op numerically.
                per_sequence_loss = batch_loss.expand(batch_size)

            elif loss_type == "sigmoid_norm":
                chosen_mask, rejected_mask = completion_mask.chunk(2, dim=0)
                chosen_avg_score = chosen_scores / chosen_mask.sum(dim=1).clamp(min=1.0)
                rejected_avg_score = rejected_scores / rejected_mask.sum(dim=1).clamp(min=1.0)
                delta = chosen_avg_score - rejected_avg_score
                per_sequence_loss = -F.logsigmoid(self.beta * delta)

            else:
                raise ValueError(
                    f"Unknown loss type: {loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'exo_pair', "
                    "'nca_pair', 'robust', 'bco_pair', 'sppo_hard', 'aot', 'aot_unpaired', 'apo_zero', 'apo_down', "
                    "'discopop', 'sft', 'sigmoid_norm']"
                )

            if self.use_weighting:
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                completion_lengths = shift_completion_mask.sum(dim=1).clamp_min(1)
                with torch.no_grad():
                    aligned_logps = (per_token_logps - outputs.log_sum_sq_probs[:, -seq_len:]) * shift_completion_mask
                mean_logps = aligned_logps.sum(dim=1) / completion_lengths
                weights = torch.exp(mean_logps)
                chosen_weights, rejected_weights = weights.chunk(2, dim=0)
                per_sequence_loss *= chosen_weights * rejected_weights

            loss += per_sequence_loss.mean() * loss_weight

        if self.aux_loss_enabled:
            aux_loss = outputs.aux_loss
            loss = loss + self.router_aux_loss_coef * aux_loss
            self._metrics[mode]["aux_loss"].append(self.accelerator.gather_for_metrics(aux_loss).mean().item())

        # Log the metrics
        # Entropy
        per_token_entropy = outputs.entropy[:, -seq_len:].detach()
        mask = shift_completion_mask
        entropy_sum = (per_token_entropy * mask).sum()
        total_tokens = mask.sum()

        # Gather counts across ranks and weight-average
        entropy_sum = self.accelerator.gather_for_metrics(entropy_sum).sum()
        total_tokens = self.accelerator.gather_for_metrics(total_tokens).sum()
        entropy = (entropy_sum / total_tokens).item() if total_tokens > 0 else 0.0
        self._metrics[mode]["entropy"].append(entropy)

        # Number of tokens
        if mode == "train":
            num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            self._total_train_tokens += num_tokens_in_batch // self._tp_size
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Average logits for chosen and rejected completions
        chosen_mean_logits, rejected_mean_logits = outputs.mean_logits[:, -seq_len:].chunk(2, dim=0)
        chosen_mask, rejected_mask = shift_completion_mask.chunk(2, dim=0)
        total_chosen_logits = chosen_mean_logits[chosen_mask.bool()].sum()
        total_chosen_tokens = chosen_mask.sum()
        total_rejected_logits = rejected_mean_logits[rejected_mask.bool()].sum()
        total_rejected_tokens = rejected_mask.sum()
        total_chosen_logits = self.accelerator.gather_for_metrics(total_chosen_logits).sum().item()
        total_chosen_tokens = self.accelerator.gather_for_metrics(total_chosen_tokens).sum().item()
        total_rejected_logits = self.accelerator.gather_for_metrics(total_rejected_logits).sum().item()
        total_rejected_tokens = self.accelerator.gather_for_metrics(total_rejected_tokens).sum().item()
        avg_chosen_logits = total_chosen_logits / total_chosen_tokens if total_chosen_tokens > 0 else 0.0
        avg_rejected_logits = total_rejected_logits / total_rejected_tokens if total_rejected_tokens > 0 else 0.0
        self._metrics[mode]["logits/chosen"].append(avg_chosen_logits)
        self._metrics[mode]["logits/rejected"].append(avg_rejected_logits)

        # Token accuracy for the chosen completions
        chosen_is_top1, _ = outputs.is_top1[:, -seq_len:].chunk(2, dim=0)
        correct_predictions = chosen_is_top1 & chosen_mask.bool()
        total_tokens = chosen_mask.sum()
        correct_tokens = correct_predictions.sum()
        correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
        total_tokens = self.accelerator.gather_for_metrics(total_tokens)
        total_sum = total_tokens.sum()
        accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
        self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        # Rewards for chosen and rejected completions
        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()
        agg_chosen_rewards = self.accelerator.gather(chosen_rewards)
        agg_rejected_rewards = self.accelerator.gather(rejected_rewards)
        self._metrics[mode]["rewards/chosen"].append(agg_chosen_rewards.mean().item())
        self._metrics[mode]["rewards/rejected"].append(agg_rejected_rewards.mean().item())

        # Reward accuracy
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        agg_reward_accuracies = self.accelerator.gather(reward_accuracies)
        self._metrics[mode]["rewards/accuracies"].append(agg_reward_accuracies.mean().item())

        # Reward margins
        margins = chosen_rewards - rejected_rewards
        agg_margins = self.accelerator.gather(margins)
        self._metrics[mode]["rewards/margins"].append(agg_margins.mean().item())

        # Average log probabilities for chosen and rejected completions
        self._metrics[mode]["logps/chosen"].append(self.accelerator.gather(chosen_logps).mean().item())
        self._metrics[mode]["logps/rejected"].append(self.accelerator.gather(rejected_logps).mean().item())

        if return_outputs:
            warnings.warn(
                "`return_outputs=True`, and the logits it passes to `compute_metrics`, are deprecated and will be removed "
                "in v2.0.0. The fused LM head does not build the logits, so they are recomputed with an extra forward pass.",
                FutureWarning,
                stacklevel=2,
            )
            with torch.no_grad():
                outputs = model(**model_kwargs)
        return (loss, outputs) if return_outputs else loss

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
        # `__init__` does, so that `evaluate` accepts the same dataset types as the trainer. `_prepare_dataset` is
        # idempotent: it skips datasets that are already tokenized. A `str` selects a dataset that was already prepared
        # at init time, so it's left untouched.
        if not self._is_vision_dataset and eval_dataset is not None and not isinstance(eval_dataset, str):
            # Full fine-tuning with no `ref_model` uses `self.model` as the reference, which is only valid before
            # training. After a step (`global_step > 0`) it's the trained policy, so we can't precompute a correct
            # reference here. (PEFT is exempt: the reference is recovered by disabling the adapter.) Checked before
            # tokenizing so we fail fast.
            if (
                self.precompute_ref_logps
                and self.ref_model is None
                and not is_peft_model(self.model)
                and self.state.global_step > 0
            ):
                raise ValueError(
                    "Cannot compute reference log-probs for a dataset passed to `evaluate()` after training has "
                    "started, because `precompute_ref_log_probs=True` and no `ref_model` was provided (full "
                    "fine-tuning). In this setup the reference model is not kept in memory, so it is only available "
                    "before training. Provide this dataset as `eval_dataset` at initialization, pass an explicit "
                    "`ref_model`, or set `precompute_ref_log_probs=False`."
                )
            if isinstance(eval_dataset, dict):
                eval_dataset = {
                    key: self._prepare_dataset(dataset, self.processing_class, self.args, key)
                    for key, dataset in eval_dataset.items()
                }
            else:
                eval_dataset = self._prepare_dataset(eval_dataset, self.processing_class, self.args, "eval")
            # With `precompute_ref_log_probs`, `_compute_loss` reads the reference log-probs from the batch, so they
            # must be precomputed here as well, mirroring `__init__`.
            if self.precompute_ref_logps:
                batch_size = self.args.precompute_ref_batch_size or self.args.per_device_eval_batch_size
                if isinstance(eval_dataset, dict):
                    eval_dataset = {
                        name: self._precompute_ref_logps(dataset, name, batch_size)
                        for name, dataset in eval_dataset.items()
                    }
                else:
                    eval_dataset = self._precompute_ref_logps(eval_dataset, "eval", batch_size)
            # Call `super().evaluate()` once per split ourselves instead of handing the whole dict to it: `Trainer.evaluate`
            # would otherwise recurse into `self.evaluate` per split, re-entering this override on an already-prepared dataset.
            if isinstance(eval_dataset, dict):
                metrics = {}
                for name, dataset in eval_dataset.items():
                    metrics.update(
                        super().evaluate(
                            eval_dataset=dataset,
                            ignore_keys=ignore_keys,
                            metric_key_prefix=f"{metric_key_prefix}_{name}",
                        )
                    )
                return metrics
        return super().evaluate(
            eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        try:
            return self._compute_loss(model, inputs, return_outputs)
        except ValueError as e:
            if "Image features and image tokens do not match" in str(e) and self.args.max_length is not None:
                raise ValueError(
                    f"The current `max_length` ({self.args.max_length}) is too short and causes image placeholder "
                    f"tokens in `input_ids` to be truncated, while the corresponding image features remain intact. "
                    f"Please increase `max_length` or set it to `None` to disable truncation."
                ) from e
            raise

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

    # During eval, Trainer calls prediction_step. If no labels are present in the inputs, it only runs forward and
    # returns logits. We override prediction_step to force compute_loss, because this trainer doesn't involve labels.
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: list[str] | None = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            if prediction_loss_only:
                loss = self.compute_loss(model, inputs, return_outputs=False)
                logits, labels = None, None
            else:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                logits, labels = outputs.logits, inputs["input_ids"]
        return loss, logits, labels

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
class UnslothDPOTrainer(_UnslothDPOTrainer):
    """
    
    Trainer for Direct Preference Optimization (DPO) method. This algorithm was initially proposed in the paper [Direct
    Preference Optimization: Your Language Model is Secretly a Reward Model](https://huggingface.co/papers/2305.18290).
    This class is a wrapper around the [`~transformers.Trainer`] class and inherits all of its attributes and methods.

    Example:

    ```python
    >>> from trl import DPOTrainer
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

    >>> trainer = DPOTrainer(
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
        ref_model ([`~transformers.PreTrainedModel`], *optional*):
            Reference model used to compute the reference log probabilities.

            - If provided, this model is used directly as the reference policy.
            - If `None`, the trainer will automatically use the initial policy corresponding to `model`, i.e. the model
              state before DPO training starts.
        args ([`DPOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Function to use to form a batch from a list of elements of the processed `train_dataset` or `eval_dataset`.
            Will default to [`~trainer.dpo_trainer.DataCollatorForPreference`] if the model is a language model and
            [`~trainer.dpo_trainer.DataCollatorForVisionPreference`] if the model is a vision-language model. Custom
            collators must truncate sequences before padding; the trainer does not apply post-collation truncation.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. This trainer supports both [language modeling](#language-modeling) type and
            [prompt-completion](#prompt-completion) type. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).

            When `train_dataset` is an [`~datasets.IterableDataset`] (e.g. a streaming dataset), `max_steps` must be
            set in the training arguments, since its length cannot be inferred and the total number of training steps
            is required to bound the training loop and configure the learning rate scheduler.
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`], [`~datasets.DatasetDict`], [`~datasets.IterableDatasetDict`] or `dict[str, Dataset | IterableDataset]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`] or [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
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
        quantization_config ([`~transformers.BitsAndBytesConfig`], *optional*):
            Quantization configuration used when loading the model from a model identifier. Combine with `peft_config`
            for QLoRA training. Ignored if the model is already instantiated.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    
    """
    def __init__(
        self,
        model,
        ref_model = None,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        processing_class = None,
        compute_metrics = None,
        callbacks = None,
        quantization_config = None,
        peft_config = None,
        **kwargs
    ):
        if args is None: args = UnslothDPOConfig()
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
        _unsloth_model_msl = getattr(model, 'max_seq_length', None)
        if isinstance(_unsloth_model_msl, int) and _unsloth_model_msl > 0 and hasattr(args, 'max_length'):
            if args.max_length is None or args.max_length > _unsloth_model_msl:
                print('Unsloth: `max_length = ' + str(args.max_length) + '` exceeds the model max_seq_length of ' + str(_unsloth_model_msl) + ', so it is reduced to ' + str(_unsloth_model_msl) + '.')
                args.max_length = _unsloth_model_msl
                _unsloth_mpl = getattr(args, 'max_prompt_length', 0)
                if (_unsloth_mpl is None and args.max_length <= 128) or (_unsloth_mpl is not None and _unsloth_mpl >= args.max_length):
                    args.max_prompt_length = args.max_length // 2
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
        PatchRLStatistics('dpo_trainer', other_metrics)
        if hasattr(train_dataset, 'column_names'):
            column_names = set(train_dataset.column_names)
            check = ['chosen', 'rejected', 'prompt', 'chosen_input_ids', 'chosen_attention_mask',
                     'chosen_labels', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels',
                     'prompt_input_ids', 'prompt_attention_mask']
            if all(x in column_names for x in check):
                train_dataset = train_dataset.remove_columns(['chosen', 'rejected', 'prompt'])
            del check, column_names
        if hasattr(train_dataset, 'column_names'):
            column_names = set(train_dataset.column_names)
            is_dpo_dataset = ({'chosen', 'rejected'}.issubset(column_names) or
                              {'prompt_input_ids', 'chosen_input_ids', 'rejected_input_ids'}.issubset(column_names))
            if is_dpo_dataset and isinstance(data_collator, TransformersDataCollatorForLanguageModeling):
                data_collator = None
            del is_dpo_dataset, column_names
        from trl.trainer.dpo_trainer import DataCollatorForPreference
        if not hasattr(DataCollatorForPreference, '_unsloth_vision_keys_patch'):
            _old_dpo_collator_torch_call = DataCollatorForPreference.torch_call
        
            def _unsloth_dpo_torch_call(self, examples):
                output = _old_dpo_collator_torch_call(self, examples)
                import torch as _unsloth_torch
                try:
                    from trl.trainer.utils import pad as _unsloth_trl_pad
                except Exception:
                    _unsloth_trl_pad = None
                for _k in ('pixel_position_ids', 'image_position_ids', 'mm_token_type_ids'):
                    if not all(_k in example for example in examples):
                        continue
                    _is_position_key = _k.endswith('position_ids')
                    _padding_value = -1 if _is_position_key else 0
                    _padding_side = 'right' if _is_position_key else 'left'
                    _values = [_unsloth_torch.as_tensor(example[_k]) for example in examples]
                    try:
                        if _unsloth_trl_pad is not None:
                            output[_k] = _unsloth_trl_pad(_values, padding_value=_padding_value, padding_side=_padding_side)
                        else:
                            from torch.nn.utils.rnn import pad_sequence as _unsloth_pad_sequence
                            output[_k] = _unsloth_pad_sequence(_values, batch_first=True, padding_value=_padding_value)
                    except Exception:
                        from torch.nn.utils.rnn import pad_sequence as _unsloth_pad_sequence
                        output[_k] = _unsloth_pad_sequence(_values, batch_first=True, padding_value=_padding_value)
                return output
        
            DataCollatorForPreference.torch_call = _unsloth_dpo_torch_call
            DataCollatorForPreference._unsloth_vision_keys_patch = True
        
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
            ref_model = ref_model,
            args = args,
            data_collator = data_collator,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            processing_class = processing_class,
            compute_metrics = compute_metrics,
            callbacks = callbacks,
            quantization_config = quantization_config,
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

