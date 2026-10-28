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
from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback
from unsloth_zoo.temporary_patches.common import torch_compile
from unsloth_zoo.temporary_patches.common import _maybe_compile
import functools
import transformers
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
from trl.trainer.grpo_trainer import (Any, AutoModelForSequenceClassification, AutoProcessor, AutoTokenizer, BaseTunerLayer, BitsAndBytesConfig, Callable, CommitScheduler, DataLoader, Dataset, DatasetCard, DatasetCardData, DatasetDict, DistributedBackend, EnvironmentFactory, GRPOConfig, GRPOTrainer, GenerationConfig, IterableDataset, IterableDatasetDict, LoraConfig, PREFIX_CHECKPOINT_DIR, Path, PeftConfig, PeftModel, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, RepeatSampler, RewardFunc, RolloutFunc, Sampler, SyncRefModelCallback, TrainerCallback, VLLMGeneration, Version, _BaseTrainer, _SUPPORTS_RESPONSE_TEMPLATE, add_fused_lm_head, add_response_schema, apply_chat_template, asyncio, atexit, copy, create_model_from_path, create_repo, defaultdict, deque, disable_dropout_in_model, disable_gradient_checkpointing, gather, gather_object, get_callable_name, get_config_model_id, get_peft_model, get_training_chat_template, identity, inspect, is_async_callable, is_chat_template_prefix_preserving, is_conversational, is_jmespath_available, is_peft_available, is_peft_model, is_rich_available, json, logger, math, nanmax, nanmin, nanstd, nn, np, os, pad, parse_response, pd, peft, pkg_resources, prepare_deepspeed, prepare_fsdp, prepare_multimodal_messages, print_prompt_completions_sample, profiling_context, profiling_decorator, repeat_iterable_dataset, set_seed, shuffle_sequence_dict, shutdown_event_loop_in_daemon, split_pixel_values_by_grid, split_tensor_dict, start_event_loop_in_daemon, strip_images_from_messages, supports_tool_calling, sys, textwrap, time, torch, transformers, unsplit_pixel_values_by_grid, unwrap_model_for_generation, use_adapter, warnings, AutoModelForSequenceClassification, AutoProcessor, AutoTokenizer, BaseTunerLayer, BitsAndBytesConfig, Callable, CommitScheduler, Dataset, DatasetCard, DatasetCardData, DatasetDict, DistributedBackend, EnvironmentFactory, GRPOConfig, GRPOTrainer, GenerationConfig, IterableDataset, IterableDatasetDict, LoraConfig, PeftConfig, PeftModel, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, RepeatSampler, RewardFunc, RolloutFunc, Sampler, SyncRefModelCallback, TrainerCallback, VLLMGeneration, Version, add_fused_lm_head, add_response_schema, atexit, copy, create_model_from_path, create_repo, defaultdict, deque, disable_dropout_in_model, gather, get_callable_name, get_config_model_id, get_peft_model, get_training_chat_template, identity, inspect, is_async_callable, is_chat_template_prefix_preserving, is_jmespath_available, is_peft_available, is_peft_model, logger, nn, np, os, pad, pd, peft, pkg_resources, prepare_deepspeed, prepare_fsdp, repeat_iterable_dataset, set_seed, shutdown_event_loop_in_daemon, start_event_loop_in_daemon, supports_tool_calling, sys, time, torch, transformers, warnings, Version, copy, gather, is_conversational, np, os, pad, parse_response, profiling_context, time, torch, transformers, Any, apply_chat_template, copy, disable_gradient_checkpointing, gather, gather_object, inspect, is_conversational, math, nanmax, nanmin, nanstd, nn, np, os, pad, pd, peft, prepare_multimodal_messages, torch, use_adapter, gather, np, os, pad, profiling_context, torch, transformers, unwrap_model_for_generation, Any, copy, gather, math, nn, np, os, pad, time, torch, transformers, warnings, Any, np, profiling_decorator, shuffle_sequence_dict, split_pixel_values_by_grid, split_tensor_dict, torch, unsplit_pixel_values_by_grid, PeftModel, PreTrainedModel, is_peft_available, logger, os, peft, torch, GRPOTrainer, gather, inspect, nanmax, nanmin, np, os, pad, time, torch)


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
            "triton.enable_persistent_tma_matmul": False,
            "cuda.cutlass_epilogue_fusion_enabled": (lambda caps: bool(caps) and len(caps) == torch.cuda.device_count() and min(c[0] for c in caps) >= 9)(__import__("unsloth.device_type", fromlist = ["cuda_visible_capabilities"]).cuda_visible_capabilities()),
            "cuda.cutlass_tma_only": (lambda caps: bool(caps) and len(caps) == torch.cuda.device_count() and min(c[0] for c in caps) >= 9)(__import__("unsloth.device_type", fromlist = ["cuda_visible_capabilities"]).cuda_visible_capabilities()),
            "cuda.compile_opt_level"              : "-O2",
            "cuda.enable_cuda_lto"                : True,
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
def _unsloth_grpo_autocast_forced_float32(self):
    # Stamped by from_pretrained: UNSLOTH_FORCE_FLOAT32 is process wide, so a model loaded after this trainer was built would answer for it here.
    forced = getattr(getattr(self, "model", None), "_unsloth_forced_float32", None)
    if forced is None:
        forced = os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1"
    return forced

def _unsloth_grpo_autocast(self):
    """Decide the GRPO autocast once and latch it on the trainer. ACCELERATE_MIXED_PRECISION is process wide, so a trainer built later but run first would hand this trainer its precision; args belongs to this trainer."""
    if not hasattr(self, "_autocast_enabled"):
        args = getattr(self, "args", None)
        precision = getattr(args, "mixed_precision", None)
        use_bf16 = getattr(args, "bf16", None)
        use_fp16 = getattr(args, "fp16", None)
        if not isinstance(precision, str):
            # transformers < 5 has no args.mixed_precision, but rl.py sets the fp16 / bf16 flags on this same args for every branch it takes.
            if isinstance(use_bf16, bool) and isinstance(use_fp16, bool):
                precision = "bf16" if use_bf16 else ("fp16" if use_fp16 else "no")
            else:
                precision = os.environ.get("ACCELERATE_MIXED_PRECISION", "fp16")
        self._autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        # "no" is a real value: full finetuning and an explicit float32 load both set it, and reading it as bfloat16 raises on a T4 or V100.
        self._autocast_enabled = precision != "no"
        self._autocast_force_float32 = False
        forced = _unsloth_grpo_autocast_forced_float32(self)
        if forced and precision != "bf16":
            # Gemma3 / gpt-oss set "no" but still want float16 autocast; a trainer already on bf16 keeps it, since float16 is what the forced list avoids.
            self._autocast_dtype = torch.float16
            self._autocast_enabled = True
            self._autocast_force_float32 = True

    return self._autocast_enabled, self._autocast_dtype

def _unsloth_grpo_autocast_unscored(self):
    """Latch for a step that ran no scoring forward. Where the ordinary latch would force float16 autocast on a
    forced float32 model (Gemma3 on a T4) without vLLM, record no loss autocast instead: nothing latched there
    before the TRL old-logprob gate came back, so zoo trained it without autocast. A forced-family trainer on
    bf16 (full finetuning) keeps its bf16 latch. Recorded on the trainer rather than left to zoo's
    process-wide fallback, which a later model or trainer may have changed. Under vLLM the old-policy
    forward always ran and latched, so it latches as before."""
    fresh = not hasattr(self, "_autocast_enabled")
    latched = _unsloth_grpo_autocast(self)
    if (
        fresh
        and not getattr(self, "use_vllm", False)
        and getattr(self, "_autocast_force_float32", False)
    ):
        self._autocast_dtype = torch.float16
        self._autocast_enabled = False
        latched = (self._autocast_enabled, self._autocast_dtype)
    return latched

def _unsloth_grpo_autocast_kwargs(self, device_type = DEVICE_TYPE_TORCH):
    """torch.amp.autocast kwargs for GRPO generation."""
    enabled, dtype = _unsloth_grpo_autocast(self)
    if not getattr(self, "_autocast_force_float32", False) and torch.is_autocast_enabled(
        device_type
    ):
        # Already inside an autocast: inherit its dtype by omitting the key, since autocast passes whatever it gets to set_autocast_dtype.
        return {"enabled": enabled}
    return {"enabled": enabled, "dtype": dtype}

def _unsloth_grpo_support_mismatch(self):
    """Why kept-set scoring (UNSLOTH_GRPO_TRUNCATED_SUPPORT) cannot score this trainer, or [] when it can.

    Each scoring call rebuilds the top_k / top_p / min_p kept set from its own logits and always keeps the
    scored token. That is the sampler's set only while the scored weights are the rollout's. A reference
    model (beta != 0), or old logps TRL keeps because generation is not aligned to the optimizer step
    (gradient_accumulation_steps % (steps_per_generation * num_iterations) != 0), keeps another set: a
    token outside it gets a denominator that depends on the scored token, so the implied probabilities do
    not sum to 1 and the PPO ratio / KL inputs are wrong. A shared set would need the sampler's kept ids.
    """
    reasons = []
    if getattr(self, "_unsloth_vllm_sync_skip", False):
        reasons.append("vLLM weight sync is skipped, so rollouts use the initial weights")
    beta = getattr(self, "beta", 0.0)
    if beta is not None and float(beta) != 0.0:
        reasons.append(f"beta={beta} scores a reference model")
    args = getattr(self, "args", None)
    accumulation = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
    steps_per_generation = int(getattr(args, "steps_per_generation", None) or accumulation)
    num_iterations = getattr(self, "num_iterations", None)
    if num_iterations is None:
        num_iterations = getattr(args, "num_iterations", 1)
    num_iterations = int(num_iterations or 1)
    if accumulation % (steps_per_generation * num_iterations) != 0:
        reasons.append(
            f"num_iterations={num_iterations} / steps_per_generation={steps_per_generation} "
            f"reuse rollouts across optimizer steps"
        )
    return reasons

def _unsloth_grpo_refuse_unscored_support(self):
    """With compute_efficient no old / ref logps are scored here, so the gate below in
    _get_per_token_logps_and_entropies never runs, and the zoo loss scores the current policy over
    its own kept set against vLLM's logprobs. When that set cannot be the sampler's (an unsynced
    vLLM sampler, a reference model, reused rollouts), mark the trainer refused, which the zoo loss
    honours by scoring the full vocabulary."""
    if getattr(self, "_unsloth_truncated_support_refused", False):
        return
    try:
        from unsloth_zoo.rl_replacements import grpo_truncated_support_kwargs
        if not grpo_truncated_support_kwargs(self):
            return
    except Exception:
        return
    mismatch = _unsloth_grpo_support_mismatch(self)
    if mismatch:
        import warnings
        warnings.warn(
            "Unsloth: not scoring over the rollouts' top_k / top_p / min_p kept set "
            "(UNSLOTH_GRPO_TRUNCATED_SUPPORT=1, or automatic for truncated vLLM rollouts "
            "feeding the IS correction or off-policy mask) because "
            f"{', '.join(mismatch)}: the scored policy is not the sampler, so its top_k / top_p / "
            "min_p kept set differs from vLLM's and the PPO ratio would be unnormalised. "
            "Scoring over the full vocabulary instead.",
            stacklevel = 2,
        )
        self._unsloth_truncated_support_refused = True

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

def _unsloth_get_mm_token_id(processing_class, attr_name, token):
    tokenizer = getattr(processing_class, "tokenizer", processing_class)
    token_id = getattr(processing_class, attr_name, None)
    if token_id is None:
        token_id = getattr(tokenizer, attr_name, None)

    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if token_id is None and convert_tokens_to_ids is not None:
        token_id = convert_tokens_to_ids(token)

    if type(token_id) is int and token_id >= 0:
        if token_id != getattr(tokenizer, "unk_token_id", None):
            return token_id
    return None

def _unsloth_fix_mm_token_type_ids(
    processing_class, input_ids, mm_token_type_ids = None, completion_ids = None
):
    image_token_id = _unsloth_get_mm_token_id(
        processing_class, "image_token_id", "<|image_pad|>"
    )
    video_token_id = _unsloth_get_mm_token_id(
        processing_class, "video_token_id", "<|video_pad|>"
    )

    if image_token_id is not None or video_token_id is not None:
        rebuilt = input_ids.new_zeros(input_ids.shape)
        if image_token_id is not None:
            rebuilt = rebuilt.masked_fill(input_ids == image_token_id, 1)
        if video_token_id is not None:
            rebuilt = rebuilt.masked_fill(input_ids == video_token_id, 2)
        return rebuilt

    if (
        mm_token_type_ids is not None
        and completion_ids is not None
        and mm_token_type_ids.shape[0] == input_ids.shape[0]
        and mm_token_type_ids.shape[1] + completion_ids.shape[1] == input_ids.shape[1]
    ):
        return torch.cat(
            [mm_token_type_ids, mm_token_type_ids.new_zeros(completion_ids.shape)],
            dim = 1,
        )
    return mm_token_type_ids

def _unsloth_grpo_accumulation_steps(trainer):
    """Gradient accumulation divisor for the GRPO loss. 1 whenever the model is not training.

    TRL divides by `current_gradient_accumulation_steps` in train mode and by 1.0 in eval, since
    an eval pass accumulates nothing. Trainer sets that attribute inside the training loop and
    never clears it, so an in-training evaluation still sees the training window's size and
    eval_loss comes back that many times too small. Read off `model.training` like TRL does
    rather than a trainer flag, because that is what the eval loop actually switches.
    The attribute is also missing when evaluate() runs standalone (#2464), hence the default.
    """
    model = getattr(trainer, "model", None)
    if model is not None and not getattr(model, "training", True):
        return 1
    return getattr(trainer, "current_gradient_accumulation_steps", 1)

def _unsloth_grpo_is_metric_values(delta, flat_is_ratio, mask, sequence_level):
    # Inputs are zero-filled outside the mask; TRL reduces over masked tokens only, else min ratio reads 0.
    if mask is None or delta.shape != mask.shape or flat_is_ratio.shape != mask.shape:
        return delta.reshape(-1), flat_is_ratio.reshape(-1)
    keep = mask.to(torch.bool)
    if not sequence_level:
        return delta[keep], flat_is_ratio[keep]
    # A row with no kept tokens is exp(0) = 1, as in TRL.
    counts = keep.sum(dim = -1)
    per_row = torch.where(
        counts > 0,
        (flat_is_ratio * keep).sum(dim = -1) / counts.clamp(min = 1),
        torch.ones_like(counts, dtype = flat_is_ratio.dtype),
    )
    return delta[keep], per_row

def _unsloth_grpo_is_sequence_level(args_mode):
    # One IS ratio per sequence: TRL's sequence_* and unsloth_zoo's geometric_* (verl seq_mean_k1),
    # with the UNSLOTH_GRPO_VLLM_IS_MODE override unsloth_zoo's grpo_accumulated_loss applies.
    import os as _os
    mode = _os.environ.get("UNSLOTH_GRPO_VLLM_IS_MODE", "") or args_mode or "token_truncate"
    return str(mode).startswith(("sequence_", "geometric_"))

def _unsloth_grpo_sample_weight_mode():
    import os

    exclude = os.environ.get("UNSLOTH_GRPO_ZERO_VARIANCE_EXCLUDE", "0") == "1"
    aggregation = os.environ.get("UNSLOTH_GRPO_LOSS_AGGREGATION", "").strip().lower()
    if aggregation not in ("", "prompt"):
        raise ValueError(
            f"Unsloth: UNSLOTH_GRPO_LOSS_AGGREGATION must be 'prompt' or unset, got {aggregation}."
        )
    if not exclude and aggregation != "prompt":
        return None
    return exclude, aggregation == "prompt"

def _unsloth_grpo_rollout_logps_as_old(
    trainer,
    sampling_per_token_logps,
    completion_mask,
    tool_mask = None,
):
    """UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD=1: use the rollout engine's logprobs (vLLM, or a
    rollout_func) as pi_old and skip the old-policy forward, as verl rollout_correction
    bypass_mode (loss_type ppo_clip) and vime --use-rollout-logprobs do. The PPO ratio is then
    pi_theta / pi_rollout and its clip (plus the dual clip `delta`) is the correction, so no
    vLLM importance weight is applied on top. True when this generation batch is marked."""
    import os

    if os.environ.get("UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD", "0") != "1":
        return False

    def _skip(reason):
        trainer._unsloth_rollout_logps_as_old_fallbacks = (
            getattr(trainer, "_unsloth_rollout_logps_as_old_fallbacks", 0) + 1
        )
        warned = getattr(trainer, "_unsloth_rollout_logps_as_old_warned", None)
        if not isinstance(warned, set):
            warned = set()
            trainer._unsloth_rollout_logps_as_old_warned = warned
        if reason not in warned:
            warned.add(reason)
            import warnings
            warnings.warn(
                f"Unsloth: UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD=1 is not applied to this batch: "
                f"{reason}. The trainer scores pi_old itself instead.",
                stacklevel = 2,
            )
        return False

    reason = None
    if sampling_per_token_logps is None:
        reason = "there are no rollout logprobs (needs use_vllm=True or a rollout_func)"
    elif getattr(trainer.args, "loss_type", None) == "vespo":
        reason = "loss_type='vespo' weights by the vLLM importance ratio itself"
    elif not hasattr(trainer.args, "loss_type"):
        reason = "this TRL version predates loss_type"
    else:
        try:
            from unsloth_zoo.rl_replacements import grpo_rollout_logps_as_old  # noqa: F401
        except Exception:
            reason = "it needs a newer unsloth_zoo: pip install -U unsloth_zoo"
    if reason is None:
        loss_mask = completion_mask.to(torch.bool)
        if tool_mask is not None:
            loss_mask = loss_mask & tool_mask.to(device = loss_mask.device, dtype = torch.bool)
        if not bool(torch.isfinite(sampling_per_token_logps[loss_mask]).all()):
            reason = "some rollout logprobs on loss tokens are not finite"
    accelerator = getattr(trainer, "accelerator", None)
    if getattr(accelerator, "num_processes", 1) > 1:
        # One reduction on every rank, before any of them returns: under FSDP / ZeRO-3 the
        # old-policy forward a fallback runs is collective, so all ranks take the same branch.
        eligible = torch.tensor(
            [int(reason is None)], dtype = torch.int32, device = completion_mask.device
        )
        if not bool(accelerator.gather(eligible).min()) and reason is None:
            reason = "another process cannot use its rollout logprobs for this batch"
    if reason is not None:
        return _skip(reason)
    trainer._unsloth_rollout_logps_as_old_batches = (
        getattr(trainer, "_unsloth_rollout_logps_as_old_batches", 0) + 1
    )
    return True

def _unsloth_trl_masks_unscorable():
    try:
        import trl
        from packaging.version import Version
        return Version(trl.__version__) >= Version("1.6.0")
    except Exception:
        return True

def _unsloth_grpo_sample_weight(
    trainer, mode, rewards_per_func, completion_mask, tool_mask, process_slice
):
    """Per-completion loss weights normalised over the whole generation batch (all processes),
    or None when off. ScaleRL (arXiv 2510.13786): zero-variance filtering drops groups whose
    rewards are all equal from the loss and its token count; prompt-level aggregation averages
    tokens within each group, then over groups. Training batches only."""
    settings = _unsloth_grpo_sample_weight_mode()
    if settings is None or mode != "train":
        return None
    exclude, prompt_level = settings
    loss_type = getattr(trainer.args, "loss_type", None)
    if (prompt_level and loss_type == "luspo") or (
        not prompt_level and loss_type not in ("bnpo", "dapo", "cispo", "vespo")
    ):
        if not getattr(trainer, "_unsloth_sample_weight_warned", False):
            import warnings
            warnings.warn(
                f"Unsloth: zero-variance exclusion alone needs a token-normalised loss_type "
                f"(bnpo, dapo, cispo, vespo) and prompt-level aggregation does not support "
                f"luspo; ignored for loss_type={loss_type}.",
                stacklevel = 2,
            )
            trainer._unsloth_sample_weight_warned = True
        return None
    num_generations = trainer.num_generations
    loss_mask = completion_mask if tool_mask is None else completion_mask * tool_mask
    lengths = trainer.accelerator.gather(loss_mask.sum(dim = 1).to(torch.float32))

    def nan_variance(x, dim):
        # TRL's nanstd squared: Bessel corrected, nan (never "zero") with fewer than two values.
        centred = x - torch.nanmean(x, dim = dim, keepdim = True)
        count = (~torch.isnan(x)).sum(dim = dim, keepdim = True)
        variance = torch.nansum(centred * centred, dim = dim, keepdim = True) / (count - 1).clamp(min = 1)
        return torch.where(count > 1, variance, torch.full_like(variance, torch.nan))

    weights = trainer.reward_weights.to(rewards_per_func.device).unsqueeze(0)
    if getattr(trainer, "multi_objective_aggregation", None) == "normalize_then_sum":
        # TRL normalises each reward function within its group before the weighted sum.
        per_func = rewards_per_func.view(-1, num_generations, rewards_per_func.shape[1])
        per_func = (per_func - torch.nanmean(per_func, dim = 1, keepdim = True)) / (
            nan_variance(per_func, 1).sqrt() + 1e-4
        )
        rewards = (per_func.view_as(rewards_per_func) * weights).nansum(dim = 1)
    else:
        rewards = (rewards_per_func * weights).nansum(dim = 1)
    if _unsloth_trl_masks_unscorable():
        # TRL >= 1.6 leaves a completion no reward function scored out of its group's statistics;
        # older TRL scores it 0 (nansum), so its group has variance and real advantages there.
        rewards[torch.isnan(rewards_per_func).all(dim = 1)] = torch.nan
    variance = nan_variance(rewards.view(-1, num_generations), 1).squeeze(1)
    group_keep = torch.ones_like(variance, dtype = torch.bool)
    if exclude:
        # TRL's frac_reward_zero_std test, per group.
        group_keep = ~torch.isclose(variance.sqrt(), torch.zeros_like(variance))
    keep = group_keep.repeat_interleave(num_generations).to(torch.float32).to(lengths.device)
    if prompt_level:
        group_tokens = (lengths * keep).view(-1, num_generations).sum(dim = 1)
        groups = group_keep.sum().to(torch.float32).clamp(min = 1.0)
        weight = keep / (group_tokens.clamp(min = 1.0).repeat_interleave(num_generations) * groups)
    else:
        weight = keep / (lengths * keep).sum().clamp(min = 1.0)
    trainer._metrics[mode]["frac_reward_zero_std_excluded"].append(
        float(1.0 - group_keep.float().mean()) if exclude else 0.0
    )
    return weight[process_slice].to(completion_mask.device)

def _unsloth_grpo_vision_inputs(source):
    """unsloth_zoo owns this key tuple; the copy below is the fallback for a zoo predating
    GRPO_VISION_KEYS, and test_grpo_vision_kwargs_forwarded.py fails if the two diverge."""
    try:
        from unsloth_zoo.rl_replacements import grpo_get_vision_inputs
        return grpo_get_vision_inputs(source)
    except Exception:
        pass
    if source is None:
        return {}
    get = getattr(source, "get", None)
    if get is None:
        return {}
    return {
        key: get(key, None)
        for key in (
            "pixel_values",
            "image_grid_thw",
            "pixel_attention_mask",
            "image_sizes",
            "spatial_shapes",
            "num_tiles",
            # both Gemma 4 spellings: image_position_ids is TRL >= 1.1.0
            "image_position_ids",
            "pixel_position_ids",
            "num_images",
            "token_type_ids",
            "mm_token_type_ids",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
            "num_videos",
        )
    }

def _unsloth_grpo_split_vision_by_sample(batch):
    """TRL only splits the tile and image indexed vision tensors per sample from its own
    split_pixel_values_by_grid, and before TRL 1.1.0 that function knows only the
    image_grid_thw layout. Every other layout is left flat, and _prepare_inputs then
    shuffles and slices the batch by sample index, so an LFM2-VL or Gemma batch has its
    tiles reordered away from the samples they belong to. This mirrors the current
    trl.trainer.utils.split_pixel_values_by_grid so an older TRL keeps them together."""

    def _counts(value):
        if value is None:
            return None
        if hasattr(value, "tolist"):
            value = value.tolist()
        try:
            return [int(count) for count in value]
        except TypeError:
            return None

    batch = _unsloth_grpo_split_videos_by_sample(batch)
    pixel_values = batch.get("pixel_values", None)
    if pixel_values is None:
        return batch
    image_grid_thw = batch.get("image_grid_thw", None)
    num_images = _counts(batch.get("num_images", None))

    if isinstance(pixel_values, list):
        # TRL 1.1.0+ splits by sample; 0.22.x-0.23.x splits by grid row and leaves image_grid_thw flat, so
        # a two-image row is mis-shuffled. Regroup it. Not len(pixel_values) == len(num_images): [0, 2]
        # makes them agree while the axes differ.
        _prompt_ids = batch.get("prompt_ids", None)
        if (
            image_grid_thw is None
            or isinstance(image_grid_thw, list)
            or not num_images
            or not pixel_values
            or len(pixel_values) != sum(num_images)
            or all(_count == 1 for _count in num_images)
            or (_prompt_ids is not None and len(num_images) != _prompt_ids.shape[0])
        ):
            return batch
        split = dict(batch)
        _empty = pixel_values[0][:0]
        _grouped = []
        _offset = 0
        for _count in num_images:
            _group = pixel_values[_offset : _offset + _count]
            _offset += _count
            _grouped.append(torch.cat(_group, dim = 0) if _group else _empty)
        split["pixel_values"] = _grouped
        split["image_grid_thw"] = list(torch.split(image_grid_thw, num_images, dim = 0))
        for _image_key in ("pixel_attention_mask", "image_sizes"):
            _per_image = batch.get(_image_key, None)
            if (
                _per_image is not None
                and not isinstance(_per_image, list)
                and _per_image.shape[0] == sum(num_images)
                and _per_image.shape[0] != len(num_images)
            ):
                split[_image_key] = list(torch.split(_per_image, num_images, dim = 0))
        return split

    if image_grid_thw is not None:
        # An unsplit grid batch: TRL owns this layout in every version that persists it.
        return batch
    if not num_images:
        return batch
    rows = pixel_values.shape[0]
    num_tiles = _counts(batch.get("num_tiles", None))
    split = dict(batch)
    for _position_key in ("image_position_ids", "pixel_position_ids"):
        _position_ids = batch.get(_position_key, None)
        if _position_ids is None or isinstance(_position_ids, list):
            continue
        if rows != sum(num_images) or _position_ids.shape[0] != sum(num_images):
            continue
        split["pixel_values"] = list(torch.split(pixel_values, num_images, dim = 0))
        split[_position_key] = list(torch.split(_position_ids, num_images, dim = 0))
        return split
    if num_tiles and rows == sum(num_tiles):
        split["pixel_values"] = list(torch.split(pixel_values, num_tiles, dim = 0))
        for _tile_key in ("pixel_attention_mask", "spatial_shapes"):
            _tiled = batch.get(_tile_key, None)
            if (
                _tiled is not None
                and not isinstance(_tiled, list)
                and _tiled.shape[0] == sum(num_tiles)
            ):
                split[_tile_key] = list(torch.split(_tiled, num_tiles, dim = 0))
        return split
    if rows != sum(num_images):
        # One padded row per sample already (Idefics, SmolVLM): TRL leaves this alone.
        return batch
    if rows == len(num_images) and any(_count != 1 for _count in num_images):
        # Padded-sample and flat-image axes are ambiguous: keep TRL layout, else [2, 0] misassigns images
        return batch
    split["pixel_values"] = list(torch.split(pixel_values, num_images, dim = 0))
    _image_sizes = batch.get("image_sizes", None)
    if (
        _image_sizes is not None
        and not isinstance(_image_sizes, list)
        and _image_sizes.shape[0] == sum(num_images)
    ):
        split["image_sizes"] = list(torch.split(_image_sizes, num_images, dim = 0))
    return split

def _unsloth_grpo_unsplit_vision(batch):
    """Undo _unsloth_grpo_split_vision_by_sample once this step's slice has been taken, so
    the forward sees the layout the processor produced. TRL's own unsplit merges only
    pixel_values before 1.1.0, and pixel_values plus the grid and position ids from 1.1.0."""
    merged = None
    for key in (
        "pixel_values",
        "image_grid_thw",
        "pixel_attention_mask",
        "spatial_shapes",
        "image_sizes",
        "image_position_ids",
        "pixel_position_ids",
        "pixel_values_videos",
        "video_grid_thw",
        "second_per_grid_ts",
    ):
        value = batch.get(key, None)
        if not isinstance(value, list) or len(value) == 0:
            continue
        if not hasattr(value[0], "shape"):
            # num_images and num_tiles are plain counts, not tensors to merge.
            continue
        if merged is None:
            merged = dict(batch)
        merged[key] = torch.cat(value, dim = 0)
    return batch if merged is None else merged

def _unsloth_grpo_split_videos_by_sample(batch):
    """Group the flat video rows, grid and second_per_grid_ts by sample (num_videos) so the
    _prepare_inputs shuffle keeps them with their sample; _unsloth_grpo_unsplit_vision merges back."""
    pixel_values_videos = batch.get("pixel_values_videos", None)
    video_grid_thw = batch.get("video_grid_thw", None)
    num_videos = batch.get("num_videos", None)
    if (
        pixel_values_videos is None
        or video_grid_thw is None
        or num_videos is None
        or isinstance(pixel_values_videos, list)
    ):
        return batch
    num_videos = [int(n) for n in num_videos]
    if sum(num_videos) != video_grid_thw.shape[0]:
        return batch
    rows_per_sample = [
        int(group.prod(dim = -1).sum()) for group in torch.split(video_grid_thw, num_videos)
    ]
    split = dict(batch)
    split["pixel_values_videos"] = list(torch.split(pixel_values_videos, rows_per_sample))
    split["video_grid_thw"] = list(torch.split(video_grid_thw, num_videos))
    second_per_grid_ts = batch.get("second_per_grid_ts", None)
    if second_per_grid_ts is not None:
        second_per_grid_ts = torch.as_tensor(second_per_grid_ts)
        split["second_per_grid_ts"] = list(torch.split(second_per_grid_ts, num_videos))
    return split

def _unsloth_grpo_prompt_videos(prompt, kind = "video"):
    """Number of `kind` parts in a conversational prompt."""
    if not isinstance(prompt, list):
        return 0
    count = 0
    for message in prompt:
        content = message.get("content", None) if isinstance(message, dict) else None
        if isinstance(content, list):
            count += sum(
                1 for part in content if isinstance(part, dict) and part.get("type") == kind
            )
    return count

def _unsloth_grpo_has_video_key(prompt):
    if not isinstance(prompt, list):
        return False
    for message in prompt:
        content = message.get("content", None) if isinstance(message, dict) else None
        if isinstance(content, list) and any(
            isinstance(part, dict) and "video" in part for part in content
        ):
            return True
    return False

def _unsloth_grpo_clean_video_prompts(prompts, trainer = None):
    """Arrow adds `video: None` to every content part of a column holding videos; Qwen2-VL
    templates test `'video' in content` and render a placeholder per part (StopIteration)."""
    if not any(_unsloth_grpo_has_video_key(prompt) for prompt in prompts):
        return prompts
    if getattr(trainer, "use_vllm", False) and any(
        _unsloth_grpo_prompt_videos(prompt) for prompt in prompts
    ):
        # Before generation, which vLLM would otherwise run without the videos.
        raise NotImplementedError(
            "Unsloth: video GRPO runs on the transformers generation path only. "
            "Load the model with fast_inference = False (use_vllm = False)."
        )
    cleaned = []
    for prompt in prompts:
        if not _unsloth_grpo_has_video_key(prompt):
            cleaned.append(prompt)
            continue
        messages = []
        for message in prompt:
            content = message.get("content", None)
            if isinstance(content, list):
                message = dict(message)
                message["content"] = [
                    {k: v for k, v in part.items() if v is not None}
                    if isinstance(part, dict)
                    else part
                    for part in content
                ]
            messages.append(message)
        cleaned.append(messages)
    return cleaned

def _unsloth_grpo_prompt_key(value):
    """Dedup key: equal for the repeated copies of a prompt, never shared by different media.
    repr() truncates arrays, so in-memory frames are keyed by identity, not by content."""
    if isinstance(value, dict):
        return tuple((k, _unsloth_grpo_prompt_key(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_unsloth_grpo_prompt_key(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return ("id", id(value))

def _unsloth_grpo_video_inputs(
    trainer,
    prompts,
    images = None,
):
    """Video tensors for the logprob forwards, which TRL builds from images only. Rendered with
    the same chat template call generation makes, once per distinct prompt. None without video."""
    num_videos = [_unsloth_grpo_prompt_videos(prompt) for prompt in prompts]
    if not any(num_videos):
        return None
    if getattr(trainer, "use_vllm", False):
        raise NotImplementedError(
            "Unsloth: video GRPO runs on the transformers generation path only. "
            "Load the model with fast_inference = False (use_vllm = False)."
        )
    if (images is not None and any(images)) or any(
        _unsloth_grpo_prompt_videos(prompt, "image") for prompt in prompts
    ):
        raise NotImplementedError(
            "Unsloth: GRPO does not support image and video inputs in the same batch yet."
        )
    try:
        from unsloth_zoo.rl_replacements import GRPO_VISION_KEYS
    except Exception:
        GRPO_VISION_KEYS = ()
    if "pixel_values_videos" not in GRPO_VISION_KEYS:
        raise RuntimeError(
            "Unsloth: video GRPO needs a newer unsloth_zoo that forwards video inputs to "
            "both GRPO logprob passes. Please upgrade: pip install -U unsloth_zoo"
        )
    unique, order = {}, []
    for prompt in prompts:
        order.append(unique.setdefault(_unsloth_grpo_prompt_key(prompt), len(unique)))
    first = {}
    for i, u in enumerate(order):
        first.setdefault(u, i)
    unique_prompts = [prompts[first[u]] for u in range(len(unique))]
    template_kwargs = dict(getattr(trainer, "chat_template_kwargs", None) or {})
    if getattr(trainer, "chat_template", None) is not None:
        # GRPOConfig.chat_template, which generation renders with too
        template_kwargs["chat_template"] = trainer.chat_template
    processed = trainer.processing_class.apply_chat_template(
        conversation = unique_prompts,
        add_generation_prompt = True,
        tokenize = True,
        return_dict = True,
        padding = True,
        return_tensors = "pt",
        **template_kwargs,
    )
    pixel_values_videos = processed.get("pixel_values_videos", None)
    video_grid_thw = processed.get("video_grid_thw", None)
    if pixel_values_videos is None or video_grid_thw is None:
        raise NotImplementedError(
            f"Unsloth: video GRPO supports processors that return pixel_values_videos and "
            f"video_grid_thw (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5); "
            f"{type(trainer.processing_class).__name__} does not."
        )
    second_per_grid_ts = processed.get("second_per_grid_ts", None)
    if second_per_grid_ts is not None:
        second_per_grid_ts = torch.as_tensor(second_per_grid_ts)
    unique_counts = [num_videos[first[u]] for u in range(len(unique))]
    video_index = torch.split(torch.arange(video_grid_thw.shape[0]), unique_counts)
    row_ends = video_grid_thw.prod(dim = -1).cumsum(0)
    row_starts = row_ends - video_grid_thw.prod(dim = -1)
    videos, rows = [], []
    for u in order:
        for v in video_index[u].tolist():
            videos.append(v)
            rows.append(pixel_values_videos[int(row_starts[v]) : int(row_ends[v])])
    device = trainer.accelerator.device
    out = {
        "pixel_values_videos": torch.cat(rows, dim = 0).to(device),
        "video_grid_thw": video_grid_thw[videos].to(device),
        "num_videos": num_videos,
    }
    if second_per_grid_ts is not None:
        out["second_per_grid_ts"] = second_per_grid_ts[videos].to(device)
    return out

def _unsloth_grpo_image_cell(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]

def _unsloth_reject_grpo_image_list(inputs, trainer = None):
    """Refuse it where the rewrite above missed TRL's spelling: the processor's own error
    names neither the column nor the fix.

    `trainer` narrows the refusal to the one runtime mode a legacy TRL cannot carry a multi
    image row through. Its vLLM server path sends the raw cells to `VLLMClient.generate`,
    which does `[pil_to_base64(img) for img in images]` over the top level entries, so a cell
    holding two images reaches `list.save(...)` and dies with an AttributeError naming neither.
    Colocate mode and the no vLLM path both go through the processor, which this change fixed,
    so they keep working and must not be refused."""
    if trainer is not None:
        if not getattr(trainer, "use_vllm", False):
            return
        if getattr(trainer, "vllm_mode", None) != "server":
            return
    # Every row: one list cell anywhere desyncs the images and the placeholders.
    try:
        rows = list(inputs)
    except Exception:
        return
    for _row_index, _row in enumerate(rows):
        value = _row.get("image", None) if isinstance(_row, dict) else None
        if isinstance(value, (list, tuple)) and len(value) > 1:
            raise ValueError(
                f"Unsloth: GRPO received a singular `image` column holding {len(value)} "
                f"images in row {_row_index}, and this TRL version cannot carry more than "
                "one image per row through to the model. "
                "Rename the column to `images`, which TRL reads as the per example list of "
                "images, or keep one image per row. "
                "See https://github.com/unslothai/unsloth/issues/3605"
            )

def _unsloth_clear_stateful_mrope(model):
    modules = getattr(model, "modules", None)
    if modules is None:
        return False

    cleared = False
    for module in modules():
        if hasattr(module, "compute_3d_position_ids") and hasattr(module, "rope_deltas"):
            module.rope_deltas = None
            cleared = True
    return cleared

def grpo_compute_loss(
    ref,
    new,
    old,
    sampling_per_token_logps,
    input_ids,
    mask,
    beta,
    advantages,
    **kwargs
):
    # All Unsloth Zoo code licensed under AGPL3
    # Optional argument defaults.
    loss_type = kwargs.get("loss_type", "grpo")
    epsilon_low = kwargs.get("epsilon_low", 0.2)
    epsilon_high = kwargs.get("epsilon_high", 0.2)
    max_completion_length = kwargs.get("max_completion_length", 8192)
    delta = kwargs.get("delta", None)
    importance_sampling_level = kwargs.get("importance_sampling_level", "token")
    num_items_in_batch = kwargs.get("num_items_in_batch", None)
    current_gradient_accumulation_steps = kwargs.get("current_gradient_accumulation_steps", 1)
    steps_per_generation = kwargs.get("steps_per_generation", None)
    num_processes = kwargs.get("num_processes", 1)
    use_vllm = kwargs.get("use_vllm", False)
    # The off-policy mask uses vLLM sampling logprobs whenever the batch supplies them (matching TRL);
    # the vLLM importance-sampling ratio is applied to the loss only when this flag is on.
    vllm_importance_sampling_correction = kwargs.get("vllm_importance_sampling_correction", False)
    vllm_importance_sampling_mode = kwargs.get("vllm_importance_sampling_mode", "sequence_mask")
    vllm_importance_sampling_cap = kwargs.get("vllm_importance_sampling_cap", 2.0)
    vllm_importance_sampling_clip_min = kwargs.get("vllm_importance_sampling_clip_min", None)
    vllm_importance_sampling_clip_max = kwargs.get("vllm_importance_sampling_clip_max", 3.0)
    get_sapo_token_loss = kwargs.get("get_sapo_token_loss", None)
    sapo_temperature_pos = kwargs.get("sapo_temperature_pos", 1.0)
    sapo_temperature_neg = kwargs.get("sapo_temperature_neg", 1.05)
    get_gamma_weights = kwargs.get("get_gamma_weights", None)
    vespo_k_pos = kwargs.get("vespo_k_pos", 2.0)
    vespo_lambda_pos = kwargs.get("vespo_lambda_pos", 3.0)
    vespo_k_neg = kwargs.get("vespo_k_neg", 3.0)
    vespo_lambda_neg = kwargs.get("vespo_lambda_neg", 2.0)
    get_off_policy_mask = kwargs.get("get_off_policy_mask", None)
    off_policy_mask_threshold  = kwargs.get("off_policy_mask_threshold", None)
    # TRL top_entropy_quantile: the policy term (not the KL) on high-entropy tokens only.
    entropy_mask = kwargs.get("entropy_mask", None)
    # Parsed (option, lower_log, upper) tuples from grpo_parse_rollout_rs, or None.
    rollout_rs = kwargs.get("rollout_rs", None)
    # Per-completion weights from unsloth's generation step (ScaleRL zero-variance exclusion /
    # prompt-level aggregation), normalised over the whole generation batch; replaces the loss_type normaliser.
    sample_weight = kwargs.get("sample_weight", None)
    # Only direct callers see this fallback; the trainer always forwards an explicit value.
    use_bias_correction_kl = kwargs.get("use_bias_correction_kl", False)
    input_ids = input_ids.unsqueeze(-1)

    importance_sampling_ratio = None
    # Bool (B, T) keep mask from rollout_rs; None when rejection is off.
    rs_keep = None

    # exp(new - old) and exp(ref - new) below are taken before `mask` is applied. A sequence-packed
    # logp path leaves the masked (prompt/pad) columns at 0 while a padded one fills them with a real
    # logp, so when new and old/ref disagree there those ratios can overflow to inf and inf * 0 (the
    # masked-out loss) becomes nan. Force new/old/ref to share 0 on the masked columns so both ratios
    # are exp(0) = 1 there; every loss term below multiplies by `mask`, so this changes nothing.
    if mask is not None:
        _keep = mask.to(torch.bool)
        new = torch.where(_keep, new, 0.0)
        if old is not None: old = torch.where(_keep, old, 0.0)
        if ref is not None: ref = torch.where(_keep, ref, 0.0)

    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)

    if off_policy_mask_threshold is not None:
        # DeepSeek-V3.2 off-policy mask. The mismatch logprobs are sampling_per_token_logps (vLLM
        # sampling logprobs) if present, else old, else new.detach() when both are absent
        # (num_iterations == 1 with no vLLM). This mirrors TRL, which defaults old_per_token_logps to
        # per_token_logps.detach() so get_off_policy_mask never receives None (it computes
        # mismatch - per_token_logps.detach(), so new.detach() yields a zero-KL keep-all mask). The
        # callable is a signature-stable adapter installed in grpo_accumulated_loss, so this stays
        # fixed across TRL versions with no signature introspection inside this compiled function.
        off_policy_mask = get_off_policy_mask(
            advantages=advantages,
            per_token_logps=new,
            sampling_per_token_logps=sampling_per_token_logps if sampling_per_token_logps is not None else (old if old is not None else new.detach()),
            mask=mask,
            off_policy_threshold=off_policy_mask_threshold,
        )

    with torch.no_grad():
        if use_vllm and sampling_per_token_logps is not None and vllm_importance_sampling_correction:
            # Filter out extra leading prompt tokens after left-padding input_ids.
            # Match TRL: aggregate log-ratios then exp (product), not sum of exp ratios.
            importance_sampling_ratio = (old - sampling_per_token_logps) * mask
            # Unscored vLLM tokens arrive as nan and nan * 0 survives the mask: ratio 1, as TRL does.
            importance_sampling_ratio = torch.nan_to_num(importance_sampling_ratio, nan = 0.0)

            if vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]:
                importance_sampling_ratio = importance_sampling_ratio.sum(dim=-1, keepdim=True)
            elif vllm_importance_sampling_mode in ["geometric_mask", "geometric_truncate"]:
                # Geometric mean of the token ratios (verl seq_mean_k1): the product blows up on long rows.
                importance_sampling_ratio = importance_sampling_ratio.sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).clamp(min=1.0)

            importance_sampling_ratio = torch.exp(importance_sampling_ratio)

            if vllm_importance_sampling_mode in ["token_truncate", "sequence_truncate", "geometric_truncate"]:
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, 
                    min=vllm_importance_sampling_clip_min,
                    max=vllm_importance_sampling_clip_max
                )
            elif vllm_importance_sampling_mode in ["token_mask", "sequence_mask", "geometric_mask"]:
                min_val = (
                    vllm_importance_sampling_clip_min
                    if vllm_importance_sampling_clip_min is not None
                    else -math.inf
                )

                max_val = (
                    vllm_importance_sampling_clip_max
                    if vllm_importance_sampling_clip_max is not None
                    else math.inf
                )

                invalid_mis_mask = (importance_sampling_ratio < min_val) | (
                        importance_sampling_ratio > max_val
                )

                importance_sampling_ratio = importance_sampling_ratio.masked_fill(
                        invalid_mis_mask, value=0.0
                )
            else:
                raise ValueError(
                        f"Unknown vLLM importance sampling mode: {vllm_importance_sampling_mode}. Possible values are 'token_truncate', 'token_mask', 'sequence_truncate', 'sequence_mask', 'geometric_truncate' and 'geometric_mask'."
                )

        # verl compute_rollout_rejection_mask: every option must pass. verl then replaces
        # response_mask with response_mask * keep, so a rejected token leaves the policy term
        # and the KL loss (both selected out below). It stays in the loss normaliser like
        # TRL's *_mask modes, since num_items_in_batch is counted before the loss.
        # Log ratio is pi_old / pi_rollout; with rollout logps as pi_old (verl bypass mode) it is
        # pi_theta / pi_rollout, verl compute_policy_loss_bypass_mode passing old_log_prob=log_prob.
        rs_log_ratio = None
        if rollout_rs is not None:
            if use_vllm and sampling_per_token_logps is not None and vllm_importance_sampling_correction:
                rs_log_ratio = old - sampling_per_token_logps
            elif kwargs.get("rollout_logps_as_old", False) and old is not None:
                rs_log_ratio = new.detach() - old
        if rs_log_ratio is not None:
            rs_log_ratio = torch.clamp(torch.nan_to_num(rs_log_ratio * mask, nan = 0.0), min = -20.0, max = 20.0)
            rs_mask = mask.to(rs_log_ratio.dtype)
            rs_count = rs_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            rs_keep = torch.ones_like(rs_log_ratio)
            for rs_option, rs_lower, rs_upper in rollout_rs:
                if rs_option.endswith("k1"):
                    rs_stat = -rs_log_ratio
                elif rs_option.endswith("k2"):
                    rs_stat = 0.5 * rs_log_ratio * rs_log_ratio
                else:
                    rs_stat = torch.expm1(rs_log_ratio) - rs_log_ratio
                rs_stat = rs_stat * rs_mask
                if rs_option.startswith("seq_sum"):
                    rs_stat = rs_stat.sum(dim=-1, keepdim=True)
                elif rs_option.startswith("seq_mean"):
                    rs_stat = rs_stat.sum(dim=-1, keepdim=True) / rs_count
                elif rs_option.startswith("seq_max"):
                    # k2, k3 >= 0 and masked entries are 0, so this is the masked max (0 on an empty row).
                    rs_stat = rs_stat.amax(dim=-1, keepdim=True)
                if rs_lower is not None:
                    rs_pass = (rs_stat >= rs_lower) & (rs_stat <= rs_upper)
                else:
                    rs_pass = rs_stat <= rs_upper
                rs_keep = rs_keep * rs_pass.to(rs_keep.dtype)
            rs_keep = rs_keep > 0
            if importance_sampling_ratio is not None:
                # Select, not multiply: an unclipped ratio can overflow to inf and inf * 0 is nan.
                importance_sampling_ratio = torch.where(rs_keep, importance_sampling_ratio, 0.0)
    pass

    # Must detach when old is None: exp(new - new.detach()) == 1 but keeps grads correct.
    if old is not None:
        log_ratio = new - old
    else:
        log_ratio = new - new.detach()
    if kwargs.get("rollout_logps_as_old", False):
        # Rollout gap can pass 88 nats (float32 exp overflow, 0 * inf backward); verl
        # compute_policy_loss_vanilla clamps the log ratio to [-20, 20] too.
        log_ratio = torch.clamp(log_ratio, min = -20.0, max = 20.0)

    if importance_sampling_level == "token":
        log_importance_weights = log_ratio
    elif importance_sampling_level == "sequence":
        log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        log_importance_weights = log_importance_weights.unsqueeze(-1)
    else:
        raise ValueError(
            f"Unknown importance sampling level: {importance_sampling_level}. Possible values are 'token' "
            "and 'sequence'."
        )

    coef_1 =  torch.exp(log_importance_weights)

    # Reverse KL: low-variance low-bias estimator as used in the GRPO paper.
    if beta != 0.0:
        # Keep expm1 finite, then cap the final KL term (TRL does not clamp).
        kl_log_ratio = torch.clamp(ref - new, min = -20.0, max = 20.0)
        kl_i = torch.expm1(kl_log_ratio) - kl_log_ratio
        # TRL order: pre-clamp non-detached coef_1, before the loss_type dispatch.
        if use_bias_correction_kl:
            kl_i = kl_i * coef_1
        # Clamp after correction so a saturated KL also stops the ratio's gradient.
        kl_i = torch.clamp(kl_i, min = -10.0, max = 10.0)
    else:
        # Zeros with the correct shape.
        if importance_sampling_level == "sequence":
            kl_i = new.new_zeros(new.size(0), 1)
        else:
            kl_i = torch.zeros_like(new)

    if loss_type == "cispo":
        clamped_ratios = torch.clamp(coef_1, max=epsilon_high).detach()
        loss_i = -clamped_ratios * advantages * new
    elif loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
        coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)

        if delta is not None:
            loss_1 = torch.clamp(coef_1, max=delta) * advantages
        else:
            loss_1 = coef_1 * advantages
        pass
        loss_2 = coef_2 * advantages
        loss_i = -torch.min(loss_1, loss_2)
    elif loss_type == "sapo":
        temperatures = torch.where(advantages > 0, sapo_temperature_pos, sapo_temperature_neg)
        soft_coef_1 = torch.sigmoid(temperatures * (coef_1 - 1)) * 4 / temperatures
        loss_i = -soft_coef_1 * advantages
    elif loss_type == "vespo":
        if get_gamma_weights is None:
            raise Exception("vespo is only available in TRL 0.26.0+")
        phi_seq = get_gamma_weights(
            advantages=advantages,
            log_ratio_per_token=log_ratio,
            mask=mask,
            importance_sampling_ratio=importance_sampling_ratio,
            k_pos=vespo_k_pos,
            lambda_pos=vespo_lambda_pos,
            k_neg=vespo_k_neg,
            lambda_neg=vespo_lambda_neg,
        )
        loss_i = -phi_seq * advantages * new
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    if off_policy_mask_threshold is not None:
        loss_i = loss_i * off_policy_mask

    if entropy_mask is not None:
        loss_i = loss_i * entropy_mask

    if use_vllm and sampling_per_token_logps is not None and vllm_importance_sampling_correction:
        # vespo applies the IS ratio inside get_gamma_weights, so skip it here.
        if loss_type != "vespo":
            loss_i = loss_i * importance_sampling_ratio
        # delta for the metric.
        with torch.no_grad():
            delta = torch.abs(old - sampling_per_token_logps)
            delta = delta * mask
            # Zeroed rather than filtered like TRL, to keep the returned shape.
            delta = torch.nan_to_num(delta, nan = 0.0, posinf = math.inf)
            flat_is_ratio = importance_sampling_ratio * mask
    else:
        delta = torch.tensor([]).detach()
        flat_is_ratio = torch.tensor([]).detach()
    if beta != 0.0:
        loss_i = loss_i + beta * kl_i
    if rs_keep is not None:
        # Rejected tokens drop out of every loss term, KL included (verl rollout_rs). A select, so
        # an inf or nan there neither reaches the loss nor turns into a nan gradient.
        loss_i = torch.where(rs_keep, loss_i, 0.0)

    mask = mask.to(torch.float32)
    n_mask_per_reward = mask.sum(1)

    # https://github.com/huggingface/trl/blob/e8b8499f1f8d76838155b515e414ee98f757d6d5/trl/trainer/grpo_trainer.py#L1624
    if sample_weight is not None:
        # Same process / accumulation rescale as the dapo branch, whose weights are 1 / num_items_in_batch.
        normalizer = 1.0 / num_processes
        if steps_per_generation:
            normalizer = normalizer * current_gradient_accumulation_steps / steps_per_generation
        loss = (loss_i * mask * sample_weight.to(loss_i.dtype).unsqueeze(-1)).sum() / normalizer
    elif loss_type in ["grpo", "sapo"]:
        loss = ((loss_i * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
        loss = loss / current_gradient_accumulation_steps
    elif loss_type == "bnpo":
        # BNPO uses the local micro-batch token count even when TRL supplies a global count.
        loss = (loss_i * mask).sum() / mask.sum().clamp(min=1.0)
        loss = loss / current_gradient_accumulation_steps
    elif loss_type == "dr_grpo":
        loss = (loss_i * mask).sum() / (loss_i.size(0) * max_completion_length)
        loss = loss / current_gradient_accumulation_steps
    elif loss_type in ["cispo", "dapo", "vespo"]:
        # Floor at 1 like TRL: a fully masked batch (mask_truncated_completions) is 0/0 = nan otherwise.
        if torch.is_tensor(num_items_in_batch):
            normalizer = num_items_in_batch.clamp(min = 1.0) / num_processes
        else:
            normalizer = max(float(num_items_in_batch), 1.0) / num_processes
        # num_items_in_batch spans the whole generation batch; rescale to one accumulation window like TRL.
        if steps_per_generation:
            normalizer = normalizer * current_gradient_accumulation_steps / steps_per_generation
        loss = (loss_i * mask).sum() / normalizer
    elif loss_type == "luspo":
        # loss_i is (B, T) unless sequence level with beta 0, so mask elementwise (TRL >= 1.10).
        loss = (loss_i * mask).sum(-1).mean()
        normalizer = current_gradient_accumulation_steps
        loss = loss / normalizer
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Folded metrics.
    def masked_batch_mean(x):
        with torch.inference_mode():
            completion_length = n_mask_per_reward.mean()
            # Sequence level only if x is narrower than mask: one-token completions are (B, 1) too.
            if x.shape[1] == 1 and mask.shape[1] != 1:
                return completion_length, x.mean()
            else:
                # Rows with no tokens (mask_truncated_completions) are left out, as in TRL.
                mean_kl_per_reward = (x * mask).sum(1) / n_mask_per_reward.clamp(min = 1.0)
                kept_rows = (n_mask_per_reward > 0).sum()
                mean_kl = torch.where(
                    kept_rows == n_mask_per_reward.numel(),
                    mean_kl_per_reward.mean(),
                    mean_kl_per_reward.sum() / kept_rows.clamp(min = 1),
                )
                return completion_length, mean_kl
    # Rejected tokens add nothing to the reported KL either; the denominator keeps every completion
    # token, like the loss normaliser.
    kl_metric = kl_i if rs_keep is None or beta == 0.0 else torch.where(rs_keep, kl_i, 0.0)
    completion_length, mean_kl = masked_batch_mean(kl_metric)
    return loss, completion_length, mean_kl, delta, flat_is_ratio, coef_1, mask

class UnslothEfficientGRPO(torch.autograd.Function):
    # All Unsloth Zoo code licensed under AGPL3
    @staticmethod
    def forward(ctx, _new_logps, _old_logps, _ref_logps, _sampling_per_token_logps, lm_head, _input_ids, _mask, _advantages, beta, scaler = None, n_chunks = 1, extra_kwargs=None, upstream_scale = None):
        if extra_kwargs is None:
            extra_kwargs = {}
        def compute_loss(new_logps, old_logps, ref_logps, sampling_per_token_logps, input_ids, mask, advantages, scaling):
            loss, completion_length, mean_kl, delta, flat_is_ratio, coef_1, _mask  = grpo_compute_loss(
                ref_logps,
                new_logps,
                old_logps,
                sampling_per_token_logps,
                input_ids,
                mask,
                beta,
                advantages,
                **extra_kwargs,
            )

            # Scale for mixed precision; return loss.detach() or autograd uses 2x VRAM.
            scaled_loss = loss * scaling
            return scaled_loss, (loss.detach(), completion_length, mean_kl, delta, flat_is_ratio, coef_1)
        pass

        device =_new_logps.device
        grad_inputs = torch.empty_like(_new_logps)
        accumulated_loss              = torch.zeros(1, device = device)[0]
        accumulated_completion_length = torch.zeros(1, device = device)[0]
        accumulated_mean_kl           = torch.zeros(1, device = device)[0]
        accumulated_delta             = []
        accumulated_flat_is_ratio     = []
        accumulated_coef_1            = []

        def accumulate_chunk(
            new_logps_j,
            old_logps_j,
            ref_logps_j,
            sampling_per_token_logps_j,
            input_ids_j,
            mask_j,
            advantages_j,
            scaling,
            grad_inputs_j,
        ):
            (chunk_grad_input,), (chunk_loss, (unscaled_loss, chunk_completion_length, chunk_mean_kl, chunk_delta, chunk_flat_is_ratio, chunk_coef_1)) = torch.func.grad_and_value(
                compute_loss,
                argnums = (0,),
                has_aux = True,
            )(new_logps_j, old_logps_j, ref_logps_j, sampling_per_token_logps_j, input_ids_j, mask_j, advantages_j, scaling)
            accumulated_loss             .add_(unscaled_loss)
            accumulated_completion_length.add_(chunk_completion_length)
            accumulated_mean_kl          .add_(chunk_mean_kl)
            accumulated_delta            .append(chunk_delta)
            accumulated_flat_is_ratio    .append(chunk_flat_is_ratio)
            accumulated_coef_1           .append(chunk_coef_1)
            grad_inputs_j[:] = chunk_grad_input
        pass

        from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback
        accumulate_chunk = torch_compile_with_fallback(
            fullgraph = True,
            # [TODO] Dynamic marking causes torch.compile errors if sequence length is long
            dynamic = True,
            options = torch_compile_options,
        )(accumulate_chunk)

        grad_inputs_chunks = torch.chunk(grad_inputs,        chunks = n_chunks, dim = 0)
        new_logps  = torch.chunk(_new_logps, chunks = n_chunks, dim = 0)
        if _old_logps is not None:
            old_logps  = torch.chunk(_old_logps, chunks = n_chunks, dim = 0)
        else:
            old_logps = [None] * n_chunks
        if _ref_logps is not None:
            ref_logps  = torch.chunk(_ref_logps, chunks = n_chunks, dim = 0)
        else:
            ref_logps = [None] * n_chunks
        if _sampling_per_token_logps is not None:
            sampling_per_token_logps  = torch.chunk(_sampling_per_token_logps, chunks = n_chunks, dim = 0)
        else:
            sampling_per_token_logps = [None] * n_chunks
        input_ids          = torch.chunk(_input_ids,         chunks = n_chunks, dim = 0)
        mask               = torch.chunk(_mask,              chunks = n_chunks, dim = 0)
        advantages         = torch.chunk(_advantages,        chunks = n_chunks, dim = 0)

        # Mixed precision scaling if present.
        scaling = scaler.get_scale() if scaler is not None else 1.0
        # A float scale is a dynamo guard (recompile per change); a float32 0-dim tensor is not.
        if scaler is not None and not torch.is_tensor(scaling):
            scaling = torch.full((), float(scaling), dtype = torch.float32, device = device)

        for (grad_inputs_j, new_logps_j, old_logps_j, ref_logps_j, sampling_per_token_logps_j, input_ids_j, mask_j, advantages_j, ) in\
            zip(grad_inputs_chunks, new_logps, old_logps, ref_logps, sampling_per_token_logps, input_ids, mask, advantages):

            # [TODO] Dynamic marking causes torch.compile errors if sequence length is long

            # mark_dynamic(new_hidden_states_j)
            # mark_dynamic(ref_hidden_states_j)
            # if old_hidden_states_j is not None:
            #     mark_dynamic(old_hidden_states_j)
            # mark_dynamic(input_ids_j)
            # mark_dynamic(mask_j)
            accumulate_chunk(
                new_logps_j,
                old_logps_j,
                ref_logps_j,
                sampling_per_token_logps_j,
                input_ids_j,
                mask_j,
                advantages_j,
                scaling,
                grad_inputs_j,
            )
        pass

        grad_inputs                  .div_(n_chunks)
        accumulated_loss             .div_(n_chunks)
        accumulated_completion_length.div_(n_chunks)
        accumulated_mean_kl          .div_(n_chunks)

        if _sampling_per_token_logps is not None:
            accumulated_delta = torch.cat(accumulated_delta, dim=0)
            accumulated_flat_is_ratio = torch.cat(accumulated_flat_is_ratio, dim=0)
        else:
            accumulated_delta = None
            accumulated_flat_is_ratio = None
        accumulated_coef_1  = torch.cat(accumulated_coef_1, dim=0)
        ctx.save_for_backward(grad_inputs)
        ctx.upstream_scale = upstream_scale
        return (
            accumulated_loss,
            accumulated_completion_length,
            accumulated_mean_kl,
            accumulated_delta,
            accumulated_flat_is_ratio,
            accumulated_coef_1
        )
    pass

    @staticmethod
    def backward(ctx, grad_output, dcompletion_length, dmean_kl, ddelta, ddflat_is_ratio, dcoef_1):
        (grad_input,) = ctx.saved_tensors
        if ctx.upstream_scale is not None:
            grad_input = grad_input * (grad_output * ctx.upstream_scale)
        return (grad_input, None, None, None, None, None, None, None, None, None, None, None, None)
    pass

def grpo_accumulated_loss(
    trainer,
    input_ids,
    attention_mask,
    logits_to_keep,
    completion_mask,
    advantages,
    old_logps,
    ref_logps,
    n_chunks = -1,
    tool_mask = None,
    **kwargs,
):
    # All Unsloth Zoo code licensed under AGPL3
    # Body-local import so the copy inlined into the generated trainer cache resolves.
    try:
        from unsloth_zoo.rl_replacements import _warn_unsupported_grpo_options
        _warn_unsupported_grpo_options(trainer)
    except Exception:
        pass

    # Body-local: this source is copied into the generated trainer without its imports.
    from unsloth_zoo.rl_replacements import (
        grpo_shared_vision_inputs as _grpo_get_vision_inputs,
        grpo_vision_chunks as _grpo_vision_chunks,
        grpo_chunked_hidden_states_logprobs as chunked_hidden_states_selective_log_softmax,
        chunked_selective_log_softmax,
        grpo_rollout_logps_as_old,
        grpo_rollout_as_old_ratio_metrics,
        grpo_rollout_old_fill_unscored,
    )
    if os.environ.get("UNSLOTH_GRPO_LOGPROB_RECOMPUTE", "0") == "1":
        from unsloth_zoo.rl_replacements import (
            chunked_hidden_states_selective_log_softmax_recompute as chunked_hidden_states_selective_log_softmax,
        )
    # Adaptive entropy control is not implemented, so it disables the bonus (and warns).
    _entropy_quantile = getattr(trainer.args, "top_entropy_quantile", 1.0)
    _entropy_mask_on = (
        _entropy_quantile is not None and _entropy_quantile < 1.0
        and hasattr(trainer, "get_high_entropy_mask")
    )
    _entropy_coef = float(getattr(trainer, "entropy_coef", getattr(trainer.args, "entropy_coef", 0.0)) or 0.0)
    _entropy_bonus_on = _entropy_coef != 0.0 and not getattr(trainer.args, "use_adaptive_entropy", False)
    _need_entropy = _entropy_mask_on or _entropy_bonus_on
    if _need_entropy:
        from unsloth_zoo.rl_replacements import (
            chunked_hidden_states_selective_log_softmax_recompute as _entropy_log_softmax,
            UnslothGRPOEntropyBonus as _UnslothGRPOEntropyBonus,
            grpo_entropy_bonus_weights as _grpo_entropy_bonus_weights,
            grpo_high_entropy_mask as _grpo_high_entropy_mask,
        )
    vision_inputs = _grpo_get_vision_inputs(kwargs)
    pixel_values = vision_inputs.get('pixel_values', None)
    image_grid_thw = vision_inputs.get('image_grid_thw', None)
    video_grid_thw = vision_inputs.get('video_grid_thw', None)
    if pixel_values is None:
        # Only a sentinel below (the chunks carry the tensors): video rows must take the vision
        # path too, since left and sequence packing move M-RoPE video positions.
        pixel_values = vision_inputs.get('pixel_values_videos', None)
    # Released unsloth 2026.9.4 decides whether multi-image GRPO is supported by grepping
    # inspect.getsource(grpo_accumulated_loss) for "num_images", so moving the handling into
    # grpo_vision_chunks makes that probe answer no and raise "Please upgrade unsloth_zoo" at
    # the user who just did. The chunker reads num_images out of vision_inputs itself; this
    # binding is what the released probe looks for, and it keeps the name meaningful here.
    num_images = vision_inputs.get('num_images', None)
    # Transformers 5.x requires token_type_ids/mm_token_type_ids for some vision models
    token_type_ids = vision_inputs.get('token_type_ids', None)
    mm_token_type_ids = vision_inputs.get('mm_token_type_ids', None)
    if mm_token_type_ids is not None or image_grid_thw is not None or video_grid_thw is not None:
        mm_token_type_ids = _unsloth_fix_mm_token_type_ids(
            trainer.processing_class, input_ids, mm_token_type_ids
        )
        vision_inputs['mm_token_type_ids'] = mm_token_type_ids
    # Thread vLLM sampling logprobs when something actually consumes them: the off-policy mask
    # (off_policy_mask_threshold) or the IS ratio (vllm_importance_sampling_correction). The mask
    # needs them regardless of IS correction (matching TRL, which feeds them to get_off_policy_mask
    # either way); the IS ratio stays gated on the correction flag inside grpo_compute_loss. On the
    # plain vLLM path (neither active) they are dropped so nothing pays for an unused aligned/compiled
    # input and grpo_compute_loss returns None (not empty) delta/flat_is_ratio.
    # Set by unsloth when UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD=1 skipped the old-policy forward.
    _rollout_logps_as_old = bool(kwargs.pop("rollout_logps_as_old", False))
    _sampling_logps_used = (
        getattr(trainer, "vllm_importance_sampling_correction", False)
        or getattr(trainer.args, "off_policy_mask_threshold", None) is not None
        or _rollout_logps_as_old
    )
    sampling_per_token_logps = kwargs.get("sampling_per_token_logps", None) if _sampling_logps_used else None
    temperature = kwargs.get("temperature", 1.0)
    logit_scale_multiply = kwargs.get("logit_scale_multiply", 0.0)
    logit_scale_divide   = kwargs.get("logit_scale_divide", 0.0)
    logit_softcapping    = kwargs.get("logit_softcapping", 0.0)
    prev_max_left_pad    = kwargs.get("max_left_pad", 0) # max_left_pad for LLM training, enabled by default.

    # Pop from kwargs to avoid downstream issues.
    _ = kwargs.pop("sampling_per_token_logps", None)
    kwargs["vllm_importance_sampling_cap"] = getattr(trainer.args, "vllm_importance_sampling_cap", None)
    # Older TRL lacks this arg; fall back to token_truncate (legacy clamp(max=cap) behavior).
    kwargs["vllm_importance_sampling_mode"] = getattr(trainer.args, "vllm_importance_sampling_mode", None) or "token_truncate"
    # Opt-in for TRL versions without the field or the geometric modes, e.g. geometric_mask.
    if os.environ.get("UNSLOTH_GRPO_VLLM_IS_MODE", ""):
        kwargs["vllm_importance_sampling_mode"] = os.environ["UNSLOTH_GRPO_VLLM_IS_MODE"]
    # Like TRL, only check the mode where it is used: with the correction off it stays a no-op.
    if getattr(trainer, "use_vllm", False) and getattr(trainer, "vllm_importance_sampling_correction", False) and kwargs["vllm_importance_sampling_mode"] not in (
        "token_truncate", "token_mask", "sequence_truncate", "sequence_mask", "geometric_truncate", "geometric_mask",
    ):
        raise ValueError(f"Unsloth: unknown vLLM importance sampling mode {kwargs['vllm_importance_sampling_mode']}.")
    # TRL validates this in __init__, before the env override: get_gamma_weights sums a per-token log ratio.
    if (
        kwargs.get("loss_type", getattr(trainer, "loss_type", None)) == "vespo"
        and getattr(trainer, "use_vllm", False)
        and getattr(trainer, "vllm_importance_sampling_correction", False)
        and kwargs["vllm_importance_sampling_mode"] not in ("token_truncate", "token_mask")
    ):
        raise ValueError(
            "Unsloth: VESPO loss requires vllm_importance_sampling_mode to be 'token_truncate' or 'token_mask'. "
            f"Got: {kwargs['vllm_importance_sampling_mode']}."
        )
    kwargs["vllm_importance_sampling_clip_min"] = getattr(trainer.args, "vllm_importance_sampling_clip_min", None)
    kwargs["vllm_importance_sampling_clip_max"] = getattr(trainer.args, "vllm_importance_sampling_clip_max", None)
    kwargs["get_sapo_token_loss"] = trainer.get_sapo_token_loss if hasattr(trainer, "get_sapo_token_loss") else None
    kwargs["sapo_temperature_pos"] = trainer.args.sapo_temperature_pos if hasattr(trainer.args, "sapo_temperature_pos") else None
    kwargs["sapo_temperature_neg"] = trainer.args.sapo_temperature_neg if hasattr(trainer.args, "sapo_temperature_neg") else None
    kwargs["get_gamma_weights"] = trainer.get_gamma_weights if hasattr(trainer, "get_gamma_weights") else None
    kwargs["vespo_k_pos"] = trainer.args.vespo_k_pos if hasattr(trainer.args, "vespo_k_pos") else 2.0
    kwargs["vespo_k_neg"] = trainer.args.vespo_k_neg if hasattr(trainer.args, "vespo_k_neg") else 3.0
    kwargs["vespo_lambda_pos"] = trainer.args.vespo_lambda_pos if hasattr(trainer.args, "vespo_lambda_pos") else 3.0
    kwargs["vespo_lambda_neg"] = trainer.args.vespo_lambda_neg if hasattr(trainer.args, "vespo_lambda_neg") else 2.0
    off_policy_mask_threshold = trainer.args.off_policy_mask_threshold if hasattr(trainer.args, "off_policy_mask_threshold") else None
    kwargs["off_policy_mask_threshold"] = off_policy_mask_threshold
    # get_off_policy_mask exists on TRL >= 0.27.0; its 3rd parameter was `old_per_token_logps` in 0.27.0
    # and renamed to `sampling_per_token_logps` in 0.27.1 (huggingface/trl#4857), so a fixed keyword call
    # crashes on one side of the rename. Wrap it in a signature-stable adapter here, outside the compiled
    # loss, so grpo_compute_loss always calls it with one keyword. Detect the real name once via inspect
    # and cache the adapter on the trainer; a fresh closure every step would re-trigger torch.compile.
    _off_policy_mask_fn = trainer.get_off_policy_mask if hasattr(trainer, "get_off_policy_mask") else None
    if _off_policy_mask_fn is None or off_policy_mask_threshold is None:
        kwargs["get_off_policy_mask"] = None
    else:
        _adapter = getattr(trainer, "_unsloth_off_policy_mask_adapter", None)
        # Compare by value, not identity: trainer.get_off_policy_mask returns a fresh bound-method
        # object on every access, so `is not` would always miss and rebuild the adapter each step
        # (re-triggering torch.compile). `!=` on bound methods compares __self__ and __func__, so the
        # cached adapter is reused; the first call still rebuilds since None != the bound method.
        if getattr(_adapter, "_unsloth_wrapped", None) != _off_policy_mask_fn:
            import inspect as _inspect
            try:
                _params = _inspect.signature(_off_policy_mask_fn).parameters
            except (TypeError, ValueError):
                _params = {}
            if "old_per_token_logps" in _params and "sampling_per_token_logps" not in _params:
                # TRL 0.27.0 named the mismatch-logprobs parameter old_per_token_logps.
                def _adapter(advantages, per_token_logps, sampling_per_token_logps, mask, off_policy_threshold):
                    return _off_policy_mask_fn(
                        advantages=advantages,
                        per_token_logps=per_token_logps,
                        old_per_token_logps=sampling_per_token_logps,
                        mask=mask,
                        off_policy_threshold=off_policy_threshold,
                    )
            else:
                # TRL >= 0.27.1 / 1.7.x use sampling_per_token_logps (also the default going forward).
                def _adapter(advantages, per_token_logps, sampling_per_token_logps, mask, off_policy_threshold):
                    return _off_policy_mask_fn(
                        advantages=advantages,
                        per_token_logps=per_token_logps,
                        sampling_per_token_logps=sampling_per_token_logps,
                        mask=mask,
                        off_policy_threshold=off_policy_threshold,
                    )
            _adapter._unsloth_wrapped = _off_policy_mask_fn
            trainer._unsloth_off_policy_mask_adapter = _adapter
        kwargs["get_off_policy_mask"] = _adapter
    # Read inside the compiled loss to gate the IS ratio, which the off-policy mask must not gate.
    kwargs["vllm_importance_sampling_correction"] = getattr(trainer, "vllm_importance_sampling_correction", False)
    # Follows TRL's own value; older TRL has no such field and False is correct there.
    kwargs["use_bias_correction_kl"] = getattr(trainer.args, "use_bias_correction_kl", False)
    kwargs["use_vllm"] = trainer.use_vllm
    # Eval batches are not split or accumulated, so keep the full count.
    _training = getattr(getattr(trainer, "model", None), "training", True)
    kwargs["steps_per_generation"] = getattr(trainer.args, "steps_per_generation", None) if _training else None
    # Generated trainers still pass unsloth_num_chunks; nothing downstream reads it.
    try:
        from unsloth_zoo.rl_replacements import _warn_deprecated_n_chunks
        _warn_deprecated_n_chunks(n_chunks)
    except Exception:
        pass

    if kwargs["vllm_importance_sampling_clip_max"] is None and kwargs["vllm_importance_sampling_cap"] is not None:
        kwargs["vllm_importance_sampling_clip_min"] = 0
        kwargs["vllm_importance_sampling_clip_max"] = kwargs["vllm_importance_sampling_cap"]
    # "lower_upper" (either side may be empty), e.g. "0.999_1.001" for the geometric modes.
    _is_clip_env = os.environ.get("UNSLOTH_GRPO_VLLM_IS_CLIP", "")
    if _is_clip_env:
        from unsloth_zoo.rl_replacements import grpo_parse_is_clip as _grpo_parse_is_clip
        kwargs["vllm_importance_sampling_clip_min"], kwargs["vllm_importance_sampling_clip_max"] = _grpo_parse_is_clip(
            _is_clip_env, kwargs["vllm_importance_sampling_mode"],
        )
    # verl rollout_rs rejection (token_k3, seq_mean_k3, seq_mean_k1, ...), applied with the vLLM IS ratio.
    from unsloth_zoo.rl_replacements import grpo_parse_rollout_rs as _grpo_parse_rollout_rs
    _rollout_rs = _grpo_parse_rollout_rs(
        os.environ.get("UNSLOTH_GRPO_ROLLOUT_RS", ""), os.environ.get("UNSLOTH_GRPO_ROLLOUT_RS_THRESHOLD", ""),
    )
    if _rollout_rs is not None:
        kwargs["rollout_rs"] = _rollout_rs
    _rollout_metrics_on = os.environ.get("UNSLOTH_GRPO_ROLLOUT_METRICS", "0") == "1"

    if not hasattr(trainer, '_autocast_dtype'):
        # "no" is float32 training (a T4 / V100 without bfloat16): autocasting it to bfloat16 raises there.
        _mixed_precision = os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16')
        trainer._autocast_dtype = None if _mixed_precision == 'no' else (torch.float16 if _mixed_precision == 'fp16' else torch.bfloat16)
        if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1': trainer._autocast_dtype = None
    pass
    # Restored in `finally`: an OOM or interrupt below must not leave later forwards returning hidden states.
    _unsloth_prior_hidden_states = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    try:
        output_embeddings = trainer.model.get_output_embeddings()
        lm_head = output_embeddings.weight
        lm_head_bias = getattr(output_embeddings, "bias", None)
        # ZeRO-3 released the head after the forward: score with one gathered copy per call.
        _zero3_head = None
        from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_zero3_partitioned, unsloth_zero3_gathered
        if unsloth_zero3_partitioned(lm_head) or unsloth_zero3_partitioned(lm_head_bias):
            _zero3_head = (unsloth_zero3_gathered(lm_head), unsloth_zero3_gathered(lm_head_bias))
            lm_head, lm_head_bias = _zero3_head

        # UNSLOTH_VLLM_FP32_LM_HEAD: one fp32 head copy per call, shared by every mini-batch kept for the loss.
        import inspect as _inspect
        from unsloth_zoo.rl_replacements import chunked_hidden_states_selective_log_softmax as _projector
        _fp32_lm_head = _inspect.signature(_projector).parameters["fp32_lm_head"].default
        _fp32_cache = {}

        def _fp32_bind():
            nonlocal lm_head, lm_head_bias
            if not _fp32_lm_head or lm_head.dtype == torch.float32:
                return
            # FSDP2 unshard / ZeRO-3 gather swap the objects; a grad-mode change needs a fresh cast too.
            key = (lm_head, lm_head_bias, torch.is_grad_enabled())
            if _fp32_cache.get("key") is None or any(a is not b for a, b in zip(_fp32_cache["key"], key)):
                bias = None if lm_head_bias is None else lm_head_bias.to(torch.float32)
                _fp32_cache.update(key = key, value = (lm_head.to(torch.float32), bias))
            lm_head, lm_head_bias = _fp32_cache["value"]

        def _refresh_lm_head():
            # FSDP2's forward unshard swaps the Parameters: the bindings above are metadata-only until then.
            nonlocal lm_head, lm_head_bias
            if _zero3_head is not None:
                lm_head, lm_head_bias = _zero3_head
                return
            from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_fsdp_unshard
            unsloth_fsdp_unshard(output_embeddings)
            lm_head = output_embeddings.weight
            lm_head_bias = getattr(output_embeddings, "bias", None)

        from unsloth_zoo.rl_replacements import grpo_truncated_support_kwargs
        _support_kwargs = grpo_truncated_support_kwargs(trainer)
        # old / ref logps come from Unsloth's _get_per_token_logps, which marks the trainer when
        # it scored them over the same set; mixing the two normalisations would bias every ratio.
        if _support_kwargs and (old_logps is not None or ref_logps is not None)\
                and not getattr(trainer, "_unsloth_truncated_support_scored", False):
            if not getattr(trainer, "_unsloth_truncated_support_refused", False):
                from unsloth_zoo.log import logger as _zoo_logger
                _zoo_logger.warning(
                    "Unsloth: UNSLOTH_GRPO_TRUNCATED_SUPPORT (scoring over the rollouts' kept set) needs an "
                    "unsloth build that scores the old / reference logprobs over the same kept set. "
                    "Scoring the full vocabulary instead: pip install -U unsloth"
                )
                trainer._unsloth_truncated_support_refused = True
            _support_kwargs = {}
        if _support_kwargs and getattr(trainer, "_unsloth_truncated_support_refused", False):
            # Unsloth found the scored policy is not the sampler (an unsynced vLLM sampler, a reference
            # model, reused rollouts): the current policy's own kept set would not be vLLM's either.
            _support_kwargs = {}
        if _support_kwargs:
            from functools import partial
            from unsloth_zoo.rl_replacements import (
                _GRPO_TRUNCATED_SUPPORT_STATS as _support_stats,
                grpo_truncated_support_setting as _support_setting,
            )
            _support_stats["on" if _support_setting() == "on" else "auto"] += 1
            if os.environ.get("UNSLOTH_GRPO_LOGPROB_RECOMPUTE", "0") == "1":
                # The recompute helper scores the full vocabulary only, so the kept set uses the plain helper
                from unsloth_zoo.rl_replacements import chunked_hidden_states_selective_log_softmax
                if not getattr(trainer, "_unsloth_truncated_support_recompute_warned", False):
                    from unsloth_zoo.log import logger as _zoo_logger
                    _zoo_logger.warning(
                        "Unsloth: UNSLOTH_GRPO_LOGPROB_RECOMPUTE=1 is not used with UNSLOTH_GRPO_TRUNCATED_SUPPORT=1; "
                        "the logprobs are scored with the standard helper over the kept set."
                    )
                    trainer._unsloth_truncated_support_recompute_warned = True
            chunked_hidden_states_selective_log_softmax = partial(
                chunked_hidden_states_selective_log_softmax, **_support_kwargs,
            )
            chunked_selective_log_softmax = partial(chunked_selective_log_softmax, **_support_kwargs)
        # Unsloth keeps _autocast_dtype set when it turns autocast off (float32 training on a GPU without bfloat16).
        _autocast_on = trainer._autocast_dtype is not None and getattr(trainer, "_autocast_enabled", True)
        dtype_bytes = 16 if _autocast_on and trainer._autocast_dtype in [torch.float16, torch.bfloat16] else 32
        if _support_kwargs:
            # Kept-set scoring saves one more float32 [rows, vocab] tensor per chunk for the backward
            # (the masked logits), so the mini-batch autotune counts the logits twice. No effect
            # unless free memory already limits the mini-batch.
            dtype_bytes = dtype_bytes * 2

        total_rows = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        hidden_dim = lm_head.shape[1]
        vocab_dim = lm_head.shape[0]

        if trainer.args.unsloth_grpo_mini_batch is None:
            # Size per call, as unsloth's copy does: caching in args froze the first step's plan.
            B, multiplier = autotune_batch_and_chunks(
                total_rows, seq_len, hidden_dim, vocab_dim, dtype_bytes, trainer.args.unsloth_logit_chunk_multiplier
            )
            B = max(1, total_rows//B)
        else:
            if trainer.args.unsloth_grpo_mini_batch > total_rows:
                B = total_rows
            else:
                B = trainer.args.unsloth_grpo_mini_batch

            if trainer.args.unsloth_logit_chunk_multiplier is None:
                multiplier = max(4, seq_len // 4096)
            else:
                multiplier = trainer.args.unsloth_logit_chunk_multiplier

        # The text path rebuilds completion_mask from token ids, which would undo TRL's
        # mask_truncated_completions row zeroing; keep TRL's rows and reapply them before the loss.
        kept_completion_rows = None
        if (
            pixel_values is None
            and getattr(trainer, "mask_truncated_completions", False)
            and torch.is_tensor(completion_mask)
            and completion_mask.dim() == 2
            and completion_mask.shape[0] == input_ids.shape[0]
        ):
            kept_completion_rows = completion_mask.sum(dim = 1, keepdim = True) > 0

        if pixel_values is None:
            left_pad_tokens_per_prompt = calculate_pad_tokens_in_prompt(input_ids, logits_to_keep, trainer.processing_class.pad_token_id)

            # Determine max_left_pad from precomputed logprobs shape for consistency
            if old_logps is not None:
                max_left_pad = old_logps.shape[1] - logits_to_keep
            elif ref_logps is not None:
                max_left_pad = ref_logps.shape[1] - logits_to_keep
            else:
                max_left_pad = torch.max(left_pad_tokens_per_prompt).item()

            input_ids = left_pack_padding(input_ids, trainer.processing_class.pad_token_id)

            completion_input_ids = input_ids[:, -(logits_to_keep +max_left_pad):]
            completion_mask = create_completion_attention_mask(completion_input_ids, left_pad_tokens_per_prompt, max_left_pad, trainer.processing_class.pad_token_id).to(attention_mask.dtype)

            if sampling_per_token_logps is not None:
                sampling_per_token_logps = align_logprobs_with_mask(sampling_per_token_logps, completion_mask)
            else:
                sampling_per_token_logps = None
            completion_mask = align_completion_tool_mask(tool_mask, completion_mask)
            attention_mask =  input_ids != trainer.processing_class.pad_token_id
            attention_mask = attention_mask.to(attention_mask.dtype)
        else:
            completion_input_ids = input_ids[:, -logits_to_keep:]
            completion_mask = align_completion_tool_mask(tool_mask, completion_mask)

        if _rollout_logps_as_old:
            old_logps, sampling_per_token_logps = grpo_rollout_logps_as_old(sampling_per_token_logps, old_logps)

        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper = False)

        for module in unwrapped_model.modules():
            if hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "io_same_decice"):
                module._hf_hook.io_same_decice = False
        pass

        all_logprobs_list = []
        all_entropies_list = []

        # No width dispatch in these helpers: unsloth rewrites every
        # `if <x>.shape[-1] == lm_head.shape[1]:` in this function and refuses any other spelling.
        def _hidden_logps_and_entropies(hidden_states, ids, chunks):
            logps, entropies = _entropy_log_softmax(
                hidden_states, lm_head, ids, chunks,
                logit_scale_multiply, logit_scale_divide, logit_softcapping, temperature,
                lm_head_bias = lm_head_bias, return_entropy = True,
                # Entropy reaches the loss only through the entropy bonus; a mask / metric needs no gradient.
                entropy_requires_grad = _entropy_bonus_on,
            )
            if _support_kwargs:
                # Entropies stay full-vocabulary as TRL's; the logprobs take the kept set
                logps = chunked_hidden_states_selective_log_softmax(
                    hidden_states, lm_head, ids, chunks,
                    logit_scale_multiply, logit_scale_divide, logit_softcapping, temperature,
                    lm_head_bias = lm_head_bias,
                )
            return logps, entropies

        def _raw_logits_entropies(logits):
            # The forward already scaled / softcapped these; same temperature as the logprobs.
            if _entropy_bonus_on:
                log_p = torch.log_softmax(logits.to(torch.float32) / temperature, dim = -1)
                return -(log_p.exp() * log_p).sum(-1)
            # Mask only (top_entropy_quantile): TRL scores entropies under no_grad, so no graph is kept,
            # and rows go in slices so the fp32 copies stay near 64M elements, not the whole [B, T, V].
            with torch.no_grad():
                flat = logits.reshape(-1, logits.shape[-1])
                out = torch.empty(flat.shape[0], dtype = torch.float32, device = logits.device)
                step = max(1, (1 << 26) // max(int(flat.shape[-1]), 1))
                for lo in range(0, flat.shape[0], step):
                    log_p = torch.log_softmax(flat[lo : lo + step].to(torch.float32) / temperature, dim = -1)
                    out[lo : lo + step] = -(log_p.exp() * log_p).sum(-1)
                    del log_p
                return out.reshape(logits.shape[:-1])

        import math
        total_samples = input_ids.shape[0]
        batch_size = math.ceil(total_samples / B)
        input_ids_chunks = []
        attention_mask_chunks = []
        completion_ids_chunks = []
        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            input_ids_chunks.append(input_ids[start:end])
            attention_mask_chunks.append(attention_mask[start:end])
            completion_ids_chunks.append(completion_input_ids[start:end])

        # Shared with the no-grad pass, so the two cannot slice the same tensors differently.
        vision_chunks = _grpo_vision_chunks(vision_inputs, total_samples, batch_size)

        zipped_inputs = zip(
            input_ids_chunks,
            attention_mask_chunks,
            vision_chunks,
            completion_ids_chunks,
        )

        # Bound in the body, not at module scope, for the reason spelled out just below: this
        # function's source is copied into the generated UnslothGRPOTrainer cache without
        # unsloth_zoo's module imports, so a module-level import reaches the import path and
        # not the one that actually runs in production.
        from contextlib import nullcontext

        if not _autocast_on:
            autocaster = nullcontext()
        else:
            autocaster = torch.amp.autocast(device_type = trainer.model.device.type, dtype = trainer._autocast_dtype)

        # PrefixGrouper grad path. This function's source is copied into the generated
        # UnslothGRPOTrainer cache without unsloth_zoo's module imports, so bind names
        # inside the body; the prefix_grouper import stays lazy + guarded (circular import,
        # may be absent) and a failed import just leaves PG off.
        from unsloth_zoo.temporary_patches.common import UNSLOTH_ENABLE_LOGGING

        # Memoize env gate + import once per process on the function object (which survives
        # into the cache). Env gate checked first so =0 never imports PG code; () = PG off.
        _pg_funcs = getattr(grpo_accumulated_loss, "_pg_funcs", None)
        if _pg_funcs is None:
            _pg_funcs = ()
            if os.environ.get("UNSLOTH_GRPO_PREFIX_GROUPER", "1").lower() not in (
                "0", "false", "no", "off",
            ):
                try:
                    from unsloth.utils.prefix_grouper import (
                        build_group_layout as _pg_build_layout,
                        prefix_grouper_enabled as _pg_enabled_fn,
                        verify_on as _pg_verify_on,
                        tol_ok as _pg_tol_ok,
                        TOL_KILL as _PG_TOL_KILL,
                    )
                    _pg_funcs = (
                        _pg_build_layout, _pg_enabled_fn, _pg_verify_on, _pg_tol_ok, _PG_TOL_KILL,
                    )
                except Exception:
                    _pg_funcs = ()
            grpo_accumulated_loss._pg_funcs = _pg_funcs
        # Skip PG under vLLM (fast_inference=True): rollout dominates the step, so the
        # saving is small and the first-use self-verify is net overhead.
        _pg_engage = bool(_pg_funcs) and not getattr(trainer, "use_vllm", False) and not _need_entropy

        # ---- PrefixGrouper (GRPO shared-prompt dedup; UNSLOTH_GRPO_PREFIX_GROUPER=0 disables) ----
        # Each prompt's G completions share the prefix; PG forwards it once + the G suffixes
        # (FlexAttention shared-prefix mask), cutting G*(P+R) tokens to P+G*R. First-use
        # self-verify vs the full-row packed new_logprobs; grads flow through the shared stream
        # (prefix grad once = sum of G repeats, identical math). Off/failed/unverified ->
        # full-row packed path runs as before.
        _pg_result = None
        _pg_use = False
        _pg_skip_pack = False
        _pg_num_gen = getattr(trainer, "num_generations", None)
        # Runtime gate; broad except -> engage False.
        if _pg_engage and _pg_funcs:
            try:
                _pg_build_layout, _pg_enabled_fn, _pg_verify_on, _pg_tol_ok, _PG_TOL_KILL = _pg_funcs
                # Exclusions: softcap models (gemma2) - the FlexAttention kernel skips
                # attn_logit_softcapping; hybrid SSM (FalconH1) and MoE (Qwen3-MoE) - their
                # decoders do not thread prefix_seg_info, so state would leak across suffixes.
                _pg_cfg = getattr(unwrapped_model, "config", None)
                _pg_engage = (
                    _pg_enabled_fn()
                    and pixel_values is None
                    and token_type_ids is None
                    and mm_token_type_ids is None
                    and _pg_num_gen is not None
                    and _pg_num_gen >= 2
                    and not getattr(_pg_cfg, "attn_logit_softcapping", None)
                    and not any(
                        getattr(_pg_cfg, _pg_a, None) is not None
                        for _pg_a in ("mamba_d_ssm", "mamba_d_state", "mamba_expand")
                    )
                    and not any(
                        getattr(_pg_cfg, _pg_a, None) is not None
                        for _pg_a in (
                            "num_experts", "num_experts_per_tok", "num_local_experts",
                            "n_routed_experts", "moe_intermediate_size",
                        )
                    )
                )
            except Exception:
                _pg_engage = False
        else:
            _pg_engage = False
        _pg_layout = None
        _pg_trusted = False   # signature already verified -> skip the full-row forward this step
        if _pg_engage:
            try:
                _pg_pad_id = trainer.processing_class.pad_token_id
                # Build the layout from the left-packed input_ids with the original left-pad
                # counts so the prefix/suffix split matches the packed path (_pack_cstart) and
                # the verify is apples-to-apples. Cap the PG span at any sliding window,
                # mirroring the packed _pack_sw guard.
                _pg_sw = getattr(getattr(unwrapped_model, "config", None), "sliding_window", None)
                if not (isinstance(_pg_sw, int) and _pg_sw > 0):
                    _pg_sw = None
                _pg_layout = _pg_build_layout(
                    input_ids, logits_to_keep, _pg_pad_id, _pg_num_gen, left_pad_tokens_per_prompt,
                    max_segment_cap = _pg_sw,
                )
                _pg_unsafe = getattr(unwrapped_model, "_unsloth_prefix_grouper_grad_unsafe", None)
                if _pg_unsafe is None:
                    _pg_unsafe = set()
                if _pg_layout is not None and _pg_layout.signature in _pg_unsafe:
                    _pg_layout = None
                elif _pg_layout is not None:
                    _pg_layout.W = logits_to_keep + max_left_pad
                    _pg_verified = getattr(unwrapped_model, "_unsloth_prefix_grouper_grad_verified", None)
                    # trust only if the verified envelope covers this batch's lengths
                    # (re-verify when T or the longest segment grows)
                    _pg_T = int(_pg_layout.flat_ids.shape[1])
                    _pg_maxseg = int(_pg_layout.position_ids.max()) + 1
                    _pg_env = (
                        _pg_verified.get(_pg_layout.signature)
                        if isinstance(_pg_verified, dict) else None
                    )
                    if (not _pg_verify_on()) or (
                        _pg_env is not None and _pg_T <= _pg_env[0] and _pg_maxseg <= _pg_env[1]
                    ):
                        _pg_trusted = True
                        _pg_skip_pack = True   # trusted shape -> skip the full-row forward
            except Exception as _pg_err:
                _pg_layout = None
                _pg_trusted = False
                _pg_skip_pack = False
                if isinstance(_pg_err, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
                if UNSLOTH_ENABLE_LOGGING:
                    print(f"[Unsloth] GRPO PrefixGrouper (grad) disabled (fell back to packed): {_pg_err!r}", flush = True)

        # ---- Sequence packing (default-on; disable with UNSLOTH_GRPO_SEQ_PACKING=0) ----
        # One varlen [1, sum L] block-diagonal forward replaces the padded [B, Lmax] loop: the exact per-row
        # result, and it fixes the padded path's left-pad RoPE error. Loss/gradients flow through it. Self-
        # verified against the per-row forward (shape/RoPE-aware, re-checked as T grows); falls back if a
        # backend ignores packed_seq_lengths. lm_head runs on completion positions only.
        new_logprobs = None
        _pack_result = None
        _pack_use = False
        _pack_T = None
        # True only while the packed forward and its scatter run, so an OOM elsewhere (setup, per-row verifier)
        # is not read as a length limit
        _pack_in_packed = False
        _pack_enabled = os.environ.get("UNSLOTH_GRPO_SEQ_PACKING", "1").lower() not in ("0", "false", "no", "off")
        _pack_ok = getattr(unwrapped_model, "_unsloth_seq_packing_grad_ok", None)
        if (_pack_enabled and not _pg_skip_pack and pixel_values is None
                and token_type_ids is None and mm_token_type_ids is None and _pack_ok is not False):
            try:
                _pack_pad_id = trainer.processing_class.pad_token_id
                _pack_keep = input_ids != _pack_pad_id
                _pack_lengths = _pack_keep.sum(dim = 1)
                _pack_lengths_cpu = _pack_lengths.tolist()                 # single GPU->CPU sync, reused below
                _pack_nz_cpu = [_n for _n in _pack_lengths_cpu if _n > 0]
                _pack_T = sum(_pack_nz_cpu)
                _pack_L = input_ids.shape[1]
                _pack_W = logits_to_keep + max_left_pad
                _pack_maxseg = max(_pack_nz_cpu) if _pack_nz_cpu else 0
                # sliding-window models lose the per-sequence local window in a packed stream
                _pack_sw = getattr(getattr(unwrapped_model, "config", None), "sliding_window", None)
                _pack_sw_ok = not (isinstance(_pack_sw, int) and _pack_sw > 0 and _pack_maxseg > _pack_sw)
                _pack_unsafe = getattr(unwrapped_model, "_unsloth_seq_packing_grad_unsafe_T", None)
                # skip the whole packed forward for a known-unsafe length region (a prior moderate mismatch or a
                # packed OOM), before the flatten and the active-row read, so a capped batch costs one sync
                _pack_capped = _pack_unsafe is not None and _pack_T >= _pack_unsafe
                if not _pack_capped:
                    _pack_flat_ids = input_ids[_pack_keep].unsqueeze(0)
                    _pack_active = int((completion_mask.sum(dim = 1) > 0).sum())
                if (not _pack_capped) and _pack_T >= 2 and len(_pack_nz_cpu) > 0 and _pack_sw_ok\
                        and (_pack_ok is True or _pack_active >= 2):
                    _pack_psl = torch.tensor(_pack_nz_cpu, dtype = torch.int32, device = input_ids.device)
                    # reset 0-based position_ids per segment
                    _pack_pos = (_pack_keep.cumsum(dim = 1) - 1)[_pack_keep].unsqueeze(0)
                    _pack_chunks = max(1, total_rows * multiplier)
                    _pack_nz_idx = _pack_keep.nonzero(as_tuple = False)            # [T, 2] = (row, col)
                    _pack_within = _pack_nz_idx[1:, 0] == _pack_nz_idx[:-1, 0]     # [T-1]
                    # completion start is per-row after left-packing: (L - logits_to_keep) minus that
                    # row's left-pad (matches create_completion_attention_mask exactly)
                    _pack_cstart = (_pack_L - logits_to_keep) - left_pad_tokens_per_prompt  # [rows]
                    _pack_ctgt = (_pack_nz_idx[1:, 1] >= _pack_cstart[_pack_nz_idx[1:, 0]]) & _pack_within
                    _pack_in_packed = True
                    with autocaster:
                        # use_cache=False: a KV cache silently disables varlen packing
                        _pack_hidden = unwrapped_model(
                            input_ids = _pack_flat_ids,
                            position_ids = _pack_pos,
                            packed_seq_lengths = _pack_psl,
                            use_cache = False,
                        ).logits
                        _refresh_lm_head()
                        # `.logits` carries hidden states only when the forward is the
                        # Unsloth generated one honouring UNSLOTH_RETURN_HIDDEN_STATES;
                        # otherwise it is real [T, vocab] logits and the lm_head matmul
                        # dies. Dispatch on width, as the padded path already does.
                        _pack_h   = _pack_hidden[0, :-1, :][_pack_ctgt].unsqueeze(0)
                        _pack_tid = _pack_flat_ids[0, 1:][_pack_ctgt].unsqueeze(0)
                        if _unsloth_grpo_returns_hidden_states(unwrapped_model, _pack_h, lm_head):
                            _fp32_bind()
                            if _need_entropy:
                                _pack_sel, _pack_ent = _hidden_logps_and_entropies(_pack_h, _pack_tid, _pack_chunks)
                                _pack_sel, _pack_ent = _pack_sel[0], _pack_ent[0]
                            else:
                                _pack_sel = chunked_hidden_states_selective_log_softmax(
                                    _pack_h, lm_head, _pack_tid, _pack_chunks,
                                    logit_scale_multiply, logit_scale_divide, logit_softcapping, temperature,
                                    lm_head_bias = lm_head_bias,
                                )[0]
                        else:
                            # Raw logits: the forward already applied scale/softcap.
                            _pack_sel = chunked_selective_log_softmax(
                                _pack_h, _pack_tid,
                                temperature = temperature, chunks = _pack_chunks,
                            )[0]
                            if _need_entropy:
                                _pack_ent = _raw_logits_entropies(_pack_h)[0]
                    # GPT-OSS offload race guard (matches the padded loop)
                    device_synchronize()
                    # scatter each completion logprob back to its (row, col) so [:, -_pack_W:] matches padded
                    _pack_tgt = (_pack_nz_idx[1:, 0] * _pack_L + _pack_nz_idx[1:, 1])[_pack_ctgt]
                    _pack_result = torch.zeros(
                        total_rows * _pack_L, dtype = torch.float32, device = input_ids.device,
                    ).index_put((_pack_tgt,), _pack_sel.to(torch.float32)).view(total_rows, _pack_L)[:, -_pack_W:]
                    if _need_entropy:
                        _pack_entropies = torch.zeros(
                            total_rows * _pack_L, dtype = torch.float32, device = input_ids.device,
                        ).index_put((_pack_tgt,), _pack_ent.to(torch.float32)).view(total_rows, _pack_L)[:, -_pack_W:]
                    _pack_in_packed = False
                    # trust decision: re-verify when T or the longest segment grows past what was verified
                    # (a LongRoPE cache switch can change the result)
                    _pack_vT = int(getattr(unwrapped_model, "_unsloth_seq_packing_grad_verified_T", 0))
                    _pack_vS = int(getattr(unwrapped_model, "_unsloth_seq_packing_grad_verified_seg", 0))
                    _pack_force_verify = os.environ.get("UNSLOTH_GRPO_SEQ_PACKING_VERIFY", "0") == "1"
                    if (not _pack_force_verify) and _pack_ok is True and _pack_T <= _pack_vT and _pack_maxseg <= _pack_vS:
                        _pack_use = True                                           # already verified for this shape
                    else:
                        # verify against the per-row clean forward (exact ground truth; no grad, value check)
                        _pack_ref = torch.zeros_like(_pack_result)
                        with torch.no_grad(), autocaster:
                            for _pack_i in range(total_rows):
                                _pack_ni = _pack_lengths_cpu[_pack_i]
                                if _pack_ni < 2: continue
                                _pack_rmask = _pack_keep[_pack_i]
                                _pack_real = input_ids[_pack_i][_pack_rmask].unsqueeze(0)
                                _pack_rpos = torch.arange(_pack_ni, device = input_ids.device).unsqueeze(0)
                                _pack_rh = unwrapped_model(input_ids = _pack_real, position_ids = _pack_rpos, use_cache = False).logits
                                _refresh_lm_head()
                                # same width dispatch as the packed call above: this forward
                                # returns raw logits whenever that one did, and the first
                                # packed batch always lands here
                                if _unsloth_grpo_returns_hidden_states(unwrapped_model, _pack_rh, lm_head):
                                    _fp32_bind()
                                    _pack_rsel = chunked_hidden_states_selective_log_softmax(
                                        _pack_rh[:, :-1, :], lm_head, _pack_real[:, 1:], 1,
                                        logit_scale_multiply, logit_scale_divide, logit_softcapping, temperature,
                                        lm_head_bias = lm_head_bias,
                                    )[0]
                                else:
                                    _pack_rsel = chunked_selective_log_softmax(
                                        _pack_rh[:, :-1, :], _pack_real[:, 1:],
                                        temperature = temperature, chunks = 1,
                                    )[0]
                                _pack_rcols = _pack_rmask.nonzero(as_tuple = False).squeeze(1)[1:] - (_pack_L - _pack_W)
                                _pack_rkeep = _pack_rcols >= 0
                                _pack_ref[_pack_i, _pack_rcols[_pack_rkeep]] = _pack_rsel[_pack_rkeep].to(torch.float32)
                        device_synchronize()
                        # compare over the exact loss-mask region (same mask the loss uses; pure
                        # create_completion_attention_mask, before any tool_mask is applied)
                        _pack_cm = create_completion_attention_mask(
                            input_ids[:, -_pack_W:], left_pad_tokens_per_prompt, max_left_pad, _pack_pad_id
                        ).float()
                        _pack_diff = float(((_pack_result.detach() - _pack_ref).abs() * _pack_cm).max())
                        if UNSLOTH_ENABLE_LOGGING:
                            print(f"[Unsloth] GRPO seq-packing (grad) verify: T={_pack_T} maxseg={_pack_maxseg} packed-vs-perrow max|d|={_pack_diff:.4f}", flush = True)
                        # floor ~0.25 through different kernels; cross-sample contamination is >= 2.4
                        if _pack_diff < 7e-1:
                            unwrapped_model._unsloth_seq_packing_grad_ok = True
                            # only widen the trusted shape when >= 2 completion rows actually exercised
                            # cross-sample packing; a < 2 row pass proves nothing, so keep re-verifying
                            # larger shapes until a real multi-row batch clears them
                            if _pack_active >= 2:
                                unwrapped_model._unsloth_seq_packing_grad_verified_T = max(_pack_vT, _pack_T)
                                unwrapped_model._unsloth_seq_packing_grad_verified_seg = max(_pack_vS, _pack_maxseg)
                            _pack_ok = True
                            _pack_use = True
                        else:
                            _pack_use = False
                            if _pack_diff >= 1.5:
                                # large mismatch = contamination (attention ignores the packed mask, e.g.
                                # some MoE): disable packing for this model
                                unwrapped_model._unsloth_seq_packing_grad_ok = False
                            else:
                                # moderate mismatch -> likely a length boundary (LongRoPE): mark unsafe but
                                # keep packing for smaller shapes
                                unwrapped_model._unsloth_seq_packing_grad_unsafe_T = (
                                    _pack_T if _pack_unsafe is None else min(_pack_unsafe, _pack_T)
                                )
                            if UNSLOTH_ENABLE_LOGGING:
                                print(f"[Unsloth] GRPO seq-packing (grad) fell back at T={_pack_T} (diff={_pack_diff:.3f})", flush = True)
            except Exception as _pack_err:
                # any failure -> drop intermediates, use the padded loop, do not retry
                _pack_hidden = None
                _pack_sel = None
                _pack_result = None
                _pack_use = False
                if isinstance(_pack_err, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                if isinstance(_pack_err, torch.cuda.OutOfMemoryError) and _pack_in_packed and _pack_T is not None:
                    # an OOM in the packed forward means this length does not fit, not that packing is wrong
                    _pack_unsafe = getattr(unwrapped_model, "_unsloth_seq_packing_grad_unsafe_T", None)
                    _pack_unsafe = _pack_T if _pack_unsafe is None else min(_pack_unsafe, _pack_T)
                    unwrapped_model._unsloth_seq_packing_grad_unsafe_T = _pack_unsafe
                    if UNSLOTH_ENABLE_LOGGING:
                        print(f"[Unsloth] GRPO sequence-packing OOM at T={_pack_T}: packing capped below T={_pack_unsafe}, this batch uses the padded loop: {_pack_err!r}", flush = True)
                else:
                    unwrapped_model._unsloth_seq_packing_grad_ok = False
                    if UNSLOTH_ENABLE_LOGGING:
                        print(f"[Unsloth] GRPO sequence-packing disabled (fell back to padded): {_pack_err!r}", flush = True)
        # ---- PrefixGrouper resolution + first-use self-verify (grad) ----
        # Verify runs under no_grad, then a separate grad forward builds new_logprobs,
        # so no inference tensors are saved for backward.
        def _pg_grad_forward():
            _pg_chunks = max(1, total_rows * multiplier)
            with autocaster:
                _h = unwrapped_model(
                    input_ids = _pg_layout.flat_ids,
                    position_ids = _pg_layout.position_ids,
                    prefix_seg_info = _pg_layout.prefix_seg_info,
                    use_cache = False,
                ).logits
                _refresh_lm_head()
                # Same width dispatch as the packed path and compute_logprobs_chunk.
                # `.logits` carries hidden states only when the forward is the Unsloth
                # generated one honouring UNSLOTH_RETURN_HIDDEN_STATES; otherwise it is
                # real [T, vocab] logits. extract_logps always calls its helper as
                # (hidden, lm_head, ids, chunks, ...), so pass a raw-logits helper with
                # that same signature, which skips the lm_head matmul and the scale /
                # softcap the forward already applied.
                if not _unsloth_grpo_returns_hidden_states(unwrapped_model, _h, lm_head):
                    def _pg_fn(_pg_h, _pg_lm, _pg_ids, _pg_n, _pg_lsm, _pg_lsd, _pg_lsc, _pg_t):
                        return chunked_selective_log_softmax(
                            _pg_h, _pg_ids, temperature = _pg_t, chunks = _pg_n,
                        )
                else:
                    _fp32_bind()
                    _pg_fn = chunked_hidden_states_selective_log_softmax
                    if lm_head_bias is not None:
                        from functools import partial
                        _pg_fn = partial(_pg_fn, lm_head_bias = lm_head_bias)
                _pg_lp = _pg_layout.extract_logps(
                    _h, lm_head, _pg_fn,
                    _pg_chunks, logit_scale_multiply, logit_scale_divide,
                    logit_softcapping, temperature,
                )  # [total_rows, W] with grad
                # GPT-OSS offload race guard
                device_synchronize()
                return _pg_lp

        if _pg_layout is not None:
            # A verify-phase OOM (packed graph co-resident) does not prove PG alone cannot fit;
            # only an OOM after the packed graph is freed is worth marking unsafe.
            _pg_phase_verify = False
            try:
                if not _pg_trusted:
                    # first use: verify vs the packed new_logprobs. < tol_ok -> trust;
                    # >= TOL_KILL -> unsafe forever; borderline -> fall back this shape.
                    if _pack_use and _pack_result is not None:
                        _pg_phase_verify = True   # packed graph still co-resident
                        with torch.no_grad():
                            _pg_ref = _pg_grad_forward()
                        _pg_W2 = logits_to_keep + max_left_pad
                        _pg_cm = create_completion_attention_mask(
                            input_ids[:, -_pg_W2:], left_pad_tokens_per_prompt, max_left_pad,
                            trainer.processing_class.pad_token_id,
                        ).float()
                        _pg_a = _pg_ref[:, -_pg_W2:].float()
                        _pg_b = _pack_result.detach()[:, -_pg_W2:].float()
                        _pg_diff = float(((_pg_a - _pg_b).abs() * _pg_cm).max())
                        if UNSLOTH_ENABLE_LOGGING:
                            print(
                                f"[Unsloth] GRPO PrefixGrouper (grad) verify: sig={_pg_layout.signature} "
                                f"shared-prefix vs full-row-packed max|d|={_pg_diff:.4f}", flush = True,
                            )
                        if _pg_diff < _pg_tol_ok():
                            _pg_v = getattr(unwrapped_model, "_unsloth_prefix_grouper_grad_verified", None)
                            if not isinstance(_pg_v, dict):
                                _pg_v = {}
                            _pg_vT = int(_pg_layout.flat_ids.shape[1])
                            _pg_vS = int(_pg_layout.position_ids.max()) + 1
                            _pg_old = _pg_v.get(_pg_layout.signature, (0, 0))
                            _pg_v[_pg_layout.signature] = (
                                max(_pg_vT, _pg_old[0]), max(_pg_vS, _pg_old[1]),
                            )
                            unwrapped_model._unsloth_prefix_grouper_grad_verified = _pg_v
                            _pg_trusted = True
                        else:
                            _pg_u = getattr(unwrapped_model, "_unsloth_prefix_grouper_grad_unsafe", None)
                            if _pg_u is None:
                                _pg_u = set()
                            if _pg_diff >= _PG_TOL_KILL:
                                _pg_u.add(_pg_layout.signature)
                                unwrapped_model._unsloth_prefix_grouper_grad_unsafe = _pg_u
                            _pg_trusted = False
                    # else: no packed reference -> cannot verify -> fall back.
                if _pg_trusted:
                    # free the packed graph BEFORE the grad forward: holding both can OOM when
                    # PG alone would fit, and on PG failure the padded loop recomputes anyway.
                    _pack_hidden = _pack_sel = _pack_result = None
                    _pg_phase_verify = False   # packed freed: an OOM below is PG-alone
                    _pg_result = _pg_grad_forward()
                    _pg_use = True
            except Exception as _pg_err2:
                _pg_use = False
                os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
                # untrust this signature so the next batch runs the packed path again
                _pg_v = getattr(unwrapped_model, "_unsloth_prefix_grouper_grad_verified", None)
                if isinstance(_pg_v, dict):
                    _pg_v.pop(_pg_layout.signature, None)
                if isinstance(_pg_err2, torch.cuda.OutOfMemoryError):
                    # mark unsafe only for a PG-alone OOM (deterministic at these lengths);
                    # a verify-phase OOM (packed co-resident) proves nothing, just retry.
                    if not _pg_phase_verify:
                        _pg_u = getattr(unwrapped_model, "_unsloth_prefix_grouper_grad_unsafe", None)
                        if _pg_u is None:
                            _pg_u = set()
                        _pg_u.add(_pg_layout.signature)
                        unwrapped_model._unsloth_prefix_grouper_grad_unsafe = _pg_u
                    torch.cuda.empty_cache()
                if UNSLOTH_ENABLE_LOGGING:
                    print(f"[Unsloth] GRPO PrefixGrouper (grad) forward failed -> packed/padded fallback: {_pg_err2!r}", flush = True)

        if _pg_use and _pg_result is not None:
            new_logprobs = _pg_result            # PrefixGrouper verified -> skip the loop
            zipped_inputs = []
        elif _pack_use and _pack_result is not None:
            new_logprobs = _pack_result          # verified -> skip the loop
            if _need_entropy:
                all_entropies_list = [_pack_entropies]
            zipped_inputs = []
        else:
            # packing rejected/unused: drop the packed graph and its index buffers before the padded loop so
            # neither co-resides with it
            _pack_hidden = _pack_sel = _pack_result = None
            _pack_ent = _pack_entropies = None
            if _pack_T is not None:
                _pack_keep = _pack_lengths = _pack_flat_ids = _pack_pos = _pack_psl = None
                _pack_nz_idx = _pack_within = _pack_cstart = _pack_ctgt = _pack_tgt = None
                _pack_h = _pack_tid = _pack_ref = _pack_cm = None

        def to_device(tensor, device, non_blocking=True):
            if tensor is None: return None
            return tensor.to(device, non_blocking=non_blocking)

        def _offload_device_module(tensor_or_device):
            # Stream/Event module for the offload copy. torch.cuda is also the HIP
            # backend, so ROCm reports is_cuda and needs no branch of its own; XPU has
            # its own namespace, matching gradient_checkpointing.py. Anything else
            # (CPU, MPS, ...) returns None and takes the pageable copy.
            device = getattr(tensor_or_device, "device", tensor_or_device)
            if device.type == "cuda": return torch.cuda
            if device.type == "xpu": return getattr(torch, "xpu", None)
            return None

        class Unsloth_Offloaded_Log_Softmax(torch.autograd.Function):
            """Manual gradient checkpointing / CPU offloading for log softmax."""
            @staticmethod
            def forward(ctx, hidden_states, lm_head, index, chunks,
                        logit_scale_multiply, logit_scale_divide,
                        logit_softcapping, temperature, lm_head_bias = None):
                # Detach so we don't keep the graph (and extra memory) on CPU.
                detached_hidden_states = hidden_states.detach().contiguous()
                ctx.device = hidden_states.device
                ctx.copy_event = None
                # Backward runs outside autocast; recompute must match forward (torch.utils.checkpoint does the same).
                ctx.autocast_kwargs = dict(
                    device_type = lm_head.device.type,
                    enabled = torch.is_autocast_enabled(lm_head.device.type),
                    dtype = torch.get_autocast_dtype(lm_head.device.type),
                    cache_enabled = torch.is_autocast_cache_enabled(),
                )

                # Always offload: this path only runs when the caller is already memory bound
                # (long completions / large batches), so the win is overlapping the copy.
                saved_hidden_states = None
                device_module = _offload_device_module(detached_hidden_states)
                if device_module is not None:
                    # Async D2H on a side stream; backward MUST wait on copy_event before
                    # the H2D reload or it races the copy.
                    try:
                        pinned_buffer = torch.empty_like(detached_hidden_states, device = "cpu", pin_memory = True)
                        if pinned_buffer is not None:
                            current_stream = device_module.current_stream(detached_hidden_states.device)
                            copy_stream = device_module.Stream(device = detached_hidden_states.device)
                            copy_stream.wait_stream(current_stream)
                            with device_module.stream(copy_stream):
                                pinned_buffer.copy_(detached_hidden_states, non_blocking = True)
                            # Keeps the GPU storage alive until the side-stream copy finishes.
                            detached_hidden_states.record_stream(copy_stream)
                            copy_event = device_module.Event()
                            copy_event.record(copy_stream)
                            saved_hidden_states = pinned_buffer
                            ctx.copy_event = copy_event
                    except (RuntimeError, OSError, AttributeError):
                        # Any accelerator that cannot do pinned side-stream copies falls
                        # back below; correctness never depends on this path.
                        saved_hidden_states = None
                        ctx.copy_event = None
                if saved_hidden_states is None:
                    # No accelerator, or the async copy is unavailable: pageable copy.
                    saved_hidden_states = detached_hidden_states.to("cpu", non_blocking = True)
                ctx.saved_hidden_states = saved_hidden_states
                # Drop the clone before the log-softmax below. hidden_states is usually a
                # [:, :-1, :] slice, so .contiguous() allocated a full copy; holding the
                # reference across the forward would keep it resident alongside the chunk
                # logits. record_stream still blocks reuse until the D2H lands, so the
                # allocator reclaims it mid-compute rather than at the end of forward.
                del detached_hidden_states

                ctx.lm_head = lm_head
                ctx.lm_head_requires_grad = lm_head.requires_grad
                ctx.lm_head_bias = lm_head_bias
                ctx.bias_requires_grad = lm_head_bias is not None and lm_head_bias.requires_grad
                ctx.index = index
                ctx.args = (chunks, logit_scale_multiply, logit_scale_divide, logit_softcapping, temperature)

                bias_kwargs = {} if lm_head_bias is None else {"lm_head_bias": lm_head_bias}
                with torch.no_grad():
                    output = chunked_hidden_states_selective_log_softmax(
                        hidden_states, lm_head, index, *ctx.args, **bias_kwargs,
                    )

                return output

            @staticmethod
            def backward(ctx, grad_output):
                if ctx.copy_event is not None:
                    # The offload copy must land before the H2D reload.
                    device_module = _offload_device_module(ctx.device)
                    ctx.copy_event.wait(device_module.current_stream(ctx.device))
                hidden_states = to_device(ctx.saved_hidden_states, ctx.device)
                hidden_states.requires_grad_(True)

                lm_head = ctx.lm_head
                if ctx.lm_head_requires_grad:
                    # Recompute against a private leaf. A Tensor.register_hook on the real
                    # lm_head fires for tensors named in autograd.grad's inputs, so reusing
                    # it here would run a user's grad mask / scaler once on this local
                    # gradient and again when the returned gradient reaches lm_head.
                    lm_head = lm_head.detach().requires_grad_(True)
                lm_head_bias = ctx.lm_head_bias
                if ctx.bias_requires_grad:
                    lm_head_bias = lm_head_bias.detach().requires_grad_(True)
                index = ctx.index

                bias_kwargs = {} if lm_head_bias is None else {"lm_head_bias": lm_head_bias}
                with torch.enable_grad(), torch.autocast(**ctx.autocast_kwargs):
                    output = chunked_hidden_states_selective_log_softmax(
                        hidden_states, lm_head, index, *ctx.args, **bias_kwargs,
                    )

                # autograd.grad, not backward: backward writes into leaf .grad, which the
                # outer AccumulateGrad would then double-count.
                inputs = [hidden_states]
                if ctx.lm_head_requires_grad:
                    inputs.append(lm_head)
                if ctx.bias_requires_grad:
                    inputs.append(lm_head_bias)
                grad_inputs = torch.autograd.grad(output, inputs, grad_output)

                return (
                    grad_inputs[0],
                    grad_inputs[1] if ctx.lm_head_requires_grad else None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    grad_inputs[-1] if ctx.bias_requires_grad else None,
                )

        def efficient_log_softmax(hidden_states, lm_head, index, chunks=32,
                                logit_scale_multiply=0.0, logit_scale_divide=0.0,
                                logit_softcapping=0.0, temperature=1, batch_size=8,
                                lm_head_bias=None):
            # The old chunked scorer keeps every chunk's logits, so long completions keep the offloaded scorer.
            from unsloth_zoo.rl_replacements import grpo_scorer_keeps_logits as _keeps_logits
            _plain = os.environ.get("UNSLOTH_GRPO_LOGPROB_RECOMPUTE", "0") != "1"
            if not ((index.shape[1] <= 1024 and batch_size <= 8) or batch_size == 1)\
                    and (bool(_support_kwargs) or (_plain and _keeps_logits())):
                return Unsloth_Offloaded_Log_Softmax.apply(
                    hidden_states, lm_head, index, chunks,
                    logit_scale_multiply, logit_scale_divide,
                    logit_softcapping, temperature, lm_head_bias,
                )
            return chunked_hidden_states_selective_log_softmax(
                hidden_states,
                lm_head,
                index,
                chunks,
                logit_scale_multiply,
                logit_scale_divide,
                logit_softcapping,
                temperature,
                lm_head_bias = lm_head_bias,
            )

        def compute_logprobs_chunk(new_hidden_states_chunk, completion_ids, input_ids_chunk):
            _refresh_lm_head()
            # Hidden states -> lm_head matmul path; raw logits -> skip matmul and
            # skip scale/softcap (model forward already applied them).
            chunks = input_ids_chunk.shape[0] * multiplier
            if _unsloth_grpo_returns_hidden_states(unwrapped_model, new_hidden_states_chunk, lm_head):
                _fp32_bind()
                if _need_entropy:
                    logps, entropies = _hidden_logps_and_entropies(new_hidden_states_chunk, completion_ids, chunks)
                    all_entropies_list.append(entropies)
                    return logps
                return efficient_log_softmax(
                    new_hidden_states_chunk,
                    lm_head,
                    completion_ids,
                    chunks = chunks,
                    logit_scale_multiply = logit_scale_multiply,
                    logit_scale_divide = logit_scale_divide,
                    logit_softcapping = logit_softcapping,
                    temperature = temperature,
                    batch_size = B,
                    lm_head_bias = lm_head_bias,
                )
            if _need_entropy:
                all_entropies_list.append(_raw_logits_entropies(new_hidden_states_chunk))
            return chunked_selective_log_softmax(
                new_hidden_states_chunk,
                completion_ids,
                temperature = temperature,
                chunks = chunks,
            )
        for (
            input_ids_chunk,
            attention_mask_chunk,
            vision_chunk,
            completion_ids
        ) in zipped_inputs:
                with autocaster:
                    if pixel_values is None:
                        new_hidden_states_chunk = unwrapped_model(
                            input_ids = input_ids_chunk,
                            attention_mask = attention_mask_chunk,
                            **vision_chunk,
                        ).logits

                        new_hidden_states_chunk = new_hidden_states_chunk[:, -(logits_to_keep + max_left_pad + 1): , :]
                        new_hidden_states_chunk = new_hidden_states_chunk[:, :-1, :]
                        logprobs_chunk = compute_logprobs_chunk(new_hidden_states_chunk, completion_ids, input_ids_chunk)
                    else:
                        new_hidden_states_chunk = unwrapped_model(
                            input_ids = input_ids_chunk,
                            attention_mask = attention_mask_chunk,
                            logits_to_keep = logits_to_keep + 1,
                            **vision_chunk,
                        ).logits

                        new_hidden_states_chunk = new_hidden_states_chunk[:, :-1, :]
                        logprobs_chunk = compute_logprobs_chunk(new_hidden_states_chunk, completion_ids, input_ids_chunk)
                    # Avoids race conditions with GPT OSS offload_embbed=True; no measurable slowdown.
                    device_synchronize()
                all_logprobs_list.append(logprobs_chunk)

        if new_logprobs is None:
            # padded fallback (packing disabled / unsupported / not verified for this length)
            new_logprobs = torch.cat(all_logprobs_list, dim=0)

        if kept_completion_rows is not None:
            completion_mask = completion_mask * kept_completion_rows.to(
                device = completion_mask.device, dtype = completion_mask.dtype,
            )

        entropy_mask = None
        if _need_entropy:
            new_entropies = torch.cat(all_entropies_list, dim = 0)
            if _entropy_mask_on:
                # TRL's own method, so its quantile population, ties and empty-batch rule match per version
                # (multi-process TRL < 0.24 takes the padded gather so an empty rank cannot hang the rest).
                entropy_mask = _grpo_high_entropy_mask(
                    trainer, new_entropies.detach(), completion_mask, 1 - _entropy_quantile,
                    sample_weight = kwargs.get("sample_weight", None),
                )
                kwargs["entropy_mask"] = entropy_mask

        # DeepSpeed (no scaler) only: undo training_step's 1/GAS, which transformers >= 5.19 skips when
        # loss_is_scaled_for_ga (TRL >= 1.15 GRPO) and older ones when compute_loss_func is set.
        upstream_scale = None
        if trainer.accelerator.scaler is None and getattr(trainer, "is_deepspeed_enabled", False):
            import transformers as _unsloth_transformers
            loss_is_scaled_for_ga = None
            if hasattr(_unsloth_transformers.Trainer, "loss_is_scaled_for_ga"):
                loss_is_scaled_for_ga = getattr(trainer, "loss_is_scaled_for_ga", None)
            if loss_is_scaled_for_ga is None:
                loss_is_scaled_for_ga = getattr(trainer, "compute_loss_func", None) is not None
            upstream_scale = 1.0
            if not loss_is_scaled_for_ga:
                upstream_scale = float(kwargs.get("current_gradient_accumulation_steps", 1))

        _rollout_unscored = None
        if _rollout_logps_as_old:
            old_logps, _rollout_unscored = grpo_rollout_old_fill_unscored(old_logps, new_logprobs)
            # Only on this path: grpo_compute_loss clamps pi_theta / pi_rollout's log to [-20, 20] like verl.
            kwargs["rollout_logps_as_old"] = True
        # The rollout_rs keep mask grpo_compute_loss applies, for the IS metric weights and the entropy bonus.
        _rs_keep = None
        if kwargs.get("rollout_rs", None) is not None and old_logps is not None and old_logps.shape == completion_mask.shape:
            _rs_log_ratio = None
            if _rollout_logps_as_old:
                _rs_log_ratio = new_logprobs.detach() - old_logps
            elif trainer.use_vllm and sampling_per_token_logps is not None and kwargs.get("vllm_importance_sampling_correction", False):
                _rs_log_ratio = old_logps - sampling_per_token_logps
            if _rs_log_ratio is not None:
                from unsloth_zoo.rl_replacements import grpo_rollout_rs_keep as _grpo_rollout_rs_keep
                _rs_keep = _grpo_rollout_rs_keep(_rs_log_ratio, completion_mask, kwargs["rollout_rs"])
        with autocaster:
            loss, completion_length, mean_kl, delta, flat_is_ratio, coef_1 = UnslothEfficientGRPO.apply(
                new_logprobs,
                old_logps,
                ref_logps,
                sampling_per_token_logps,
                lm_head,
                completion_input_ids,
                completion_mask,
                advantages,
                trainer.beta,
                trainer.accelerator.scaler,
                1,
                kwargs,
                upstream_scale,
            )
        _rollout_stats_args = None
        if _rollout_logps_as_old:
            # Unclamped log(pi_theta / pi_rollout) for the KL diagnostics; the loss ratio is clamped to [-20, 20].
            _raw_log_ratio = new_logprobs.detach().to(torch.float64) - old_logps.detach().to(torch.float64)
            if _raw_log_ratio.shape != completion_mask.shape:
                _raw_log_ratio = torch.log(coef_1.detach().to(torch.float64))
            delta, flat_is_ratio = grpo_rollout_as_old_ratio_metrics(coef_1, completion_mask, _raw_log_ratio)
            if _rs_keep is not None and flat_is_ratio is not None:
                # Rejected tokens get IS weight 0, as on the vLLM IS path: rejected_frac counts them, ESS drops them.
                flat_is_ratio = torch.where(_rs_keep, flat_is_ratio, torch.zeros_like(flat_is_ratio))
            if _rollout_metrics_on and flat_is_ratio is not None:
                _log_ratio = torch.nan_to_num(_raw_log_ratio, nan = 0.0, posinf = 670.0, neginf = -670.0)
                _log_ratio = torch.where(completion_mask.to(torch.bool), _log_ratio, torch.zeros_like(_log_ratio))
                if _rollout_unscored is not None:
                    # Unscored tokens carry no mismatch information: NaN leaves them out of the sums.
                    _log_ratio = torch.where(_rollout_unscored, torch.full_like(_log_ratio, float("nan")), _log_ratio)
                # An overflowed ratio is capped (finite squares) so ESS tends to its 1 / N limit, not nan.
                _ess_weights = torch.clamp(flat_is_ratio.to(torch.float64), max = 1e100)
                _rollout_stats_args = (_log_ratio, torch.zeros_like(_log_ratio), completion_mask, _ess_weights)
        elif (
            _rollout_metrics_on and old_logps is not None and sampling_per_token_logps is not None
            and flat_is_ratio is not None and flat_is_ratio.numel() > 0
        ):
            _rollout_stats_args = (old_logps, sampling_per_token_logps, completion_mask, flat_is_ratio)
        if _rollout_stats_args is not None:
            from unsloth_zoo.rl_replacements import (
                grpo_rollout_mismatch_stats as _grpo_rollout_mismatch_stats,
                grpo_rollout_mismatch_metrics as _grpo_rollout_mismatch_metrics,
            )
            _rollout_stats = _grpo_rollout_mismatch_stats(*_rollout_stats_args)
            _rollout_stats = trainer.accelerator.gather(_rollout_stats.reshape(1, -1)).reshape(-1, _rollout_stats.numel())
            _rollout_stats = torch.cat([_rollout_stats[:, :3].sum(0), _rollout_stats[:, 3:4].amax(0), _rollout_stats[:, 4:].sum(0)])
            _metrics = trainer._metrics
            _bucket = "train" if _training else "eval"
            if "train" in _metrics:
                _metrics = _metrics[_bucket]
            # Pool the sums over every micro-batch since the last log instead of averaging
            # per-micro-batch ratios: TRL's log() clears _metrics, which restarts the window.
            _rollout_acc = getattr(trainer, "_unsloth_rollout_mismatch_stats", None)
            if not isinstance(_rollout_acc, dict):
                _rollout_acc = {}
                trainer._unsloth_rollout_mismatch_stats = _rollout_acc
            _prev = _rollout_acc.get(_bucket)
            if _prev is not None and "sampling/rollout_kl_k1/mean" in _metrics:
                _prev = _prev.to(_rollout_stats.device)
                _rollout_stats = torch.cat([
                    _prev[:3] + _rollout_stats[:3],
                    torch.maximum(_prev[3:4], _rollout_stats[3:4]),
                    _prev[4:] + _rollout_stats[4:],
                ])
            _rollout_acc[_bucket] = _rollout_stats.detach()
            for _name, _value in _grpo_rollout_mismatch_metrics(_rollout_stats).items():
                _metrics[_name] = [_value]
        if _entropy_bonus_on:
            _entropy_weights = _grpo_entropy_bonus_weights(
                completion_mask, entropy_mask, _entropy_coef, kwargs, training = _training,
            )
            if _rs_keep is not None:
                # Rejected tokens leave the entropy bonus too (verl drops them from response_mask);
                # its normaliser keeps them, like the policy term's.
                _entropy_weights = torch.where(_rs_keep.to(_entropy_weights.device), _entropy_weights, 0.0)
            _scaler = trainer.accelerator.scaler
            loss = loss + _UnslothGRPOEntropyBonus.apply(
                new_entropies, _entropy_weights.to(new_entropies.device),
                _scaler.get_scale() if _scaler is not None else 1.0, upstream_scale,
            )
    finally:
        if _unsloth_prior_hidden_states is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = _unsloth_prior_hidden_states

    # Force logits (not hidden states) again or output is gibberish.
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"

    return loss, completion_length, mean_kl, delta, flat_is_ratio, coef_1, completion_mask

from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback
@torch_compile_with_fallback(dynamic = True, fullgraph = True, options = torch_compile_options)
def grpo_compute_loss_slow(
    ref,
    new,
    old,
    sampling_per_token_logps,
    input_ids,
    mask,
    beta,
    advantages,
    **kwargs
):
    # All Unsloth Zoo code licensed under AGPL3
    # Optional argument defaults.
    loss_type = kwargs.get("loss_type", "grpo")
    epsilon_low = kwargs.get("epsilon_low", 0.2)
    epsilon_high = kwargs.get("epsilon_high", 0.2)
    max_completion_length = kwargs.get("max_completion_length", 8192)
    delta = kwargs.get("delta", None)
    importance_sampling_level = kwargs.get("importance_sampling_level", "token")
    num_items_in_batch = kwargs.get("num_items_in_batch", None)
    current_gradient_accumulation_steps = kwargs.get("current_gradient_accumulation_steps", 1)
    steps_per_generation = kwargs.get("steps_per_generation", None)
    num_processes = kwargs.get("num_processes", 1)
    use_vllm = kwargs.get("use_vllm", False)
    # The off-policy mask uses vLLM sampling logprobs whenever the batch supplies them (matching TRL);
    # the vLLM importance-sampling ratio is applied to the loss only when this flag is on.
    vllm_importance_sampling_correction = kwargs.get("vllm_importance_sampling_correction", False)
    vllm_importance_sampling_mode = kwargs.get("vllm_importance_sampling_mode", "sequence_mask")
    vllm_importance_sampling_cap = kwargs.get("vllm_importance_sampling_cap", 2.0)
    vllm_importance_sampling_clip_min = kwargs.get("vllm_importance_sampling_clip_min", None)
    vllm_importance_sampling_clip_max = kwargs.get("vllm_importance_sampling_clip_max", 3.0)
    get_sapo_token_loss = kwargs.get("get_sapo_token_loss", None)
    sapo_temperature_pos = kwargs.get("sapo_temperature_pos", 1.0)
    sapo_temperature_neg = kwargs.get("sapo_temperature_neg", 1.05)
    get_gamma_weights = kwargs.get("get_gamma_weights", None)
    vespo_k_pos = kwargs.get("vespo_k_pos", 2.0)
    vespo_lambda_pos = kwargs.get("vespo_lambda_pos", 3.0)
    vespo_k_neg = kwargs.get("vespo_k_neg", 3.0)
    vespo_lambda_neg = kwargs.get("vespo_lambda_neg", 2.0)
    get_off_policy_mask = kwargs.get("get_off_policy_mask", None)
    off_policy_mask_threshold  = kwargs.get("off_policy_mask_threshold", None)
    # TRL top_entropy_quantile: the policy term (not the KL) on high-entropy tokens only.
    entropy_mask = kwargs.get("entropy_mask", None)
    # Parsed (option, lower_log, upper) tuples from grpo_parse_rollout_rs, or None.
    rollout_rs = kwargs.get("rollout_rs", None)
    # Per-completion weights from unsloth's generation step (ScaleRL zero-variance exclusion /
    # prompt-level aggregation), normalised over the whole generation batch; replaces the loss_type normaliser.
    sample_weight = kwargs.get("sample_weight", None)
    # Only direct callers see this fallback; the trainer always forwards an explicit value.
    use_bias_correction_kl = kwargs.get("use_bias_correction_kl", False)
    input_ids = input_ids.unsqueeze(-1)

    importance_sampling_ratio = None
    # Bool (B, T) keep mask from rollout_rs; None when rejection is off.
    rs_keep = None

    # exp(new - old) and exp(ref - new) below are taken before `mask` is applied. A sequence-packed
    # logp path leaves the masked (prompt/pad) columns at 0 while a padded one fills them with a real
    # logp, so when new and old/ref disagree there those ratios can overflow to inf and inf * 0 (the
    # masked-out loss) becomes nan. Force new/old/ref to share 0 on the masked columns so both ratios
    # are exp(0) = 1 there; every loss term below multiplies by `mask`, so this changes nothing.
    if mask is not None:
        _keep = mask.to(torch.bool)
        new = torch.where(_keep, new, 0.0)
        if old is not None: old = torch.where(_keep, old, 0.0)
        if ref is not None: ref = torch.where(_keep, ref, 0.0)

    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)

    if off_policy_mask_threshold is not None:
        # DeepSeek-V3.2 off-policy mask. The mismatch logprobs are sampling_per_token_logps (vLLM
        # sampling logprobs) if present, else old, else new.detach() when both are absent
        # (num_iterations == 1 with no vLLM). This mirrors TRL, which defaults old_per_token_logps to
        # per_token_logps.detach() so get_off_policy_mask never receives None (it computes
        # mismatch - per_token_logps.detach(), so new.detach() yields a zero-KL keep-all mask). The
        # callable is a signature-stable adapter installed in grpo_accumulated_loss, so this stays
        # fixed across TRL versions with no signature introspection inside this compiled function.
        off_policy_mask = get_off_policy_mask(
            advantages=advantages,
            per_token_logps=new,
            sampling_per_token_logps=sampling_per_token_logps if sampling_per_token_logps is not None else (old if old is not None else new.detach()),
            mask=mask,
            off_policy_threshold=off_policy_mask_threshold,
        )

    with torch.no_grad():
        if use_vllm and sampling_per_token_logps is not None and vllm_importance_sampling_correction:
            # Filter out extra leading prompt tokens after left-padding input_ids.
            # Match TRL: aggregate log-ratios then exp (product), not sum of exp ratios.
            importance_sampling_ratio = (old - sampling_per_token_logps) * mask
            # Unscored vLLM tokens arrive as nan and nan * 0 survives the mask: ratio 1, as TRL does.
            importance_sampling_ratio = torch.nan_to_num(importance_sampling_ratio, nan = 0.0)

            if vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]:
                importance_sampling_ratio = importance_sampling_ratio.sum(dim=-1, keepdim=True)
            elif vllm_importance_sampling_mode in ["geometric_mask", "geometric_truncate"]:
                # Geometric mean of the token ratios (verl seq_mean_k1): the product blows up on long rows.
                importance_sampling_ratio = importance_sampling_ratio.sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).clamp(min=1.0)

            importance_sampling_ratio = torch.exp(importance_sampling_ratio)

            if vllm_importance_sampling_mode in ["token_truncate", "sequence_truncate", "geometric_truncate"]:
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, 
                    min=vllm_importance_sampling_clip_min,
                    max=vllm_importance_sampling_clip_max
                )
            elif vllm_importance_sampling_mode in ["token_mask", "sequence_mask", "geometric_mask"]:
                min_val = (
                    vllm_importance_sampling_clip_min
                    if vllm_importance_sampling_clip_min is not None
                    else -math.inf
                )

                max_val = (
                    vllm_importance_sampling_clip_max
                    if vllm_importance_sampling_clip_max is not None
                    else math.inf
                )

                invalid_mis_mask = (importance_sampling_ratio < min_val) | (
                        importance_sampling_ratio > max_val
                )

                importance_sampling_ratio = importance_sampling_ratio.masked_fill(
                        invalid_mis_mask, value=0.0
                )
            else:
                raise ValueError(
                        f"Unknown vLLM importance sampling mode: {vllm_importance_sampling_mode}. Possible values are 'token_truncate', 'token_mask', 'sequence_truncate', 'sequence_mask', 'geometric_truncate' and 'geometric_mask'."
                )

        # verl compute_rollout_rejection_mask: every option must pass. verl then replaces
        # response_mask with response_mask * keep, so a rejected token leaves the policy term
        # and the KL loss (both selected out below). It stays in the loss normaliser like
        # TRL's *_mask modes, since num_items_in_batch is counted before the loss.
        # Log ratio is pi_old / pi_rollout; with rollout logps as pi_old (verl bypass mode) it is
        # pi_theta / pi_rollout, verl compute_policy_loss_bypass_mode passing old_log_prob=log_prob.
        rs_log_ratio = None
        if rollout_rs is not None:
            if use_vllm and sampling_per_token_logps is not None and vllm_importance_sampling_correction:
                rs_log_ratio = old - sampling_per_token_logps
            elif kwargs.get("rollout_logps_as_old", False) and old is not None:
                rs_log_ratio = new.detach() - old
        if rs_log_ratio is not None:
            rs_log_ratio = torch.clamp(torch.nan_to_num(rs_log_ratio * mask, nan = 0.0), min = -20.0, max = 20.0)
            rs_mask = mask.to(rs_log_ratio.dtype)
            rs_count = rs_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            rs_keep = torch.ones_like(rs_log_ratio)
            for rs_option, rs_lower, rs_upper in rollout_rs:
                if rs_option.endswith("k1"):
                    rs_stat = -rs_log_ratio
                elif rs_option.endswith("k2"):
                    rs_stat = 0.5 * rs_log_ratio * rs_log_ratio
                else:
                    rs_stat = torch.expm1(rs_log_ratio) - rs_log_ratio
                rs_stat = rs_stat * rs_mask
                if rs_option.startswith("seq_sum"):
                    rs_stat = rs_stat.sum(dim=-1, keepdim=True)
                elif rs_option.startswith("seq_mean"):
                    rs_stat = rs_stat.sum(dim=-1, keepdim=True) / rs_count
                elif rs_option.startswith("seq_max"):
                    # k2, k3 >= 0 and masked entries are 0, so this is the masked max (0 on an empty row).
                    rs_stat = rs_stat.amax(dim=-1, keepdim=True)
                if rs_lower is not None:
                    rs_pass = (rs_stat >= rs_lower) & (rs_stat <= rs_upper)
                else:
                    rs_pass = rs_stat <= rs_upper
                rs_keep = rs_keep * rs_pass.to(rs_keep.dtype)
            rs_keep = rs_keep > 0
            if importance_sampling_ratio is not None:
                # Select, not multiply: an unclipped ratio can overflow to inf and inf * 0 is nan.
                importance_sampling_ratio = torch.where(rs_keep, importance_sampling_ratio, 0.0)
    pass

    # Must detach when old is None: exp(new - new.detach()) == 1 but keeps grads correct.
    if old is not None:
        log_ratio = new - old
    else:
        log_ratio = new - new.detach()
    if kwargs.get("rollout_logps_as_old", False):
        # Rollout gap can pass 88 nats (float32 exp overflow, 0 * inf backward); verl
        # compute_policy_loss_vanilla clamps the log ratio to [-20, 20] too.
        log_ratio = torch.clamp(log_ratio, min = -20.0, max = 20.0)

    if importance_sampling_level == "token":
        log_importance_weights = log_ratio
    elif importance_sampling_level == "sequence":
        log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        log_importance_weights = log_importance_weights.unsqueeze(-1)
    else:
        raise ValueError(
            f"Unknown importance sampling level: {importance_sampling_level}. Possible values are 'token' "
            "and 'sequence'."
        )

    coef_1 =  torch.exp(log_importance_weights)

    # Reverse KL: low-variance low-bias estimator as used in the GRPO paper.
    if beta != 0.0:
        # Keep expm1 finite, then cap the final KL term (TRL does not clamp).
        kl_log_ratio = torch.clamp(ref - new, min = -20.0, max = 20.0)
        kl_i = torch.expm1(kl_log_ratio) - kl_log_ratio
        # TRL order: pre-clamp non-detached coef_1, before the loss_type dispatch.
        if use_bias_correction_kl:
            kl_i = kl_i * coef_1
        # Clamp after correction so a saturated KL also stops the ratio's gradient.
        kl_i = torch.clamp(kl_i, min = -10.0, max = 10.0)
    else:
        # Zeros with the correct shape.
        if importance_sampling_level == "sequence":
            kl_i = new.new_zeros(new.size(0), 1)
        else:
            kl_i = torch.zeros_like(new)

    if loss_type == "cispo":
        clamped_ratios = torch.clamp(coef_1, max=epsilon_high).detach()
        loss_i = -clamped_ratios * advantages * new
    elif loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
        coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)

        if delta is not None:
            loss_1 = torch.clamp(coef_1, max=delta) * advantages
        else:
            loss_1 = coef_1 * advantages
        pass
        loss_2 = coef_2 * advantages
        loss_i = -torch.min(loss_1, loss_2)
    elif loss_type == "sapo":
        temperatures = torch.where(advantages > 0, sapo_temperature_pos, sapo_temperature_neg)
        soft_coef_1 = torch.sigmoid(temperatures * (coef_1 - 1)) * 4 / temperatures
        loss_i = -soft_coef_1 * advantages
    elif loss_type == "vespo":
        if get_gamma_weights is None:
            raise Exception("vespo is only available in TRL 0.26.0+")
        phi_seq = get_gamma_weights(
            advantages=advantages,
            log_ratio_per_token=log_ratio,
            mask=mask,
            importance_sampling_ratio=importance_sampling_ratio,
            k_pos=vespo_k_pos,
            lambda_pos=vespo_lambda_pos,
            k_neg=vespo_k_neg,
            lambda_neg=vespo_lambda_neg,
        )
        loss_i = -phi_seq * advantages * new
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    if off_policy_mask_threshold is not None:
        loss_i = loss_i * off_policy_mask

    if entropy_mask is not None:
        loss_i = loss_i * entropy_mask

    if use_vllm and sampling_per_token_logps is not None and vllm_importance_sampling_correction:
        # vespo applies the IS ratio inside get_gamma_weights, so skip it here.
        if loss_type != "vespo":
            loss_i = loss_i * importance_sampling_ratio
        # delta for the metric.
        with torch.no_grad():
            delta = torch.abs(old - sampling_per_token_logps)
            delta = delta * mask
            # Zeroed rather than filtered like TRL, to keep the returned shape.
            delta = torch.nan_to_num(delta, nan = 0.0, posinf = math.inf)
            flat_is_ratio = importance_sampling_ratio * mask
    else:
        delta = torch.tensor([]).detach()
        flat_is_ratio = torch.tensor([]).detach()
    if beta != 0.0:
        loss_i = loss_i + beta * kl_i
    if rs_keep is not None:
        # Rejected tokens drop out of every loss term, KL included (verl rollout_rs). A select, so
        # an inf or nan there neither reaches the loss nor turns into a nan gradient.
        loss_i = torch.where(rs_keep, loss_i, 0.0)

    mask = mask.to(torch.float32)
    n_mask_per_reward = mask.sum(1)

    # https://github.com/huggingface/trl/blob/e8b8499f1f8d76838155b515e414ee98f757d6d5/trl/trainer/grpo_trainer.py#L1624
    if sample_weight is not None:
        # Same process / accumulation rescale as the dapo branch, whose weights are 1 / num_items_in_batch.
        normalizer = 1.0 / num_processes
        if steps_per_generation:
            normalizer = normalizer * current_gradient_accumulation_steps / steps_per_generation
        loss = (loss_i * mask * sample_weight.to(loss_i.dtype).unsqueeze(-1)).sum() / normalizer
    elif loss_type in ["grpo", "sapo"]:
        loss = ((loss_i * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
        loss = loss / current_gradient_accumulation_steps
    elif loss_type == "bnpo":
        # BNPO uses the local micro-batch token count even when TRL supplies a global count.
        loss = (loss_i * mask).sum() / mask.sum().clamp(min=1.0)
        loss = loss / current_gradient_accumulation_steps
    elif loss_type == "dr_grpo":
        loss = (loss_i * mask).sum() / (loss_i.size(0) * max_completion_length)
        loss = loss / current_gradient_accumulation_steps
    elif loss_type in ["cispo", "dapo", "vespo"]:
        # Floor at 1 like TRL: a fully masked batch (mask_truncated_completions) is 0/0 = nan otherwise.
        if torch.is_tensor(num_items_in_batch):
            normalizer = num_items_in_batch.clamp(min = 1.0) / num_processes
        else:
            normalizer = max(float(num_items_in_batch), 1.0) / num_processes
        # num_items_in_batch spans the whole generation batch; rescale to one accumulation window like TRL.
        if steps_per_generation:
            normalizer = normalizer * current_gradient_accumulation_steps / steps_per_generation
        loss = (loss_i * mask).sum() / normalizer
    elif loss_type == "luspo":
        # loss_i is (B, T) unless sequence level with beta 0, so mask elementwise (TRL >= 1.10).
        loss = (loss_i * mask).sum(-1).mean()
        normalizer = current_gradient_accumulation_steps
        loss = loss / normalizer
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Folded metrics.
    def masked_batch_mean(x):
        with torch.inference_mode():
            completion_length = n_mask_per_reward.mean()
            # Sequence level only if x is narrower than mask: one-token completions are (B, 1) too.
            if x.shape[1] == 1 and mask.shape[1] != 1:
                return completion_length, x.mean()
            else:
                # Rows with no tokens (mask_truncated_completions) are left out, as in TRL.
                mean_kl_per_reward = (x * mask).sum(1) / n_mask_per_reward.clamp(min = 1.0)
                kept_rows = (n_mask_per_reward > 0).sum()
                mean_kl = torch.where(
                    kept_rows == n_mask_per_reward.numel(),
                    mean_kl_per_reward.mean(),
                    mean_kl_per_reward.sum() / kept_rows.clamp(min = 1),
                )
                return completion_length, mean_kl
    # Rejected tokens add nothing to the reported KL either; the denominator keeps every completion
    # token, like the loss normaliser.
    kl_metric = kl_i if rs_keep is None or beta == 0.0 else torch.where(rs_keep, kl_i, 0.0)
    completion_length, mean_kl = masked_batch_mean(kl_metric)
    return loss, completion_length, mean_kl, delta, flat_is_ratio, coef_1, mask

def grpo_update_SamplingParams(
    SamplingParams,
    generation_kwargs,
    vllm_sampling_params = None,
):
    good_sampling_params_keys = inspect.signature(SamplingParams).parameters.keys()

    new_generation_kwargs = {}
    for key in generation_kwargs.keys():
        if key in good_sampling_params_keys:
            new_generation_kwargs[key] = generation_kwargs[key]
    generation_kwargs = new_generation_kwargs

    if vllm_sampling_params is not None:
        overwrites = getattr(vllm_sampling_params, "_set_kwargs", None)
        if overwrites is None:
            default_sampling_params = SamplingParams()
            overwrites = {}
            for key in good_sampling_params_keys:
                if key.startswith("_") or not hasattr(vllm_sampling_params, key):
                    continue
                overwrited_key = getattr(vllm_sampling_params, key)
                if overwrited_key != getattr(default_sampling_params, key, None):
                    overwrites[key] = overwrited_key
        for key, overwrited_key in overwrites.items():
            if key in good_sampling_params_keys and key not in (
                "seed",
                "n",
                "temperature",
                "max_tokens",
                "logprobs",
            ):
                generation_kwargs[key] = overwrited_key
    return generation_kwargs

def _get_inference_mode_context_manager(model: torch.nn.Module):
    """A torchao-quantized state dict hits "Cannot set version_counter for inference tensor" on ops like aten.t() under inference mode, a PyTorch bug affecting all tensor subclasses (pytorch/pytorch#164872), so use `torch.no_grad()` in that case and `torch.inference_mode()` otherwise."""
    torchao_config = getattr(model, "torchao_config", None)
    if torchao_config is not None and torchao_config.qat_scheme is None:
        return torch.no_grad()
    # FSDP keeps the parameters it all-gathers for one forward and reuses them in the next, so a
    # gather under inference mode poisons the training step: "Inference tensors cannot be saved
    # for backward" (unsloth#3551). TRL generates under no_grad for the same reason.
    import os, sys

    if os.environ.get("ACCELERATE_USE_FSDP", "").strip().lower() in ("1", "true", "yes", "on"):
        return torch.no_grad()
    if os.environ.get("FSDP_VERSION", "").strip() not in ("", "0"):
        return torch.no_grad()
    # Unsloth's native FSDP2 (UNSLOTH_FSDP2=1 under torchrun) exports neither variable, so accelerate
    # does not wrap the model again: read its launch flag, and the mark shard_model leaves on the root.
    _fsdp2 = os.environ.get("UNSLOTH_FSDP2", "").strip().lower()
    if _fsdp2 in ("1", "true", "yes", "on", "native"):
        try:
            if int(os.environ.get("WORLD_SIZE", "1")) > 1:
                return torch.no_grad()
        except ValueError:
            pass
    for _m in (model, getattr(model, "module", None), getattr(model, "_orig_mod", None)):
        if _m is not None and getattr(_m, "_unsloth_fsdp2_native", False) is True:
            return torch.no_grad()
    accelerate_state = sys.modules.get("accelerate.state", None)
    shared_state = getattr(
        getattr(accelerate_state, "AcceleratorState", None), "_shared_state", None
    )
    if isinstance(shared_state, dict) and "FSDP" in str(shared_state.get("distributed_type", "")):
        return torch.no_grad()
    # ZeRO-3 reuses gathered persistent params (same poisoning); a known stage below 3 does not.
    deepspeed = os.environ.get("ACCELERATE_USE_DEEPSPEED", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ) or (
        isinstance(shared_state, dict)
        and "DEEPSPEED" in str(shared_state.get("distributed_type", ""))
    )
    if deepspeed:
        # Inline, not a helper: rl_replacements copies only this function into the GRPO trainer.
        stages = []
        hf_deepspeed = sys.modules.get("transformers.integrations.deepspeed", None)
        try:
            if hf_deepspeed is not None and hf_deepspeed.is_deepspeed_zero3_enabled():
                stages.append(3)
        except Exception:
            pass
        plugins = shared_state.get("deepspeed_plugins") if isinstance(shared_state, dict) else None
        if plugins is not None and not isinstance(plugins, dict):
            plugins = {"default": plugins}
        for plugin in (plugins or {}).values():
            if not getattr(plugin, "selected", True):
                continue
            # A config file's stage lives in hf_ds_config; zero_stage defaults to 2 regardless.
            stage = getattr(getattr(plugin, "hf_ds_config", None), "_stage", None)
            if not isinstance(stage, int) or stage < 0:
                stage = getattr(plugin, "zero_stage", None)
            stages.append(stage)
        env_stage = os.environ.get("ACCELERATE_DEEPSPEED_ZERO_STAGE", "").strip()
        if env_stage.isdigit():
            stages.append(int(env_stage))
        stages = [stage for stage in stages if isinstance(stage, int) and stage >= 0]
        if not stages or 3 in stages:
            return torch.no_grad()
    try:
        first = next(iter(model.parameters()), None)
    except Exception:
        first = None
    if first is not None and hasattr(first, "ds_id"):
        return torch.no_grad()
    return torch.inference_mode()

import os as _unsloth_os
UNSLOTH_ENABLE_LOGGING = _unsloth_os.environ.get('UNSLOTH_ENABLE_LOGGING', '0') in ('1', 'True', 'true')

UNSLOTH_GRPO_SEQ_PACKING_ON = _unsloth_os.environ.get('UNSLOTH_GRPO_SEQ_PACKING', '1').lower() not in ('0', 'false', 'no', 'off')

try:
    import inspect as _unsloth_inspect
    from unsloth_zoo.rl_replacements import RL_REPLACEMENTS as _unsloth_zoo_RL
    UNSLOTH_ZOO_HAS_MASKED_COL_GUARD = 'torch.where(_keep, new' in _unsloth_inspect.getsource(_unsloth_zoo_RL['grpo_compute_loss'])
except Exception:
    UNSLOTH_ZOO_HAS_MASKED_COL_GUARD = False

try:
    import inspect as _unsloth_inspect
    from unsloth_zoo.rl_replacements import chunked_hidden_states_selective_log_softmax as _unsloth_chunked_lsm
    UNSLOTH_ZOO_HAS_HEAD_BIAS = 'lm_head_bias' in _unsloth_inspect.signature(_unsloth_chunked_lsm).parameters
except Exception:
    UNSLOTH_ZOO_HAS_HEAD_BIAS = False

_pg_build_layout = _pg_enabled_fn = _pg_verify_on = _pg_tol_ok = _PG_TOL_KILL = None
UNSLOTH_GRPO_PREFIX_GROUPER_ON = _unsloth_os.environ.get('UNSLOTH_GRPO_PREFIX_GROUPER', '1').lower() not in ('0', 'false', 'no', 'off')
if UNSLOTH_GRPO_PREFIX_GROUPER_ON:
    try:
        from unsloth.utils.prefix_grouper import build_group_layout as _pg_build_layout, prefix_grouper_enabled as _pg_enabled_fn, verify_on as _pg_verify_on, tol_ok as _pg_tol_ok, TOL_KILL as _PG_TOL_KILL
    except Exception:
        UNSLOTH_GRPO_PREFIX_GROUPER_ON = False

try:
    from unsloth_zoo.device_map_planner import detect_logit_transforms
except Exception:
    detect_logit_transforms = None
@dataclass
class UnslothGRPOConfig(GRPOConfig):
    """
    
    Configuration class for the [`GRPOTrainer`].

    This class includes only the parameters that are specific to GRPO training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model and reference model

        model_init_kwargs (`str` or `dict[str, Any]`, *optional*):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`GRPOTrainer`] is provided as a string. The `revision` value is also used when loading
            processing classes.
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to
            [`~transformers.AutoModelForCausalLM.from_pretrained`] and [`~transformers.AutoProcessor.from_pretrained`].
            Also applied to reward-model and reward-tokenizer loads.
        router_aux_loss_coef (`float`, *optional*):
            Coefficient of the load-balancing auxiliary loss for Mixture-of-Experts (MoE) models, added to the training
            loss with this weight. When not set, the value declared by the model config is used. Set to `0.0` to
            disable it. Fails when used with a non-MoE model.
        disable_dropout (`bool`, *optional*, defaults to `False`):
            Whether to disable dropout in the model. This is useful for training with a reference model, as it prevents
            the model from generating different logprobs for the same input.
        cast_lm_head_to_fp32 (`bool`, *optional*, defaults to `False`):
            Whether to cast the language modeling head of the policy and reference models to float32. As recommended by
            the [ScaleRL](https://huggingface.co/papers/2510.13786) recipe. This flag is only supported when the model
            has untied word embedding and language modeling head layers i.e. `tie_word_embeddings` in the model config
            is False.

        > Parameters that control the data preprocessing

        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the column `"prompt"` in the dataset. If you use a custom reward function that
            requires any column other than `"prompts"` and `"completions"`, you should keep this to `False`.
        num_generations (`int`, *optional*, defaults to `8`):
            Number of generations per prompt to sample. The effective batch size (num_processes * per_device_batch_size
            * gradient_accumulation_steps) must be evenly divisible by this value.
        num_generations_eval (`int`, *optional*):
            Number of generations to sample during evaluation. This allows using fewer generations during evaluation to
            save computation. If `None`, uses the value of `num_generations`.
        max_completion_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum length of the generated completion.
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

        generation_batch_size (`int`, *optional*):
            Batch size to use for generation. If `None`, it defaults to the effective training batch size:
            `per_device_train_batch_size * num_processes * steps_per_generation`. In other words, there is one
            generation batch processed per optimization step. Mutually exclusive with `steps_per_generation`.
        steps_per_generation (`int`, *optional*):
            Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`. Mutually exclusive
            with `generation_batch_size`.
        temperature (`float`, *optional*, defaults to `1.0`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        top_p (`float`, *optional*, defaults to `1.0`):
            Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to
            `1.0` to consider all tokens.
        top_k (`int`, *optional*, defaults to `0`):
            Number of highest probability vocabulary tokens to keep for top-k-filtering. If `0`, top-k-filtering is
            disabled and all tokens are considered.
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
            Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for generation
            instead of the default model.generate(). Requires `vllm` to be installed.
        vllm_mode (`str`, *optional*, defaults to `"colocate"`):
            Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `"server"` or
            `"colocate"`.

            - `"server"`: The trainer will send generation requests to a separate vLLM server. Make sure a vLLM server
              is running (start with `vllm serve`).
            - `"colocate"`: vLLM will run in the same process and share the training GPUs. This avoids the need for a
              separate server but may cause resource contention with training.
        vllm_model_impl (`str`, *optional*, defaults to `"vllm"`):
            Model implementation to use for vLLM. Must be one of `"transformers"` or `"vllm"`. `"transformers"`: Use
            the `transformers` backend for model implementation. `"vllm"`: Use the `vllm` library for model
            implementation.
        vllm_enable_sleep_mode (`bool`, *optional*, defaults to `False`):
            Enable vLLM sleep mode to offload weights/cache during the optimizer step. Keeps GPU memory usage low, but
            waking the engine adds host–device transfer latency.
        vllm_structured_outputs_regex (`str`, *optional*):
            Regex for vLLM structured outputs. If `None` (default), structured outputs is disabled.

        > Parameters that control the vLLM server (only used when `vllm_mode` is `"server"`)

        vllm_server_base_url (`str`, *optional*):
            Base URL for the vLLM server (e.g., `"http://localhost:8000"`). If provided, `vllm_server_host` and
            `vllm_server_port` are ignored.
        vllm_server_host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host of the vLLM server to connect to. Ignored if `vllm_server_base_url` is provided.
        vllm_server_port (`int`, *optional*, defaults to `8000`):
            Port of the vLLM server to connect to. Ignored if `vllm_server_base_url` is provided.
        vllm_server_timeout (`float`, *optional*, defaults to `240.0`):
            Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up after the
            timeout, a `ConnectionError` is raised.
        vllm_group_port (`int`, *optional*, defaults to `51216`):
            Port number for the weight update group. This is used to communicate with the vLLM server. Unless the port
            is occupied, there is no need to change it.

        > Parameters that control colocated vLLM execution (only used when `vllm_mode` is `"colocate"`)

        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.3`):
            Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set to
            `"colocate"`. If you are using `vllm_mode="server"`, this parameter must be passed separately when
            launching the vLLM server via the `--vllm_gpu_memory_utilization` flag.
        vllm_max_model_length (`int`, *optional*):
            Context window for vLLM. Set it to at least the maximum prompt length in the dataset plus
            `max_completion_length`; if omitted, it is inferred from the model config.
        vllm_tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set to
            `"colocate"`. If you are using `vllm_mode="server"`, this parameter must be passed separately when
            launching the vLLM server via the `--vllm_tensor_parallel_size` flag.

        > Parameters that control the training

        beta (`float`, *optional*, defaults to `0.0`):
            KL coefficient. If `0.0` (default), the reference model is not loaded, reducing memory usage and improving
            training speed. [DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement
            learning](https://huggingface.co/papers/2501.12948) use a value of `0.001`.
        num_iterations (`int`, *optional*, defaults to `1`):
            Number of iterations per batch (denoted as μ in the algorithm).
        epsilon (`float`, *optional*, defaults to `0.2`):
            Epsilon value for clipping.
        delta (`float`, *optional*):
            Enables the upper clipping bound in two-sided GRPO loss when set to a float. If `None` (default), standard
            GRPO clipping is used. Recommended to be greater than `1 + ε` when enabled. This method is introduced in
            the [INTELLECT-2 tech report](https://huggingface.co/papers/2505.07291).
        epsilon_high (`float`, *optional*):
            Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the lower-bound
            specified in argument `epsilon`. Paper [DAPO](https://huggingface.co/papers/2503.14476) recommends `0.28`.
            When used with `loss_type='cispo'`, this corresponds to the ε_max param specified in the [ScaleRL
            paper](https://huggingface.co/papers/2510.13786) and the recommended value is `5.0`.
        sapo_temperature_neg (`float`, *optional*, defaults to `1.05`):
            Temperature for tokens with non-positive advantage scores used in the `sapo` loss function. This parameter
            is introduced in the [Soft Adaptive Policy Optimization paper](https://huggingface.co/papers/2511.20347).
        sapo_temperature_pos (`float`, *optional*, defaults to `1.0`):
            Temperature for tokens with positive advantage scores used in the `sapo` loss function. This parameter is
            introduced in the [Soft Adaptive Policy Optimization paper](https://huggingface.co/papers/2511.20347).
        vespo_k_pos (`float`, *optional*, defaults to `2.0`):
            k parameter for positive advantages, it is the power exponent in the VESPO loss. Controls how aggressively
            we down-weight samples with low importance weights (when the importance sampling ratio < 1).
        vespo_lambda_pos (`float`, *optional*, defaults to `3.0`):
            lambda parameter for positive advantages, it is the decay factor in the VESPO loss. Controls how
            aggressively we down-weight samples with high importance weights (when the importance sampling ratio > 1).
        vespo_k_neg (`float`, *optional*, defaults to `3.0`):
            k parameter for negative advantages, it is the power exponent in the VESPO loss. Controls how aggressively
            we down-weight samples with low importance weights (when the importance sampling ratio < 1).
        vespo_lambda_neg (`float`, *optional*, defaults to `2.0`):
            lambda parameter for negative advantages, it is the exponential decay factor in the VESPO loss. Controls
            how aggressively we down-weight samples with high importance weights (when the importance sampling ratio >
            1).
        importance_sampling_level (`str`, *optional*, defaults to `"token"`):
            Controls whether importance sampling ratios are computed at the `"token"` or `"sequence"` level. `"token"`
            keeps the raw per-token log-probability ratios (one weight per token). `"sequence"` averages the
            log-probability ratios across valid tokens to produce a single ratio per sequence. The [GSPO
            paper](https://huggingface.co/papers/2507.18071) shows that sequence-level sampling often yields more
            stable training and better alignment with sequence-level rewards.
        reward_weights (`list[float]`, *optional*):
            Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are
            weighted equally with weight `1.0`.
        multi_objective_aggregation (`str`, *optional*, defaults to `"sum_then_normalize"`):
            Method to aggregate multiple reward functions. Supported values are:

            - `"sum_then_normalize"` (default): First sums the weighted rewards from each reward function, then applies
              reward scaling/normalization as specified by `scale_rewards` (see `scale_rewards` for details).
            - `"normalize_then_sum"`: First normalizes/scales each reward function across generations (within each
              group), then sums the normalized rewards using the specified weights. The aggregated reward is then
              normalized at the batch level when forming advantages. This is the suggested approach from the paper
              [GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL
              Optimization](https://huggingface.co/papers/2601.05242).
        scale_rewards (`str` or `bool`, *optional*, defaults to `"group"`):
            Specifies the scaling strategy for rewards. Supported values are:

            - `True` or `"group"` (default): rewards are scaled by the standard deviation within each group, ensuring
              unit variance within a group.
            - `"batch"`: rewards are scaled by the standard deviation across the entire batch, as recommended in the
              [PPO Lite paper](https://huggingface.co/papers/2508.08221).
            - `False` or `"none"`: no scaling is applied. The [Dr. GRPO
              paper](https://huggingface.co/papers/2503.20783) recommends not scaling rewards, as scaling by the
              standard deviation introduces a question-level difficulty bias.
        loss_type (`str`, *optional*, defaults to `"dapo"`):
            Specifies the loss formulation to use. Supported values are:

            - `"grpo"`: Aggregates token-level losses by normalizing over sequence length. Not recommended due to
              length bias—this approach tends to prefer shorter completions with positive advantages and longer ones
              with negative advantages.
            - `"dr_grpo"`: Aggregates token-level losses by normalizing with a global constant. This method was
              introduced in the [Dr. GRPO paper](https://huggingface.co/papers/2503.20783) to eliminate length bias.
              The value of the constant corresponds to `max_completion_length`.
            - `"dapo"` (default): Aggregates token-level losses by normalizing with the number of active token in the
              global accumulated batch. This method was introduced in the [DAPO
              paper](https://huggingface.co/papers/2503.14476) to eliminate length bias.
            - `"bnpo"`: Aggregates token-level losses by normalizing with the number of active token in the local
              batch. Note that normalization is performed over the local batch only, so results may slightly vary
              depending on the local batch size, despite a constant effective batch size. When using
              `per_device_train_batch_size==1`, the loss is equivalent to the GRPO loss.
            - `"cispo"`: Clips the importance sampling weights instead of the advantage scaled importance weights. The
              clipped weights are then multiplied with the advantages and policy model's log probs. Individual token
              losses are aggregated by normalizing with the number of active tokens in the global accumulated batch.
              This method was introduced in the [MiniMax-M1 paper](https://huggingface.co/papers/2506.13585).
            - `"sapo"`: Soft Adaptive Policy Optimization loss, as introduced in the [Soft Adaptive Policy Optimization
              paper](https://huggingface.co/papers/2511.20347). Replaces hard clipping with a smooth,
              temperature-controlled gate that adaptively attenuates off-policy updates while preserving useful
              learning signals.
            - `"luspo"`: Length-Unbiased Sequence Policy Optimization loss. A sequence-level loss that scales each
              sequence's loss by its length. This is a modification of GSPO and requires
              `importance_sampling_level="sequence"`. Introduced in the [LUSPO
              paper](https://huggingface.co/papers/2602.05261).
            - `"vespo"`: Variational Sequence-Level Soft Policy Optimization. Replaces hard clipping with a smooth,
              asymmetric Gamma weighting function applied directly to sequence-level importance weights. Introduced in
              the [VESPO paper](https://huggingface.co/papers/2602.10693).
        mask_truncated_completions (`bool`, *optional*, defaults to `False`):
            When enabled, truncated completions are excluded from the loss calculation, preventing them from being
            incorrectly penalized and introducing noise during training. According to the
            [DAPO](https://huggingface.co/papers/2503.14476) paper, this is a good practice for training stability.
        sync_ref_model (`bool`, *optional*, defaults to `False`):
            Whether to synchronize the reference model with the active model every `ref_model_sync_steps` steps, using
            the `ref_model_mixup_alpha` parameter. This synchronization originates from the
            [TR-DPO](https://huggingface.co/papers/2404.09656) paper.
        ref_model_mixup_alpha (`float`, *optional*, defaults to `0.6`):
            α parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which controls the mix
            between the current policy and the previous reference policy during updates. The reference policy is
            updated according to the equation: `π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you
            must set `sync_ref_model=True`.
        ref_model_sync_steps (`int`, *optional*, defaults to `512`):
            τ parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which determines how
            frequently the current policy is synchronized with the reference policy. To use this parameter, you must
            set `sync_ref_model=True`.
        top_entropy_quantile (`float`, *optional*, defaults to `1.0`):
            ρ parameter from [Beyond the 80/20 Rule](https://huggingface.co/papers/2506.01939). Keeps in the policy
            loss term only the top-ρ quantile of tokens by entropy of the probability distribution at each sequence
            position, improving results. Range: `[0.0-1.0]`. A value of `0.0` masks all but the highest entropy token;
            `1.0` keeps all tokens. The paper recommends a value of `0.2`. If used with
            `mask_truncated_completions=True`, only tokens from non-truncated completions are considered.
        entropy_coef (`float`, *optional*, defaults to `0.0`):
            Coefficient of the entropy regularization term in the loss. A positive value adds an entropy bonus that
            encourages exploration by keeping the policy from collapsing to near-deterministic outputs. The bonus is
            always the mean per-token entropy regardless of `loss_type`; it is not rescaled to match a loss type's
            policy normalization, so `entropy_coef` has the same meaning for every loss type. When
            `use_adaptive_entropy=True`, this serves as the initial coefficient and is updated each optimizer step. Has
            no effect when set to `0.0` (default).
        use_adaptive_entropy (`bool`, *optional*, defaults to `False`):
            Whether to use adaptive entropy control, introduced in
            [Skywork-OR1](https://huggingface.co/papers/2505.22312). When enabled, the entropy coefficient
            `entropy_coef` is updated each optimizer step: incremented by `entropy_coef_delta` when the current entropy
            is below `entropy_target`, and decremented otherwise. The coefficient is only applied when entropy is at or
            below `entropy_target`.
        entropy_coef_min (`float`, *optional*, defaults to `0.0`):
            Lower bound for the entropy coefficient when using adaptive entropy control.
        entropy_coef_max (`float`, *optional*, defaults to `1.0`):
            Upper bound for the entropy coefficient when using adaptive entropy control.
        entropy_coef_delta (`float`, *optional*, defaults to `0.005`):
            Step size for adjusting the entropy coefficient at each optimizer step during adaptive entropy control.
        entropy_target (`float`, *optional*, defaults to `0.2`):
            Target mean per-token entropy (in nats) used by adaptive entropy control. The coefficient is only applied
            when the current entropy falls at or below this value. Measured over the same token set as the policy loss:
            all completion tokens by default, or only the high-entropy subset when `top_entropy_quantile < 1.0`.
            Typical language models have per-token entropies in the range 2–10 nats, so the default of `0.2` almost
            never triggers regularization (only on near-complete entropy collapse); set it close to the entropy you
            observe early in training (logged as the `entropy` metric) so the bonus engages before the policy collapses
            (and account for the token subset when using `top_entropy_quantile`).
        max_tool_calling_iterations (`int`, *optional*):
            Maximum number of tool-calling turns when training an agent. If `None`, there is no limit and generation
            stops when the model generates a response turn with no tool calls or when the total response length reaches
            `max_model_length`.
        vllm_importance_sampling_correction (`bool`, *optional*, defaults to `True`):
            Whether to apply Importance Sampling (IS) to correct for the mismatch between vLLM completion logprobs and
            recomputed training logprobs. If set to `False`, no IS is applied regardless of
            `vllm_importance_sampling_mode`. When `True`, the selected mode determines how the IS ratios are computed
            and constrained.
        vllm_importance_sampling_mode (`str`, *optional*, defaults to `"sequence_mask"`):
            Specifies how Importance Sampling is performed when `vllm_importance_sampling_correction=True`. Possible
            values are:

                - `"token_truncate"`: Token-level truncated IS (default). Per-token ratios are clipped to
                [C_min, C_max].
                - `"token_mask"`: Token-level masked IS. Per-token ratios outside [C_min, C_max] are set to zero.
                - `"sequence_truncate"`: Sequence-level truncated IS. A single sequence ratio is clipped to
                [C_min, C_max] and applied to all tokens in the sequence.
                - `"sequence_mask"`: Sequence-level masked IS. Sequences with ratios outside [C_min, C_max] are masked
                out.
        vllm_importance_sampling_clip_max (`float`, *optional*, defaults to `3.0`):
            Importance sampling upper bound C_max used by `vllm_importance_sampling_mode`. For `*_truncate` modes,
            importance ratios are clipped from above at C_max. For `*_mask` modes, ratios larger than C_max are set to
            zero.
        vllm_importance_sampling_clip_min (`float`, *optional*):
            Importance sampling lower bound C_min used by `vllm_importance_sampling_mode`. For `*_truncate` modes,
            ratios are clipped from below at C_min. For `*_mask` modes, ratios below C_min are set to zero. To strictly
            mask ratios below C_min without upper bound, set `vllm_importance_sampling_clip_max=None`.
        off_policy_mask_threshold (`float`, *optional*):
            Threshold for off-policy sequence masking. If `None`, off-policy sequence masking is disabled. When set,
            sequences with negative advantages and high KL divergence are masked out to stabilize training. This
            parameter corresponds to the `delta` threshold in Equation 9 of the [DeepSeek-V3.2
            paper](https://huggingface.co/papers/2512.02556). It expects a positive value (e.g., 0.5).
        use_bias_correction_kl (`bool`, *optional*, defaults to `True`):
            Whether to multiply the KL term by the importance sampling ratio, so that the KL gradient becomes the
            unbiased reverse-KL gradient, as described in the [DeepSeek-V3.2
            paper](https://huggingface.co/papers/2512.02556). This changes the KL gradient whenever `beta != 0`,
            including on-policy: the ratio is differentiable, so it affects the gradient even where its value is
            exactly 1. The unbiased reverse-KL property holds for `importance_sampling_level="token"`; with
            `"sequence"` a sequence-level weight is broadcast onto the per-token KL.

        > Parameters that control the logging

        log_completions (`bool`, *optional*, defaults to `False`):
            Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is installed,
            it prints the sample. If `wandb` and/or `trackio` logging is enabled, it logs it to `wandb` and/or
            `trackio`.
        log_multimodal (`bool`, *optional*, defaults to `True`):
            Whether to log multimodal content (images, videos, etc.) together with completions. Disable this to reduce
            log size when using high-resolution multimodal data.
        num_completions_to_print (`int`, *optional*):
            Number of completions to print with `rich`. If `None`, all completions are logged.
        log_unique_prompts (`bool`, *optional*, defaults to `False`):
            Whether to log unique prompts. If `True`, only unique prompts are logged. If `False`, all prompts are
            logged.
        log_completions_hub_repo (`str`, *optional*):
            Hugging Face Hub repository to save the completions. Should be a complete repository name like
            `'username/reponame'` or `'orgname/reponame'`, or just `'reponame'` in which case the repository will be
            created in the currently-logged-in Hugging Face user's namespace. Note that this repository will be public
            unless you set `hub_private_repo=True` or your organization's default is to create private repositories.

        > Parameters that control generation acceleration powered by transformers continuous batching

        use_transformers_continuous_batching (`bool`, *optional*, defaults to `False`):
            Whether to use transformers' continuous batching engine for generating completions. Requires
            `transformers>=5.8.0`.
        transformers_continuous_batching_config (`dict`, *optional*):
            Keyword arguments for [`~transformers.generation.ContinuousBatchingConfig`].

        > Deprecated parameters

        use_transformers_paged:

            <Deprecated version="1.2.0">

            Parameter `use_transformers_paged` is deprecated and will be removed in version v2.0.0. Use
            `use_transformers_continuous_batching` instead.

            </Deprecated>

        vllm_importance_sampling_cap:

            <Deprecated version="1.6.0">

            Parameter `vllm_importance_sampling_cap` is deprecated and will be removed in version v2.0.0. Use
            `vllm_importance_sampling_clip_max` instead.

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
        router_aux_loss_coef = 0.0,
        disable_dropout = False,
        cast_lm_head_to_fp32 = False,
        num_generations = 8,
        num_generations_eval = None,
        max_completion_length = 512,
        ds3_gather_for_generation = True,
        shuffle_dataset = True,
        pad_to_multiple_of = None,
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
        beta = None,
        num_iterations = 1,
        epsilon = 0.2,
        delta = None,
        epsilon_high = None,
        sapo_temperature_neg = 1.05,
        sapo_temperature_pos = 1.0,
        vespo_k_pos = 2.0,
        vespo_lambda_pos = 3.0,
        vespo_k_neg = 3.0,
        vespo_lambda_neg = 2.0,
        importance_sampling_level = 'token',
        reward_weights = None,
        multi_objective_aggregation = 'sum_then_normalize',
        scale_rewards = 'group',
        loss_type = None,
        mask_truncated_completions = None,
        sync_ref_model = False,
        ref_model_mixup_alpha = 0.6,
        ref_model_sync_steps = 512,
        top_entropy_quantile = 1.0,
        entropy_coef = 0.0,
        use_adaptive_entropy = False,
        entropy_coef_min = 0.0,
        entropy_coef_max = 1.0,
        entropy_coef_delta = 0.005,
        entropy_target = 0.2,
        max_tool_calling_iterations = None,
        vllm_importance_sampling_correction = False,
        vllm_importance_sampling_mode = 'sequence_mask',
        vllm_importance_sampling_clip_max = 3.0,
        vllm_importance_sampling_clip_min = None,
        off_policy_mask_threshold = None,
        use_bias_correction_kl = True,
        log_completions = False,
        log_multimodal = True,
        num_completions_to_print = None,
        log_unique_prompts = False,
        log_completions_hub_repo = None,
        use_transformers_continuous_batching = False,
        transformers_continuous_batching_config = None,
        use_transformers_paged = False,
        vllm_importance_sampling_cap = None,
        vllm_sampling_params = None,
        unsloth_num_chunks = -1,
        unsloth_logit_chunk_multiplier = None,
        unsloth_grpo_mini_batch = None,
        context_parallel_size = 1,context_parallel_backend = 'ulysses',
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
        
        _unsloth_default_loss_type = loss_type is None
        if _unsloth_default_loss_type:
            loss_type = 'dapo'
            if beta is None:
                print("Unsloth: GRPO now defaults to TRL's loss_type = 'dapo' with beta = 0.0. Set loss_type = 'bnpo', beta = 0.001 for Unsloth's previous default.")
        if beta is None:
            beta = 0.0 if str(loss_type).lower() in ('dapo', 'dr_grpo') else 0.001
        
        if loss_type.lower() == 'dr_grpo':
            loss_type = 'dr_grpo'
        elif loss_type.lower() == 'dapo':
            loss_type = 'dapo'
        if loss_type.lower() == 'dr_grpo':
            if scale_rewards == None:
                scale_rewards = True
            elif scale_rewards == True or scale_rewards == 'group':
                print('Unsloth: The Dr GRPO paper recommends setting `scale_rewards` to False! Will override. Set it to `None` to keep scaling.')
                scale_rewards = False
        elif loss_type.lower() == 'dapo' and not _unsloth_default_loss_type:
            if mask_truncated_completions is None:
                print('Unsloth: The DAPO paper recommends `mask_truncated_completions = True` - we will set it.')
                mask_truncated_completions = True
            if epsilon_high is None:
                print('Unsloth: The DAPO paper recommends `epsilon_high = 0.28` - we will set it.')
                epsilon_high = 0.28
            if beta != 0.0:
                print(f'[WARNING] Unsloth: The DAPO paper recommends setting `beta = 0.0` to remove the KL term - You have set it to {beta}.')
        elif loss_type.lower() == 'cispo':
            loss_type = 'cispo'
            if epsilon_high is None:
                print('Unsloth: CISPO caps the importance sampling weight at `epsilon_high`; the ScaleRL paper recommends `epsilon_high = 5.0` - we will set it.')
                epsilon_high = 5.0
            elif epsilon_high < 1.0:
                print(f'[WARNING] Unsloth: CISPO caps the importance sampling weight at `epsilon_high` itself, not 1 + epsilon_high, so {epsilon_high} caps every weight below 1. The ScaleRL paper uses 4 to 8.')
        if mask_truncated_completions is None:
            mask_truncated_completions = False
        
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
        
        if use_vllm and (top_k is None or top_k == 0): top_k = -1
        
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
            cast_lm_head_to_fp32 = cast_lm_head_to_fp32,
            num_generations = num_generations,
            num_generations_eval = num_generations_eval,
            max_completion_length = max_completion_length,
            ds3_gather_for_generation = ds3_gather_for_generation,
            shuffle_dataset = shuffle_dataset,
            pad_to_multiple_of = pad_to_multiple_of,
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
            num_iterations = num_iterations,
            epsilon = epsilon,
            delta = delta,
            epsilon_high = epsilon_high,
            sapo_temperature_neg = sapo_temperature_neg,
            sapo_temperature_pos = sapo_temperature_pos,
            vespo_k_pos = vespo_k_pos,
            vespo_lambda_pos = vespo_lambda_pos,
            vespo_k_neg = vespo_k_neg,
            vespo_lambda_neg = vespo_lambda_neg,
            importance_sampling_level = importance_sampling_level,
            reward_weights = reward_weights,
            multi_objective_aggregation = multi_objective_aggregation,
            scale_rewards = scale_rewards,
            loss_type = loss_type,
            mask_truncated_completions = mask_truncated_completions,
            sync_ref_model = sync_ref_model,
            ref_model_mixup_alpha = ref_model_mixup_alpha,
            ref_model_sync_steps = ref_model_sync_steps,
            top_entropy_quantile = top_entropy_quantile,
            entropy_coef = entropy_coef,
            use_adaptive_entropy = use_adaptive_entropy,
            entropy_coef_min = entropy_coef_min,
            entropy_coef_max = entropy_coef_max,
            entropy_coef_delta = entropy_coef_delta,
            entropy_target = entropy_target,
            max_tool_calling_iterations = max_tool_calling_iterations,
            vllm_importance_sampling_correction = vllm_importance_sampling_correction,
            vllm_importance_sampling_mode = vllm_importance_sampling_mode,
            vllm_importance_sampling_clip_max = vllm_importance_sampling_clip_max,
            vllm_importance_sampling_clip_min = vllm_importance_sampling_clip_min,
            off_policy_mask_threshold = off_policy_mask_threshold,
            use_bias_correction_kl = use_bias_correction_kl,
            log_completions = log_completions,
            log_multimodal = log_multimodal,
            num_completions_to_print = num_completions_to_print,
            log_unique_prompts = log_unique_prompts,
            log_completions_hub_repo = log_completions_hub_repo,
            use_transformers_continuous_batching = use_transformers_continuous_batching,
            transformers_continuous_batching_config = transformers_continuous_batching_config,
            use_transformers_paged = use_transformers_paged,
            vllm_importance_sampling_cap = vllm_importance_sampling_cap,**kwargs)
        super().__init__(**_unsloth_filter_config_init_kwargs(GRPOConfig, _unsloth_config_arguments, mirrored_from = __class__))
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

class _UnslothGRPOTrainer(_BaseTrainer):
    """"""

    loss_is_scaled_for_ga = True

    _tag_names = ["trl", "grpo"]
    _name = "GRPO"
    _paper = {
        "title": "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
        "id": "2402.03300",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{shao2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }"""),
    }

    def __init__(
        self,
        model: "str | PreTrainedModel | PeftModel",
        reward_funcs: RewardFunc | list[RewardFunc] | None = None,
        args: GRPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset
        | IterableDataset
        | DatasetDict
        | IterableDatasetDict
        | dict[str, Dataset | IterableDataset]
        | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        quantization_config: "BitsAndBytesConfig | None" = None,
        peft_config: "PeftConfig | None" = None,
        tools: list[Callable] | None = None,
        rollout_func: RolloutFunc | None = None,
        environment_factory: EnvironmentFactory | dict[str, EnvironmentFactory] | None = None,
    ):

        if hasattr(model, 'vllm_engine') and hasattr(args, 'use_vllm'):
            _unsloth_fsdp_cfg = getattr(args, 'fsdp_config', None)
            if getattr(args, 'fsdp', None) and isinstance(_unsloth_fsdp_cfg, dict) and str(_unsloth_fsdp_cfg.get('version', _unsloth_fsdp_cfg.get('fsdp_version', 2))) == '2':
                raise NotImplementedError('Unsloth: `fast_inference = True` (vLLM) is not supported with FSDP2.\nLoad the model with `fast_inference = False` to train with FSDP2, or launch without FSDP2 to use vLLM.')
            if (getattr(args, 'use_vllm', False) == False):
                args.use_vllm = True
            if getattr(args, 'top_k', -1) is None or getattr(args, 'top_k', -1) == 0:
                args.top_k = -1
            args.vllm_mode='colocate'
            _unsloth_esm = getattr(getattr(getattr(getattr(model.vllm_engine, 'llm_engine', None), 'vllm_config', None), 'model_config', None), 'enable_sleep_mode', None)
            if (_unsloth_esm if _unsloth_esm is not None else os.environ.get('UNSLOTH_VLLM_STANDBY', '0') != '0'):
                args.vllm_enable_sleep_mode=True
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

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
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
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
                get_config_model_id(model.config),
                revision=model_revision,
                truncation_side="left",
                padding_side="left",
                trust_remote_code=args.trust_remote_code,
            )

        if args.use_transformers_continuous_batching and isinstance(processing_class, ProcessorMixin):
            raise ValueError(
                "`use_transformers_continuous_batching` does not support multimodal models. Use `use_vllm` instead."
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
        # PEFT initialization logic removed via script for trl >= 1.4.0
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

        # Reward functions
        if reward_funcs is None:
            reward_funcs = []
        elif not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        reward_model_revisions = [None] * len(reward_funcs)
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                model_init_kwargs = args.model_init_kwargs or {}
                # Distributed training requires device_map=None ["auto" fails]
                if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                    model_init_kwargs["device_map"] = None
                model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
                reward_model_revisions[i] = model_init_kwargs.get("revision")
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(reward_funcs[i], nn.Module):  # Use Module over PretrainedModel for compat w/ compiled models
                self.reward_func_names.append(get_config_model_id(reward_funcs[i].config).split("/")[-1])
            else:
                self.reward_func_names.append(get_callable_name(reward_funcs[i]))
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        if len(reward_processing_classes) != len(reward_funcs):
            raise ValueError(
                f"The number of reward processing classes ({len(reward_processing_classes)}) must match the number of "
                f"reward functions ({len(reward_funcs)})."
            )

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs, strict=True)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        get_config_model_id(reward_func.config),
                        revision=reward_model_revisions[i],
                        trust_remote_code=args.trust_remote_code,
                    )
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.get_text_config().pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class

        self.reward_processing_classes = reward_processing_classes

        # Rollout function
        if rollout_func is not None and os.environ.get("TRL_EXPERIMENTAL_SILENCE", "0") != "1":
            warnings.warn(
                "You are using 'rollout_func', which is an experimental feature. This API may change or be removed at "
                "any time without prior notice. Silence this warning by setting environment variable "
                "TRL_EXPERIMENTAL_SILENCE=1.",
                UserWarning,
                stacklevel=2,
            )
        self.rollout_func = rollout_func
        if environment_factory is not None and os.environ.get("TRL_EXPERIMENTAL_SILENCE", "0") != "1":
            warnings.warn(
                "You are using 'environment_factory', which is an experimental feature. This API may change or be "
                "removed at any time without prior notice. Silence this warning by setting environment variable "
                "TRL_EXPERIMENTAL_SILENCE=1.",
                UserWarning,
                stacklevel=2,
            )

        # Tools
        if tools:
            if not Version(transformers.__version__) >= Version("5.0.0"):
                raise ImportError(
                    "Using tools with GRPOTrainer requires transformers version 5.0.0 or higher. Please upgrade "
                    "transformers with `pip install --upgrade transformers` to use this feature."
                )
        if environment_factory:
            if not Version(transformers.__version__) >= Version("5.2.0"):
                raise ImportError(
                    "Using `environment_factory` with GRPOTrainer requires transformers version 5.2.0 or higher. "
                    "Please install transformers from the main branch with `pip install "
                    "git+https://github.com/huggingface/transformers.git@main` to use this feature."
                )
        if tools or environment_factory:
            # jmespath is only needed by the legacy `response_schema` parser, which is all transformers < 5.13 ships.
            # The new-style `response_template` parser doesn't use it, so don't require it on newer versions.
            if not _SUPPORTS_RESPONSE_TEMPLATE and not is_jmespath_available():
                raise ImportError(
                    "Using tools with GRPOTrainer on transformers below 5.13.0 requires the jmespath library for "
                    "response parsing. Please install it with `pip install jmespath`, or upgrade transformers to "
                    "5.13.0 or higher, which doesn't need it."
                )
            if not supports_tool_calling(processing_class):
                raise ValueError(
                    "The provided chat template does not support tool calling. The template must be able to render a "
                    "full tool-calling conversation (user -> assistant with tool_calls -> tool)."
                )

        # Set up the environments and extract their methods to be used as tools.
        tools = tools or []
        self._standalone_tools = tools  # tools that are not bound to an environment

        # Normalize `environment_factory` to a `{name: factory}` mapping. A single callable is the special case of one
        # unnamed [`None`] environment shared by every example; a dict maps the `environment` field of each example to
        # its factory. `_multi_environment` records which case we are in: only then is the `environment` field read.
        self._multi_environment = isinstance(environment_factory, dict)
        if environment_factory is None:
            self.environment_factories = None
        elif self._multi_environment:
            self.environment_factories = environment_factory
        else:
            self.environment_factories = {None: environment_factory}

        if self.environment_factories is not None:
            # The environment an example uses is only known at batch time [it depends on the data]. Here we just probe
            # one instance of each environment to validate its `reset` method and extract its tool methods, used to
            # render the per-example tool schema in the prompt. Instances are pooled and reused [reset] across batches;
            # the probe seeds the pool so it is not wasted. The pool grows only when a batch needs more concurrent
            # instances of an environment than have been created so far, preserving the "construct once, reset often"
            # contract even when batches mix environments.
            self._env_tools = {}  # {environment name: tools exposed when this environment is selected}
            self._environment_pool = {}  # {environment name: list of reusable instances}
            # `self.tools` is the union of every environment's tools, accumulated below as each environment is probed.
            # Used where only the presence of a tool matters [chat template validation, async loop setup, tool metrics];
            # the per-example schema is rendered in `_tokenize_prompts`.
            self.tools = list(tools)
            env_reward_types = []  # env classes already given a reward column [dedup: same class under two names]
            for name, factory in self.environment_factories.items():
                instance = factory()
                has_reset = False
                has_reward = False
                methods = []
                # List on the class: getmembers on the instance evaluates properties
                for member_name, _ in inspect.getmembers(type(instance), predicate=inspect.isfunction):
                    if member_name == "reset":
                        has_reset = True
                    elif member_name == "get_reward":
                        has_reward = True
                    elif not member_name.startswith("_"):
                        methods.append(getattr(instance, member_name))
                if not has_reset:
                    raise ValueError(
                        "Each environment instance returned by `environment_factory` must define a callable `reset`."
                    )
                self._env_tools[name] = tools + methods
                self._environment_pool[name] = [instance]
                self.tools += [method for method in methods if method not in self.tools]

                # If this environment owns its reward via `get_reward`, expose it as an extra reward source [named after
                # the env class, weight 1]. One column per env class that defines `get_reward` [deduplicated, since a
                # dict factory may map several names to the same class]; a rollout is scored only when its environment is
                # exactly that class, so mixing an env that owns its reward with one that does not is safe [the latter's
                # rollouts return `None`, turned into NaN and ignored]. Exact-type match [not `isinstance`] keeps this
                # consistent with the async worker and avoids double-counting when one registered env subclasses another.
                # `get_reward` may be async [e.g. an LLM judge]; wrap it accordingly so `_calculate_rewards` runs it on
                # the daemon event loop like any other async reward func.
                if has_reward and type(instance) not in env_reward_types:
                    env_type = type(instance)
                    env_reward_types.append(env_type)
                    if inspect.iscoroutinefunction(instance.get_reward):

                        async def get_reward(environments, _env_type=env_type, **kwargs):
                            return [await e.get_reward() if type(e) is _env_type else None for e in environments]

                    else:

                        def get_reward(environments, _env_type=env_type, **kwargs):
                            return [e.get_reward() if type(e) is _env_type else None for e in environments]

                    self.reward_funcs.append(get_reward)
                    self.reward_func_names.append(env_type.__name__)
                    self.reward_processing_classes.append(None)
                    self.reward_weights = torch.cat([self.reward_weights, torch.ones(1)])
        else:
            self.tools = tools

        # At least one reward source is required: either `reward_funcs`, or an environment that owns the reward via a
        # `get_reward` method.
        if not self.reward_funcs:
            raise ValueError(
                "No reward source provided. Pass `reward_funcs`, or an `environment_factory` whose environment "
                "defines a `get_reward` method."
            )

        # The per-rollout environment instances and tool dicts both depend on the batch [which environment each example
        # selects], so they are built in `_generate_and_score_completions`, right before generation. Only
        # `self.environments` needs a default here, because `_calculate_rewards` reads it for every batch [including
        # batches with no environments]; the tool dicts are only read in the tool-calling loop, which runs after they
        # have been [re]built.
        self.environments = None

        # Check for async functions to start an event loop on a daemon thread
        self._has_async_funcs = any(is_async_callable(func) for func in self.reward_funcs + self.tools)

        if self._has_async_funcs:
            self.async_loop_thread, self.async_loop, self.async_loop_ready_event = start_event_loop_in_daemon(
                name="GRPOTrainer-AsyncLoop"
            )
            # wait until the event loop is running in the daemon thread
            self.async_loop_ready_event.wait()
            atexit.register(shutdown_event_loop_in_daemon, self.async_loop_thread, self.async_loop)

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

        # Training arguments
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.max_tool_calling_iterations = (
            args.max_tool_calling_iterations if args.max_tool_calling_iterations is not None else sys.maxsize
        )
        self.num_generations_eval = args.num_generations_eval or self.num_generations
        self.chat_template_kwargs = args.chat_template_kwargs or {}
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_transformers_continuous_batching = args.use_transformers_continuous_batching
        if self.use_transformers_continuous_batching:
            if not Version(transformers.__version__) >= Version("5.8.0"):
                raise ImportError(
                    "Using `use_transformers_continuous_batching` requires transformers>=5.8.0. "
                    "Please upgrade with `pip install --upgrade transformers`."
                )
            from transformers.generation import ContinuousBatchingConfig

            cb_kwargs = dict(args.transformers_continuous_batching_config or {})
            # The transformers default [0.9] leaves almost no VRAM for the training backward pass;
            # use a training-aware default unless the user has set it explicitly.
            cb_kwargs.setdefault("max_memory_percent", 0.5)
            self.continuous_batching_config = ContinuousBatchingConfig(**cb_kwargs)
        else:
            self.continuous_batching_config = None
        self.pad_to_multiple_of = args.pad_to_multiple_of
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.vllm_importance_sampling_correction = args.vllm_importance_sampling_correction
        self.vllm_importance_sampling_mode = args.vllm_importance_sampling_mode
        self.vllm_importance_sampling_clip_max = args.vllm_importance_sampling_clip_max
        self.vllm_importance_sampling_clip_min = args.vllm_importance_sampling_clip_min
        self.loss_type = args.loss_type
        self.multi_objective_aggregation = args.multi_objective_aggregation

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
        if self.aux_loss_enabled: raise NotImplementedError("Unsloth GRPO does not compute the MoE router auxiliary loss; set router_aux_loss_coef = 0 (the Unsloth default).")
        self.scale_rewards = args.scale_rewards
        self.importance_sampling_level = args.importance_sampling_level
        self.off_policy_mask_threshold = args.off_policy_mask_threshold
        if is_peft_model(model) and isinstance(model.get_output_embeddings(), BaseTunerLayer):
            # The log-probabilities are computed by multiplying the hidden states by `lm_head.weight` directly, so an
            # adapter on the LM head would be ignored and never trained.
            raise ValueError(
                "Applying a PEFT adapter to `lm_head` is not supported: the log-probabilities are computed from "
                "`lm_head.weight` directly, so the adapter would never be trained. Remove `'lm_head'` from your "
                "`target_modules`."
            )
        self.mask_truncated_completions = args.mask_truncated_completions
        self.top_entropy_quantile = args.top_entropy_quantile
        self.entropy_coef = args.entropy_coef
        self.use_adaptive_entropy = args.use_adaptive_entropy
        # Whether the entropy bonus is active. Constant for the run: entropy_coef only mutates in adaptive
        # mode, where this is True regardless of its value [so the bonus can recover after being decremented].
        self._entropy_bonus_enabled = self.entropy_coef != 0.0 or self.use_adaptive_entropy
        # Cached entropy from the last optimizer step; inf so the first accumulation window
        # applies no bonus until a real measurement arrives [conservative default].
        self._last_world_entropy = float("inf")
        # Running [entropy_sum, active_token_count] accumulated across the micro-batches of the current
        # accumulation window, so the adaptive controller sees the exact window-global entropy [not just the
        # last micro-batch]. Reset to None at each optimizer step.
        self._entropy_window_stats = None

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if train_dataset is None:
            # A dataset is optional when an environment owns the data and returns the prompt from `reset[]`; then
            # `max_steps` sets the length. Build a placeholder dataset of empty prompts to drive the loop — one
            # generation round's worth of rows, cycled across steps.
            if self.environment_factories is None:
                raise ValueError("`train_dataset` is required unless an `environment_factory` is provided.")
            if self._multi_environment:
                raise ValueError(
                    "A `dict` `environment_factory` (multiple environments) requires a `train_dataset` with an "
                    "`environment` column to route each example to its environment. Provide a dataset, or pass a "
                    "single environment factory."
                )
            if args.max_steps <= 0:
                raise ValueError(
                    "When training without a `train_dataset` (the environment owns the data and returns the prompt "
                    "from `reset()`), `max_steps` must be set to a positive value to define the training length. Set "
                    "it via `GRPOConfig(max_steps=...)`."
                )
            num_placeholder_rows = args.generation_batch_size // args.num_generations
            train_dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": ""}]] * num_placeholder_rows})
        elif not isinstance(train_dataset, (Dataset, IterableDataset, list, tuple)):
            raise TypeError(
                f"`train_dataset` must be a `Dataset` or `IterableDataset`, got `{type(train_dataset).__name__}`."
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
        # An iterable train set bakes prompt repeats into the stream, so it must be read by a single worker: multiple
        # workers would shard and interleave it, splitting the num_generations groups. Map-style train keeps its workers.
        if isinstance(train_dataset, IterableDataset) and args.dataloader_num_workers != 0:
            logger.warning(
                f"Iterable datasets require `dataloader_num_workers=0` to preserve prompt grouping; overriding the "
                f"provided value ({args.dataloader_num_workers})."
            )
            args.dataloader_num_workers = 0

        if args.loss_type == "luspo" and args.importance_sampling_level != "sequence":
            logger.warning(
                "When using `'luspo'` loss, `importance_sampling_level` should be set to `'sequence'` to mirror the "
                "paper's setup."
            )

        if args.loss_type == "vespo" and args.importance_sampling_level != "token":
            logger.warning(
                "VESPO computes sequence-level importance weights internally. `importance_sampling_level` should be "
                "set to `'token'` (the default)."
            )

        if args.importance_sampling_level == "sequence" and args.loss_type in ["bnpo", "dr_grpo", "dapo", "cispo"]:
            logger.warning(
                f"When using `importance_sampling_level='sequence'`, the `'{args.loss_type}'` loss sums per-token "
                "contributions, which effectively weights each sequence by its completion length instead of "
                "optimizing the per-sequence objective. To reproduce the GSPO paper's setup, set `loss_type='grpo'` "
                "(see https://huggingface.co/docs/trl/main/en/paper_index#group-sequence-policy-optimization]."
            )

        if self.loss_type == "vespo" and self.use_vllm and self.vllm_importance_sampling_correction:
            if self.vllm_importance_sampling_mode not in ["token_truncate", "token_mask"]:
                raise ValueError(
                    f"VESPO loss requires `vllm_importance_sampling_mode` to be either 'token_truncate' or "
                    f"'token_mask'. Got: {self.vllm_importance_sampling_mode}."
                )

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations [forward + backward passes], including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

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
            data_collator=identity,  # No data collation is needed in GRPO
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

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # For deepspeed, fsdp or non-distributed models, create a reference model from scratch
            model_init_kwargs = dict(args.model_init_kwargs or {})  # copy to avoid mutating model_init_kwargs
            if quantization_config is not None:
                model_init_kwargs["quantization_config"] = quantization_config
            # Distributed training requires device_map=None ["auto" fails]
            if self.args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
            self.ref_model = create_model_from_path(get_config_model_id(self.model.config), **model_init_kwargs)

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Cast LM Head To FP32
        if args.cast_lm_head_to_fp32:

            def _cast_lm_head_to_fp32(target_model: PreTrainedModel):
                """Cast lm_head to fp32 while preserving embedding output dtype if tied."""

                if is_peft_available() and isinstance(target_model.lm_head, BaseTunerLayer):
                    raise ValueError(
                        "`cast_lm_head_to_fp32=True` is not supported when the lm_head carries a PEFT "
                        "adapter (e.g. `target_modules` includes `lm_head`). Remove `lm_head` from the "
                        "adapter's target modules, or set `cast_lm_head_to_fp32=False`."
                    )

                original_dtype_local = target_model.lm_head.weight.dtype
                lm_head = target_model.lm_head.float()
                target_model.lm_head = lm_head

                def cast_forward_to_fp32(hidden_states):
                    return nn.functional.linear(
                        hidden_states.to(torch.float32),
                        lm_head.weight.to(torch.float32),
                        None if lm_head.bias is None else lm_head.bias.to(torch.float32),
                    )

                lm_head.forward = cast_forward_to_fp32

                if target_model.config.tie_word_embeddings:

                    def cast_outputs_to_original_dtype(module, args, output):
                        return output.to(original_dtype_local)

                    # Only cast activations; weights are now fp32 [intentional for numerical stability of logits]
                    target_model.model.embed_tokens.register_forward_hook(cast_outputs_to_original_dtype)

            _cast_lm_head_to_fp32(model)
            if self.ref_model is not None:
                _cast_lm_head_to_fp32(self.ref_model)

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
        # `entropy` feeds the `entropy` metric, the high-entropy token mask and the entropy bonus
        add_fused_lm_head(
            self.model.get_base_model() if is_peft_model(self.model) else self.model,
            temperature=self.temperature,
            cast_lm_head_to_fp32=args.cast_lm_head_to_fp32,
            outputs=("log_probs", "entropy"),
        )
        if self.ref_model is not None:
            add_fused_lm_head(
                self.ref_model, temperature=self.temperature, cast_lm_head_to_fp32=args.cast_lm_head_to_fp32
            )

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self._current_train_step_time = 0.0
        self.log_completions = args.log_completions
        self.log_multimodal = args.log_multimodal
        self.log_unique_prompts = args.log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # Keep logs sized to the generation batch to record only outputs from the latest model update.
        self._logs = {
            "images": deque(maxlen=args.generation_batch_size),
            "prompt": deque(maxlen=args.generation_batch_size),
            "completion": deque(maxlen=args.generation_batch_size),
            "rewards": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
            "advantages": deque(maxlen=args.generation_batch_size),
            "extra": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
        }
        # Buffers for user-logged data from reward functions, flushed after gathering
        self._pending_extra_logs = defaultdict(list)
        self._pending_metrics = defaultdict(list)

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            self.vllm_generation = VLLMGeneration(
                model=self.model,
                accelerator=self.accelerator,
                processing_class=self.processing_class,
                mode=args.vllm_mode,
                structured_outputs_regex=args.vllm_structured_outputs_regex,
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
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                max_completion_length=self.max_completion_length,
                logprobs=0,
                generation_kwargs=args.generation_kwargs,
            )
            self._last_loaded_step = -1
        else:
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

        self._dist = DistributedBackend(self.accelerator)

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            if self.beta == 0.0:
                raise ValueError(
                    "You passed `sync_ref_model=True` while `beta=0.0`, which means the reference model is not used "
                    "during training. Consequently, GRPOTrainer does not create a `ref_model` instance, and there is "
                    "nothing to synchronize. Please set `sync_ref_model=False`, or set `beta` to a non-zero value."
                )
            if is_peft_model(model):
                raise NotImplementedError(
                    "You passed `sync_ref_model=True` while using a PEFT model, which is currently not supported. "
                    "With PEFT, GRPOTrainer does not keep a separate reference model in memory; instead, it recovers "
                    "reference behavior by temporarily disabling the adapter. As a result, there is no standalone "
                    "`ref_model` instance to synchronize. Use `sync_ref_model=False`, or opt for full fine-tuning if "
                    "you need a synced reference model. If you need `sync_ref_model` to work with PEFT, please open a "
                    "feature request at https://github.com/huggingface/trl/issues."
                )
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    # set device placement to True to make `prepare_model` move `reward_func` to device when using fsdp
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True
                    )

        if self.accelerator.is_main_process and self.log_completions:
            os.makedirs(os.path.join(self.args.output_dir, "completions"), exist_ok=True)
            if self.args.log_completions_hub_repo is not None:
                repo_id = self.args.log_completions_hub_repo
                create_repo(repo_id, private=self.args.hub_private_repo, repo_type="dataset", exist_ok=True)
                template_path = pkg_resources.files("trl").joinpath("templates/completions_dataset_card.md")
                card_data = DatasetCardData(
                    pretty_name="TRL Completion logs",
                    tags=["trl", "trl-logs", "completions"],
                )
                card = DatasetCard.from_template(
                    card_data=card_data,
                    template_path=str(template_path),
                    repo_id=repo_id,
                    hub_model_id=self.args.hub_model_id,
                )
                card.push_to_hub(repo_id)
                self.commit_scheduler = CommitScheduler(
                    repo_id=repo_id,
                    repo_type="dataset",
                    folder_path=f"{self.args.output_dir}/completions",
                    every=2,  # minutes
                    allow_patterns=["*.parquet"],
                )

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        # and "attention_mask"). In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't
        # work. Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "image", "images"]

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × steps_per_generation`). This allows us to generate completions
    # once every steps_per_generation step—rather than once per accumulation step—which is significantly more
    # efficient. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the splitting internally.
    # Maintenance note: this method is a copy-paste of the original `Trainer.get_train_dataloader` with two changes: the
    # batch size is multiplied by `steps_per_generation`, and iterable datasets are wrapped (see `repeat_iterable_dataset`).
    def get_train_dataloader(self):
        dataset = self.train_dataset
        if isinstance(dataset, IterableDataset):
            # Iterable datasets can't be indexed, so RepeatSampler can't be attached. Reproduce its ordering by
            # transforming the stream instead (see `repeat_iterable_dataset`). The full permutation done by
            # RepeatSampler becomes a buffered shuffle here.
            if self.shuffle_dataset:
                dataset = dataset.shuffle(seed=self.args.seed)
            dataset = repeat_iterable_dataset(
                dataset,
                mini_repeat_count=self.num_generations,
                batch_size=self.args.generation_batch_size // self.num_generations,
                repeat_count=self.num_iterations * self.args.steps_per_generation,
            )
        return self._get_dataloader(
            dataset=dataset,
            description="Training",
            batch_size=self._train_batch_size * self.args.steps_per_generation,  # < this is the change
            sampler_fn=self._get_train_sampler,
            is_training=True,
        )

    def _get_train_sampler(self, dataset: Dataset | None = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. Each row shows the per-step batch
        # returned by `_prepare_inputs`; rows within a `steps_per_generation` block are slices of the same
        # generated batch. When `num_iterations > 1`, that block is reused for multiple optimization passes
        # before regenerating.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
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
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations_eval,
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
            eval_dataset = repeat_iterable_dataset(eval_dataset, mini_repeat_count=self.num_generations_eval)
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

    def get_high_entropy_mask(self, entropies: torch.Tensor, mask: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Returns a binary mask identifying tokens whose entropy exceeds a given quantile threshold.

        Args:
            entropies (`torch.Tensor`):
                Tensor of shape (batch_size, seq_len) with per-token entropy values.
            mask (`torch.Tensor`):
                Binary mask of the same shape as `entropies`, where `1` indicates valid tokens and `0` padding.
            threshold (`float`):
                Quantile threshold between `0.0` and `1.0` to select high-entropy tokens.

        Returns:
            `torch.Tensor`:
                Boolean mask of shape (batch_size, seq_len), where `True` indicates tokens with entropy >= threshold
                and `False` otherwise.
        """
        local = entropies[mask.bool()].float()

        # Use a negative pad_value as a sentinel because entropy values are always >= 0.
        # This guarantees that the sentinel cannot collide with any real entropy value.
        pad_value = -1e9

        # Pad across processes so that every rank has the same tensor length
        padded = self.accelerator.pad_across_processes(local, dim=0, pad_index=pad_value)
        gathered = self.accelerator.gather(padded)

        # Drop sentinel values (safe because no entropy can be negative)
        gathered = gathered[gathered != pad_value]

        if gathered.numel() == 0:
            return torch.zeros_like(entropies, dtype=torch.bool)

        entropy_threshold = torch.quantile(gathered, threshold)
        masked_entropies = entropies * mask.float()
        entropy_mask = masked_entropies >= entropy_threshold
        return entropy_mask & mask.bool()  # ensure padding tokens are always masked out

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size = None,
        compute_entropy = False,
        compute_efficient = False,
        *args,
        **kwargs,
    ):
        # All Unsloth code in this function is licensed under AGPL3.
        if compute_efficient:
            # No scoring forward may run (vLLM, beta = 0): latch precision here, not from the shared env.
            _unsloth_grpo_autocast_unscored(self)
            # The current policy is then scored only in the zoo loss: refuse the kept set there too.
            _unsloth_grpo_refuse_unscored_support(self)
            return None, None
        else:
            _unsloth_grpo_autocast(self)

            compute_aux_loss = kwargs.get("compute_aux_loss", None)

            # Body-local: this source is copied out without this module's imports. #6960.
            _grpo_vision_chunks = None
            try:
                from unsloth_zoo.rl_replacements import grpo_vision_chunks as _grpo_vision_chunks
            except Exception:
                pass
            # Collected even without the zoo: an older one must cost only image slicing.
            vision_inputs = _unsloth_grpo_vision_inputs(kwargs)
            if _grpo_vision_chunks is None and vision_inputs.get("pixel_values", None) is not None:
                raise RuntimeError(
                    "Unsloth: vision GRPO needs an unsloth_zoo build that exports "
                    "grpo_vision_chunks, the shared multimodal key tuple and chunker "
                    "used by both GRPO logprob paths. Please upgrade unsloth_zoo to "
                    "2026.9.5 or newer: pip install -U unsloth_zoo"
                )
            pixel_values = vision_inputs.get("pixel_values", None)
            image_grid_thw = vision_inputs.get("image_grid_thw", None)
            video_grid_thw = vision_inputs.get("video_grid_thw", None)
            if pixel_values is None:
                # Only a sentinel below (the chunks carry the tensors): video rows must take the
                # vision path too, since left and sequence packing move M-RoPE video positions.
                pixel_values = vision_inputs.get("pixel_values_videos", None)
            num_images = vision_inputs.get("num_images", None)
            # Transformers 5.x needs token_type_ids/mm_token_type_ids for some vision models.
            token_type_ids = vision_inputs.get("token_type_ids", None)
            mm_token_type_ids = vision_inputs.get("mm_token_type_ids", None)
            if (
                mm_token_type_ids is not None
                or image_grid_thw is not None
                or video_grid_thw is not None
            ):
                mm_token_type_ids = _unsloth_fix_mm_token_type_ids(
                    self.processing_class, input_ids, mm_token_type_ids
                )
                vision_inputs["mm_token_type_ids"] = mm_token_type_ids

            unwrapped_model = self.accelerator.unwrap_model(model, keep_fp32_wrapper = False)

            from unsloth_zoo.rl_replacements import (
                chunked_hidden_states_selective_log_softmax,
                chunked_selective_log_softmax,
            )

            # Old / ref logps must use the policy pass's scorer, else the ratio starts away from 1 and the KL is biased.
            try:
                if os.environ.get("UNSLOTH_GRPO_LOGPROB_RECOMPUTE", "0") == "1":
                    from unsloth_zoo.rl_replacements import grpo_truncated_support_kwargs
                    if not grpo_truncated_support_kwargs(self):
                        from unsloth_zoo.rl_replacements import (
                            chunked_hidden_states_selective_log_softmax_recompute as chunked_hidden_states_selective_log_softmax,
                        )
                else:
                    from unsloth_zoo.rl_replacements import (
                        grpo_chunked_hidden_states_logprobs as chunked_hidden_states_selective_log_softmax,
                    )
            except ImportError:
                pass

            output_embeddings = unwrapped_model.get_output_embeddings()
            lm_head = output_embeddings.weight
            lm_head_bias = getattr(output_embeddings, "bias", None)
            if lm_head_bias is not None and not UNSLOTH_ZOO_HAS_HEAD_BIAS:
                import warnings
                warnings.warn(
                    "Unsloth: this model's output head has a bias, which GRPO scoring needs "
                    "a newer unsloth_zoo to include. Please upgrade: pip install -U unsloth_zoo",
                    stacklevel = 2,
                )
                lm_head_bias = None
            _head_log_softmax = chunked_hidden_states_selective_log_softmax
            # ZeRO-3 released the head after the forward: score with one gathered read-only copy.
            _zero3_head = None
            try:
                from unsloth_zoo.fused_losses.cross_entropy_loss import (
                    unsloth_zero3_gathered,
                    unsloth_zero3_partitioned,
                )
            except (ImportError, NotImplementedError):
                # Older zoo, or zoo refusing to import without an accelerator (CPU gloo workers).
                pass
            else:
                if unsloth_zero3_partitioned(lm_head) or unsloth_zero3_partitioned(
                    getattr(output_embeddings, "bias", None)
                ):
                    _zero3_head = (
                        unsloth_zero3_gathered(lm_head),
                        unsloth_zero3_gathered(getattr(output_embeddings, "bias", None)),
                    )
                    lm_head = _zero3_head[0]
                    if lm_head_bias is not None:
                        lm_head_bias = _zero3_head[1]

            def chunked_hidden_states_selective_log_softmax(
                hidden_states, _lm_head, *args, **kwargs
            ):
                # FSDP2's forward hooks replace Parameters. Read the current head
                # at projection time, including the bias, instead of retaining a shard.
                if _zero3_head is not None:
                    head_weight, head_bias = _zero3_head
                else:
                    head_weight = output_embeddings.weight
                    head_bias = getattr(output_embeddings, "bias", None)
                if UNSLOTH_ZOO_HAS_HEAD_BIAS:
                    kwargs["lm_head_bias"] = head_bias
                return _head_log_softmax(
                    hidden_states,
                    head_weight,
                    *args,
                    **kwargs,
                )

            # UNSLOTH_GRPO_TRUNCATED_SUPPORT (=1 or zoo's automatic gate): score over the rollout's
            # top_k / top_p / min_p kept set, matching vLLM's processed_logprobs. The zoo loss refuses to mix
            # normalisations unless this marker says the old / ref logps were scored the same way.
            try:
                from unsloth_zoo.rl_replacements import grpo_truncated_support_kwargs
                _support_kwargs = grpo_truncated_support_kwargs(self)
            except Exception:
                _support_kwargs = {}
            _support_mismatch = _unsloth_grpo_support_mismatch(self) if _support_kwargs else []
            if _support_mismatch:
                # Left unmarked, the zoo loss drops the kept set for the current policy as well.
                if not getattr(self, "_unsloth_truncated_support_refused", False):
                    import warnings
                    warnings.warn(
                        "Unsloth: not scoring over the rollouts' top_k / top_p / min_p kept set "
                        "(UNSLOTH_GRPO_TRUNCATED_SUPPORT=1, or automatic for truncated vLLM "
                        "rollouts feeding the IS correction or off-policy mask) because "
                        f"{', '.join(_support_mismatch)}: those logprobs come from weights other "
                        "than the rollout's, whose top_k / top_p / min_p kept set differs from the "
                        "sampler's, so per-policy kept sets give unnormalised PPO ratios and KL. "
                        "Scoring over the full vocabulary instead.",
                        stacklevel = 2,
                    )
                    # Also silences the zoo's refusal, which would blame the unsloth version.
                    self._unsloth_truncated_support_refused = True
                _support_kwargs = {}
            if _support_kwargs:
                from functools import partial

                chunked_hidden_states_selective_log_softmax = partial(
                    chunked_hidden_states_selective_log_softmax,
                    **_support_kwargs,
                )
                chunked_selective_log_softmax = partial(
                    chunked_selective_log_softmax,
                    **_support_kwargs,
                )
                self._unsloth_truncated_support_scored = True

            # Size on the dtype the forward actually runs in: with autocast off that is the model's own dtype.
            forward_dtype = (
                self._autocast_dtype if getattr(self, "_autocast_enabled", True) else lm_head.dtype
            )
            dtype_bytes = 16 if forward_dtype in [torch.float16, torch.bfloat16] else 32
            total_rows = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            hidden_dim = lm_head.shape[1]
            vocab_dim = lm_head.shape[0]

            if self.args.unsloth_grpo_mini_batch is None:
                B, multiplier = autotune_batch_and_chunks(
                    total_rows,
                    seq_len,
                    hidden_dim,
                    vocab_dim,
                    dtype_bytes,
                    self.args.unsloth_logit_chunk_multiplier,
                )
                B = total_rows // B
            else:
                B = self.args.unsloth_grpo_mini_batch

                if self.args.unsloth_logit_chunk_multiplier is None:
                    multiplier = max(4, seq_len // 4096)
                else:
                    multiplier = self.args.unsloth_logit_chunk_multiplier

            all_logprobs_list = []
            if pixel_values is None:
                left_pad_tokens_per_prompt = calculate_pad_tokens_in_prompt(
                    input_ids, logits_to_keep, self.processing_class.pad_token_id
                )
                max_left_pad = torch.max(left_pad_tokens_per_prompt).item()
                input_ids = left_pack_padding(input_ids, self.processing_class.pad_token_id)
                attention_mask = input_ids != self.processing_class.pad_token_id
                attention_mask = attention_mask.to(attention_mask.dtype)
            else:
                max_left_pad = 0

            import math

            total_samples = input_ids.shape[0]
            batch_size = math.ceil(total_samples / B)

            input_ids_chunks = []
            attention_mask_chunks = []
            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)
                input_ids_chunks.append(input_ids[start:end])
                attention_mask_chunks.append(attention_mask[start:end])

            # One chunker shared with the gradient pass, so the two cannot disagree.
            if _grpo_vision_chunks is None:
                # Image-indexed keys already raised above, so only per-sample ones are left.
                vision_chunks = []
                for _start in range(0, total_samples, batch_size):
                    _end = min(_start + batch_size, total_samples)
                    vision_chunks.append(
                        {
                            _key: vision_inputs[_key][_start:_end]
                            for _key in ("token_type_ids", "mm_token_type_ids")
                            if vision_inputs.get(_key, None) is not None
                        }
                    )
            else:
                vision_chunks = _grpo_vision_chunks(vision_inputs, total_samples, batch_size)

            temperature = self.temperature
            model_config = _unsloth_get_model_config(model)
            if detect_logit_transforms is not None:
                # model_config, not model: under DDP/Accelerate `model` is a wrapper that does not forward .config, so the helper would report zeros.
                _transforms = detect_logit_transforms(model_config)
                logit_softcapping = _transforms["logit_softcapping"]
                logit_scale_multiply = _transforms["logit_scale_multiply"]
                logit_scale_divide = _transforms["logit_scale_divide"]
            else:
                logit_softcapping = _unsloth_get_final_logit_softcapping(model)
                logit_scale_multiply, logit_scale_divide = _unsloth_resolve_logit_scales(
                    model_config
                )

            zipped_inputs = zip(
                input_ids_chunks,
                attention_mask_chunks,
                vision_chunks,
            )
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"

            # Sequence packing (default on; UNSLOTH_GRPO_SEQ_PACKING=0 disables): one varlen [1, sum L] forward replaces the padded [B, Lmax] loop and fixes the left-pad RoPE error. Self-verified against the per-row forward, re-checked as T grows, and falls back if a backend ignores packed_seq_lengths.
            logprobs = None

            # PrefixGrouper (GRPO shared-prompt dedup, default ON): G completions share the prompt, so storing it once behind a FlexAttention shared-prefix mask cuts the trunk forward from G*(P+R) to P+G*R tokens. Gated by UNSLOTH_GRPO_PREFIX_GROUPER, a tok_r auto-gate and a first-use self-verify, so a mask/isolation regression cannot ship silently.
            _pg_result = None
            _pg_use = False
            _pg_skip_pk = False  # once a shape is PG-verified, skip the full-row forward
            _pg_forward_fn = None  # deferred PG forward (runs at the verify site below)
            _pg_num_gen = getattr(self, "num_generations", None)
            # Env gate hoisted to module level (mirrored via RL_PRE_ITEMS). Skip PG under vLLM: the rollout dominates the step, so PG saves little and its self-verify is net overhead.
            _pg_engage = (
                UNSLOTH_GRPO_PREFIX_GROUPER_ON
                and not getattr(self, "use_vllm", False)
                and not getattr(unwrapped_model, "_unsloth_prefix_grouper_nograd_disabled", False)
            )
            if _pg_engage:
                try:
                    # Skip softcap models (the flex kernel never applies attn_logit_softcapping) and hybrid SSM / MoE models: only the threaded attention forwards get shared-prefix isolation, so a decoder that does not forward prefix_seg_info leaks suffixes across completions. PG also rides on sequence packing, so it needs the same zoo masked-column guard.
                    _pg_cfg = getattr(unwrapped_model, "config", None)
                    _pg_engage = (
                        _pg_enabled_fn()
                        and UNSLOTH_ZOO_HAS_MASKED_COL_GUARD
                        and pixel_values is None
                        and token_type_ids is None
                        and mm_token_type_ids is None
                        and _pg_num_gen is not None
                        and _pg_num_gen >= 2
                        and not getattr(_pg_cfg, "attn_logit_softcapping", None)
                        # Normal backends apply config.attention_dropout in training; the flex path is deterministic, so skip PG when it is set.
                        and not getattr(_pg_cfg, "attention_dropout", 0)
                        and not any(
                            getattr(_pg_cfg, _pg_a, None) is not None
                            for _pg_a in (
                                "mamba_d_ssm",
                                "mamba_d_state",
                                "mamba_expand",
                                "num_experts",
                                "num_local_experts",
                                "n_routed_experts",
                                "moe_intermediate_size",
                            )
                        )
                    )
                except Exception:
                    _pg_engage = False
            if _pg_engage:
                try:
                    _pg_pad = self.processing_class.pad_token_id
                    # Cap the PG span (P+max(R)) at the sliding window, like the packed _pk_sw guard.
                    _pg_sw = getattr(
                        getattr(unwrapped_model, "config", None), "sliding_window", None
                    )
                    if not (isinstance(_pg_sw, int) and _pg_sw > 0):
                        _pg_sw = None
                    _pg_layout = _pg_build_layout(
                        input_ids,
                        logits_to_keep,
                        _pg_pad,
                        _pg_num_gen,
                        left_pad_tokens_per_prompt,
                        max_segment_cap = _pg_sw,
                    )
                    _pg_unsafe = getattr(
                        unwrapped_model, "_unsloth_prefix_grouper_nograd_unsafe", None
                    )
                    if _pg_unsafe is None:
                        _pg_unsafe = set()
                    if _pg_layout is not None and _pg_layout.signature not in _pg_unsafe:
                        _pg_sig = _pg_layout.signature
                        _pg_verified = getattr(
                            unwrapped_model, "_unsloth_prefix_grouper_nograd_verified", None
                        )
                        if _pg_verified is None:
                            _pg_verified = set()
                        _pg_chunks = max(1, total_rows * multiplier)

                        def _pg_run_forward(_pg_layout = _pg_layout, _pg_chunks = _pg_chunks):
                            with _get_inference_mode_context_manager(model):
                                with torch.amp.autocast(
                                    device_type = DEVICE_TYPE_TORCH,
                                    dtype = self._autocast_dtype,
                                    enabled = getattr(self, "_autocast_enabled", True),
                                ):
                                    _pg_hidden = unwrapped_model(
                                        input_ids = _pg_layout.flat_ids,
                                        position_ids = _pg_layout.position_ids,
                                        prefix_seg_info = _pg_layout.prefix_seg_info,
                                        use_cache = False,
                                    ).logits
                                    _pg_r = _pg_layout.extract_logps(
                                        _pg_hidden,
                                        lm_head,
                                        chunked_hidden_states_selective_log_softmax,
                                        _pg_chunks,
                                        logit_scale_multiply,
                                        logit_scale_divide,
                                        logit_softcapping,
                                        temperature,
                                    )
                                    _pg_hidden = None  # release before any verify forward
                            device_synchronize()
                            # Clip to the loss window [B, logits_to_keep+max_left_pad].
                            _pg_w = logits_to_keep + max_left_pad
                            if _pg_r.shape[1] > _pg_w:
                                _pg_r = _pg_r[:, -_pg_w:]
                            return _pg_r

                        # Trust only within the verified envelope: re-verify when T or the longest segment grows, like the packed path.
                        _pg_T = int(_pg_layout.flat_ids.shape[1])
                        _pg_maxseg = int(_pg_layout.position_ids.max()) + 1
                        _pg_env = (
                            _pg_verified.get(_pg_sig) if isinstance(_pg_verified, dict) else None
                        )
                        if (not _pg_verify_on()) or (
                            _pg_env is not None and _pg_T <= _pg_env[0] and _pg_maxseg <= _pg_env[1]
                        ):
                            # Trusted shape: run PG now and skip the full-row forward below.
                            _pg_result = _pg_run_forward()
                            _pg_use = True
                            _pg_skip_pk = True
                        else:
                            # Unverified shape: defer the forward until the packed reference exists, so a declined packed path never wastes a whole-batch PG forward.
                            _pg_forward_fn = _pg_run_forward
                except Exception as _pg_err:
                    _pg_result = None
                    _pg_use = False
                    _pg_skip_pk = False
                    _pg_forward_fn = None
                    # A FlexAttention/Triton compile failure or OOM here is GPU-wide, not layout-specific, so retrying every step just re-pays it. Disable PG persistently; the packed/padded path below still gives the exact result.
                    unwrapped_model._unsloth_prefix_grouper_nograd_disabled = True
                    if isinstance(_pg_err, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
                    if UNSLOTH_ENABLE_LOGGING:
                        print(
                            f"[Unsloth] GRPO PrefixGrouper (no-grad) disabled (fell back to packed): {_pg_err!r}",
                            flush = True,
                        )

            # Sequence packing (default on; UNSLOTH_GRPO_SEQ_PACKING=0 disables): one varlen block-diagonal forward replaces the padded loop exactly and fixes its left-pad RoPE error. Self-verified, re-checked as T grows, falls back if a backend ignores packed_seq_lengths, and lm_head runs on completion positions only.
            _pk_result = None
            _pk_use = False
            _pk_T = None
            # True only while the packed forward and its scatter run, so an OOM elsewhere (setup, per-row verifier) is not read as a length limit.
            _pk_in_packed = False
            _pk_enabled = UNSLOTH_GRPO_SEQ_PACKING_ON
            # Without zoo#840's masked-column guard, zeroed prompt/pad columns turn NaN in exp().
            _pk_enabled = _pk_enabled and UNSLOTH_ZOO_HAS_MASKED_COL_GUARD
            _pk_ok = getattr(unwrapped_model, "_unsloth_seq_packing_nograd_ok", None)
            if (
                _pk_enabled
                and not _pg_skip_pk
                and pixel_values is None
                and token_type_ids is None
                and mm_token_type_ids is None
                and _pk_ok is not False
            ):
                try:
                    _pk_pad = self.processing_class.pad_token_id
                    _pk_keep = input_ids != _pk_pad
                    _pk_len = _pk_keep.sum(dim = 1)
                    _pk_len_cpu = _pk_len.tolist()  # single GPU->CPU sync, reused below
                    _pk_nz_cpu = [_n for _n in _pk_len_cpu if _n > 0]
                    _pk_T = sum(_pk_nz_cpu)
                    _pk_L = input_ids.shape[1]
                    _pk_W = logits_to_keep + max_left_pad
                    _pk_maxseg = max(_pk_nz_cpu) if _pk_nz_cpu else 0
                    # Sliding-window models lose the per-sequence local window in a packed stream.
                    _pk_sw = getattr(
                        getattr(unwrapped_model, "config", None), "sliding_window", None
                    )
                    _pk_sw_ok = not (isinstance(_pk_sw, int) and _pk_sw > 0 and _pk_maxseg > _pk_sw)
                    # Skip the packed forward entirely at known-unsafe lengths, avoiding a wasted pass or OOM.
                    _pk_unsafe = getattr(
                        unwrapped_model, "_unsloth_seq_packing_nograd_unsafe_T", None
                    )
                    # Cap the flattened forward at one padded [batch_size, seq_len] mini-batch's token budget; anything larger uses the chunked padded loop.
                    _pk_cap = batch_size * seq_len
                    # Decided from the host lengths alone, so a capped batch skips the flatten and the active-row sync below.
                    _pk_fits = (
                        _pk_T >= 2
                        and _pk_T <= _pk_cap
                        and len(_pk_nz_cpu) > 0
                        and _pk_sw_ok
                        and not (_pk_unsafe is not None and _pk_T >= _pk_unsafe)
                    )
                    if _pk_fits:
                        _pk_flat = input_ids[_pk_keep].unsqueeze(0)
                        # Per-row completion mask (same as the loss); prompt-only rows count as inactive.
                        _pk_cmask = create_completion_attention_mask(
                            input_ids[:, -_pk_W:], left_pad_tokens_per_prompt, max_left_pad, _pk_pad
                        )
                        _pk_active = int(_pk_cmask.any(dim = 1).sum())
                    if _pk_fits and (_pk_ok is True or _pk_active >= 2):
                        _pk_pos = (_pk_keep.cumsum(dim = 1) - 1)[_pk_keep].unsqueeze(0)
                        _pk_chunks = max(1, total_rows * multiplier)
                        _pk_nz_idx = _pk_keep.nonzero(
                            as_tuple = False
                        )  # [T, 2] = (row, col), row-major
                        _pk_within = _pk_nz_idx[1:, 0] == _pk_nz_idx[:-1, 0]  # [T-1]
                        # Per-row completion start after left-packing, matching create_completion_attention_mask.
                        _pk_cstart = (_pk_L - logits_to_keep) - left_pad_tokens_per_prompt  # [rows]
                        _pk_ctgt = (_pk_nz_idx[1:, 1] >= _pk_cstart[_pk_nz_idx[1:, 0]]) & _pk_within
                        _pk_in_packed = True
                        with _get_inference_mode_context_manager(model):
                            with torch.amp.autocast(
                                device_type = DEVICE_TYPE_TORCH,
                                dtype = self._autocast_dtype,
                                enabled = getattr(self, "_autocast_enabled", True),
                            ):
                                # use_cache=False: a KV cache silently disables varlen packing.
                                _pk_hidden = unwrapped_model(
                                    input_ids = _pk_flat,
                                    position_ids = _pk_pos,
                                    packed_seq_lengths = torch.tensor(
                                        _pk_nz_cpu, dtype = torch.int32, device = input_ids.device
                                    ),
                                    use_cache = False,
                                ).logits
                                _pk_out = _pk_hidden[0, :-1, :][_pk_ctgt].unsqueeze(0)
                                _pk_ids = _pk_flat[0, 1:][_pk_ctgt].unsqueeze(0)
                                # Hidden states or logits? Logits mean the forward already applied scaling/softcapping.
                                if _unsloth_grpo_returns_hidden_states(
                                    unwrapped_model, _pk_out, lm_head
                                ):
                                    _pk_sel = chunked_hidden_states_selective_log_softmax(
                                        _pk_out,
                                        lm_head,
                                        _pk_ids,
                                        _pk_chunks,
                                        logit_scale_multiply,
                                        logit_scale_divide,
                                        logit_softcapping,
                                        temperature,
                                    )[0]
                                else:
                                    # Model returned logits directly: scaling/softcapping already applied by the model forward.
                                    _pk_sel = chunked_selective_log_softmax(
                                        _pk_out,
                                        _pk_ids,
                                        temperature,
                                        _pk_chunks,
                                    )[0]
                        # GPT-OSS offload race guard, matching the padded loop.
                        device_synchronize()
                        # Scatter each logprob back to its (row, col) so [:, -_pk_W:] matches the padded path.
                        _pk_tgt = (_pk_nz_idx[1:, 0] * _pk_L + _pk_nz_idx[1:, 1])[_pk_ctgt]
                        _pk_result = (
                            torch.zeros(
                                total_rows * _pk_L,
                                dtype = torch.float32,
                                device = input_ids.device,
                            )
                            .index_put((_pk_tgt,), _pk_sel.to(torch.float32))
                            .view(total_rows, _pk_L)[:, -_pk_W:]
                        )
                        _pk_in_packed = False
                        # Re-verify when T or the longest segment grows past the verified envelope; a LongRoPE cache switch can change the result.
                        _pk_vT = int(
                            getattr(unwrapped_model, "_unsloth_seq_packing_nograd_verified_T", 0)
                        )
                        _pk_vS = int(
                            getattr(unwrapped_model, "_unsloth_seq_packing_nograd_verified_seg", 0)
                        )
                        # Debug: hand-edit this condition to force re-verify every step.
                        if _pk_ok is True and _pk_T <= _pk_vT and _pk_maxseg <= _pk_vS:
                            _pk_use = True  # already verified for this shape
                        else:
                            # verify against the per-row forward (ground truth)
                            _pk_ref = torch.zeros_like(_pk_result)
                            with _get_inference_mode_context_manager(model):
                                with torch.amp.autocast(
                                    device_type = DEVICE_TYPE_TORCH,
                                    dtype = self._autocast_dtype,
                                    enabled = getattr(self, "_autocast_enabled", True),
                                ):
                                    for _pk_i in range(total_rows):
                                        _pk_ni = _pk_len_cpu[_pk_i]
                                        if _pk_ni < 2:
                                            continue
                                        _pk_rmask = _pk_keep[_pk_i]
                                        _pk_real = input_ids[_pk_i][_pk_rmask].unsqueeze(0)
                                        _pk_rpos = torch.arange(
                                            _pk_ni, device = input_ids.device
                                        ).unsqueeze(0)
                                        _pk_rh = unwrapped_model(
                                            input_ids = _pk_real,
                                            position_ids = _pk_rpos,
                                            use_cache = False,
                                        ).logits
                                        _pk_rout = _pk_rh[:, :-1, :]
                                        # Hidden states or logits? Logits mean the forward already applied scaling/softcapping.
                                        if _unsloth_grpo_returns_hidden_states(
                                            unwrapped_model, _pk_rout, lm_head
                                        ):
                                            _pk_rsel = chunked_hidden_states_selective_log_softmax(
                                                _pk_rout,
                                                lm_head,
                                                _pk_real[:, 1:],
                                                1,
                                                logit_scale_multiply,
                                                logit_scale_divide,
                                                logit_softcapping,
                                                temperature,
                                            )[0]
                                        else:
                                            # Model returned logits directly: scaling/softcapping already applied by the model forward.
                                            _pk_rsel = chunked_selective_log_softmax(
                                                _pk_rout,
                                                _pk_real[:, 1:],
                                                temperature,
                                                1,
                                            )[0]
                                        _pk_rcols = _pk_rmask.nonzero(as_tuple = False).squeeze(1)[
                                            1:
                                        ] - (_pk_L - _pk_W)
                                        _pk_rkeep = _pk_rcols >= 0
                                        _pk_ref[_pk_i, _pk_rcols[_pk_rkeep]] = _pk_rsel[
                                            _pk_rkeep
                                        ].to(torch.float32)
                            device_synchronize()
                            # Compare over the loss-mask region only.
                            _pk_cm = _pk_cmask.float()
                            _pk_diff = float(((_pk_result - _pk_ref).abs() * _pk_cm).max())
                            if UNSLOTH_ENABLE_LOGGING:
                                print(
                                    f"[Unsloth] GRPO seq-packing (no-grad) verify: T={_pk_T} maxseg={_pk_maxseg} packed-vs-perrow max|d|={_pk_diff:.4f}",
                                    flush = True,
                                )
                            # Kernel-noise floor is ~0.25; cross-sample contamination is >= 2.4.
                            if _pk_diff < 7e-1:
                                unwrapped_model._unsloth_seq_packing_nograd_ok = True
                                # Widen the trusted shape only when at least 2 completion rows exercised cross-sample packing; a single row proves nothing.
                                if _pk_active >= 2:
                                    unwrapped_model._unsloth_seq_packing_nograd_verified_T = max(
                                        _pk_vT, _pk_T
                                    )
                                    unwrapped_model._unsloth_seq_packing_nograd_verified_seg = max(
                                        _pk_vS, _pk_maxseg
                                    )
                                _pk_ok = True
                                _pk_use = True
                            else:
                                _pk_use = False
                                if _pk_diff >= 1.5:
                                    # Contamination (attention ignores the packed mask): disable packing.
                                    unwrapped_model._unsloth_seq_packing_nograd_ok = False
                                else:
                                    # Likely a length boundary (LongRoPE): mark unsafe, keep smaller shapes.
                                    unwrapped_model._unsloth_seq_packing_nograd_unsafe_T = (
                                        _pk_T if _pk_unsafe is None else min(_pk_unsafe, _pk_T)
                                    )
                                if UNSLOTH_ENABLE_LOGGING:
                                    print(
                                        f"[Unsloth] GRPO seq-packing (no-grad) fell back at T={_pk_T} (diff={_pk_diff:.3f})",
                                        flush = True,
                                    )
                except Exception as _pk_err:
                    # Any failure: drop intermediates and use the padded loop for this batch.
                    _pk_hidden = None
                    _pk_sel = None
                    _pk_result = None
                    _pk_use = False
                    if isinstance(_pk_err, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                    if (
                        isinstance(_pk_err, torch.cuda.OutOfMemoryError)
                        and _pk_in_packed
                        and _pk_T is not None
                    ):
                        # An OOM in the packed forward says this flattened length does not fit, not that packing is wrong: keep packing smaller batches.
                        _pk_unsafe = getattr(
                            unwrapped_model, "_unsloth_seq_packing_nograd_unsafe_T", None
                        )
                        _pk_unsafe = _pk_T if _pk_unsafe is None else min(_pk_unsafe, _pk_T)
                        unwrapped_model._unsloth_seq_packing_nograd_unsafe_T = _pk_unsafe
                        if UNSLOTH_ENABLE_LOGGING:
                            print(
                                f"[Unsloth] GRPO sequence-packing (no-grad) OOM at T={_pk_T}: packing capped below T={_pk_unsafe}, this batch uses the padded loop: {_pk_err!r}",
                                flush = True,
                            )
                    else:
                        unwrapped_model._unsloth_seq_packing_nograd_ok = False
                        if UNSLOTH_ENABLE_LOGGING:
                            print(
                                f"[Unsloth] GRPO sequence-packing (no-grad) disabled (fell back to padded): {_pk_err!r}",
                                flush = True,
                            )
            # PrefixGrouper first-use self-verify (no-grad): compare the untrusted PG result to the packed result over the completion mask. Below tol_ok trust the structure, at or above TOL_KILL mark it unsafe forever, borderline falls back for this shape.
            if _pg_forward_fn is not None and not _pg_use:
                if _pk_use and _pk_result is not None:
                    try:
                        # Deferred PG forward, run only now that the packed reference exists.
                        _pg_result = _pg_forward_fn()
                        _pg_W2 = logits_to_keep + max_left_pad
                        _pg_cm = create_completion_attention_mask(
                            input_ids[:, -_pg_W2:],
                            left_pad_tokens_per_prompt,
                            max_left_pad,
                            self.processing_class.pad_token_id,
                        ).float()
                        _pg_a = _pg_result[:, -_pg_W2:].float()
                        _pg_b = _pk_result[:, -_pg_W2:].float()
                        _pg_diff = float(((_pg_a - _pg_b).abs() * _pg_cm).max())
                        if UNSLOTH_ENABLE_LOGGING:
                            print(
                                f"[Unsloth] GRPO PrefixGrouper (no-grad) verify: sig={_pg_layout.signature} "
                                f"shared-prefix vs full-row-packed max|d|={_pg_diff:.4f}",
                                flush = True,
                            )
                        if _pg_diff < _pg_tol_ok():
                            _pg_v = getattr(
                                unwrapped_model, "_unsloth_prefix_grouper_nograd_verified", None
                            )
                            if not isinstance(_pg_v, dict):
                                _pg_v = {}
                            _pg_vT = int(_pg_layout.flat_ids.shape[1])
                            _pg_vS = int(_pg_layout.position_ids.max()) + 1
                            _pg_old = _pg_v.get(_pg_layout.signature, (0, 0))
                            _pg_v[_pg_layout.signature] = (
                                max(_pg_vT, _pg_old[0]),
                                max(_pg_vS, _pg_old[1]),
                            )
                            unwrapped_model._unsloth_prefix_grouper_nograd_verified = _pg_v
                            _pg_use = True
                        else:
                            _pg_u = getattr(
                                unwrapped_model, "_unsloth_prefix_grouper_nograd_unsafe", None
                            )
                            if _pg_u is None:
                                _pg_u = set()
                            if _pg_diff >= _PG_TOL_KILL:
                                _pg_u.add(_pg_layout.signature)
                                unwrapped_model._unsloth_prefix_grouper_nograd_unsafe = _pg_u
                            _pg_use = False
                    except Exception as _pg_err3:
                        _pg_result = None
                        _pg_use = False
                        if isinstance(_pg_err3, torch.cuda.OutOfMemoryError):
                            torch.cuda.empty_cache()
                        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
                        if UNSLOTH_ENABLE_LOGGING:
                            print(
                                f"[Unsloth] GRPO PrefixGrouper (no-grad) verify failed (fell back to packed): {_pg_err3!r}",
                                flush = True,
                            )
                # No packed reference (packing off or failed) means this cannot be verified, so fall back.

            if _pg_use and _pg_result is not None:
                logprobs = _pg_result  # PrefixGrouper verified/trusted -> skip the loop
                zipped_inputs = []
            elif _pk_use and _pk_result is not None:
                logprobs = _pk_result  # verified -> skip the loop
                zipped_inputs = []
            else:
                _pk_hidden = _pk_sel = _pk_result = _pk_ref = None
                # Index buffers of a declined packed batch would otherwise live through the padded loop.
                _pk_keep = _pk_len = _pk_flat = _pk_pos = _pk_cmask = None
                _pk_nz_idx = _pk_within = _pk_cstart = _pk_ctgt = _pk_tgt = _pk_ids = _pk_cm = None

            with _get_inference_mode_context_manager(model):
                for (
                    input_ids_chunk,
                    attention_mask_chunk,
                    vision_chunk,
                ) in zipped_inputs:
                    with torch.amp.autocast(
                        device_type = DEVICE_TYPE_TORCH,
                        dtype = self._autocast_dtype,
                        enabled = getattr(self, "_autocast_enabled", True),
                    ):
                        if pixel_values is None:
                            outputs = unwrapped_model(
                                input_ids = input_ids_chunk,
                                attention_mask = attention_mask_chunk,
                                **vision_chunk,
                            )

                            logits_chunk = outputs.logits
                            del outputs

                            completion_input_ids_chunk = input_ids_chunk[
                                :, -(logits_to_keep + max_left_pad) :
                            ]
                            logits_chunk = logits_chunk[
                                :, -(logits_to_keep + max_left_pad + 1) :, :
                            ]
                            logits_chunk = logits_chunk[:, :-1, :]
                            # Hidden states or logits? Logits mean the forward already applied scaling/softcapping.
                            if _unsloth_grpo_returns_hidden_states(
                                unwrapped_model, logits_chunk, lm_head
                            ):
                                logprobs_chunk = chunked_hidden_states_selective_log_softmax(
                                    logits_chunk,
                                    lm_head,
                                    completion_input_ids_chunk,
                                    chunks = input_ids_chunk.shape[0] * multiplier,
                                    logit_scale_multiply = logit_scale_multiply,
                                    logit_scale_divide = logit_scale_divide,
                                    logit_softcapping = logit_softcapping,
                                    temperature = temperature,
                                )
                            else:
                                # Model returned logits directly: scaling/softcapping already applied by the model forward.
                                logprobs_chunk = chunked_selective_log_softmax(
                                    logits_chunk,
                                    completion_input_ids_chunk,
                                    temperature,
                                    input_ids_chunk.shape[0] * multiplier,
                                )
                        else:
                            # VLMs do not take the optimized path in models/, so they never hit the Flash Attn left-padding issue.
                            outputs = unwrapped_model(
                                input_ids = input_ids_chunk,
                                attention_mask = attention_mask_chunk,
                                logits_to_keep = logits_to_keep + 1,
                                **vision_chunk,
                            )

                            logits_chunk = outputs.logits
                            del outputs

                            logits_chunk = logits_chunk[:, :-1, :]
                            completion_input_ids_chunk = input_ids_chunk[:, -logits_to_keep:]
                            # Hidden states or logits? Logits mean the forward already applied scaling/softcapping.
                            if _unsloth_grpo_returns_hidden_states(
                                unwrapped_model, logits_chunk, lm_head
                            ):
                                logprobs_chunk = chunked_hidden_states_selective_log_softmax(
                                    logits_chunk,
                                    lm_head,
                                    completion_input_ids_chunk,
                                    chunks = input_ids_chunk.shape[0] * multiplier,
                                    logit_scale_multiply = logit_scale_multiply,
                                    logit_scale_divide = logit_scale_divide,
                                    logit_softcapping = logit_softcapping,
                                    temperature = temperature,
                                )
                            else:
                                logprobs_chunk = chunked_selective_log_softmax(
                                    logits_chunk,
                                    completion_input_ids_chunk,
                                    temperature,
                                )
                    # Avoids a race with GPT OSS offload_embbed=True; it does not appear to slow models down.
                    device_synchronize()
                    all_logprobs_list.append(logprobs_chunk)
                if logprobs is None:  # padded fallback when packing was not used
                    logprobs = torch.cat(all_logprobs_list, dim = 0)

                entropies = None

            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"
            # aux loss is unused: off by default (router_aux_loss_coef = 0 in models/rl.py) and explicit opt-in is rejected at trainer init, so it is always None. Kept for TRL >= 1.7.0's 3-tuple.
            aux_loss = None
            return logprobs.detach(), entropies, aux_loss  # logps, entropies, aux_loss
            # transformers <= 4.48 does not support logits_to_keep, so drop the logits here; see huggingface/trl#2770.

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

    @profiling_decorator
    def _prepare_inputs(self, generation_batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = split_pixel_values_by_grid(generation_batch)
                generation_batch = _unsloth_grpo_split_vision_by_sample(generation_batch)

                try: generation_batch = shuffle_sequence_dict(generation_batch)

                except: pass
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                self._buffered_inputs = [_unsloth_grpo_unsplit_vision(unsplit_pixel_values_by_grid(batch)) for batch in generation_batches]
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    def _log_completion_extra(self, column: str, values: list):
        """
        Log extra columns to the completions table. Called from reward functions via the `log_extra` kwarg.

        Args:
            column (`str`):
                Name of the column to add.
            values (`list`):
                Values for the column, one per sample in the batch.
        """
        self._pending_extra_logs[column].extend(values)

    def _log_metric(self, name: str, value: float):
        """
        Log a scalar metric from a reward function. Called via the `log_metric` kwarg. Values are averaged over each
        logging step and reported alongside built-in metrics like `kl` and `entropy`.

        Args:
            name (`str`):
                Name of the metric.
            value (`float`):
                Scalar value for this batch.
        """
        self._pending_metrics[name].append(value)

    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        # Allow reward functions to log extra columns to the completions table.
        reward_kwargs["log_extra"] = self._log_completion_extra

        # Allow reward functions to log additional scalar metrics.
        reward_kwargs["log_metric"] = self._log_metric

        # Expose the per-completion environment instances to reward functions (both sync and async paths).
        if self.environments is not None:
            reward_kwargs["environments"] = self.environments

        async_funcs_info = []  # async custom functions for asyncio.gather

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names, strict=True)
        ):
            if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                with profiling_context(self, reward_func_name):
                    if is_conversational({"prompt": prompts[0]}):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions, strict=True)]
                        texts = [
                            apply_chat_template(x, reward_processing_class, **self.chat_template_kwargs)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions, strict=True)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            elif is_async_callable(reward_func):  # Separate async reward funcs to run them in parallel later
                async_funcs_info.append((i, reward_func, reward_func_name))
            else:
                # Run synchronous reward function
                with profiling_context(self, reward_func_name):
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                    if len(output_reward_func) != len(prompts):
                        raise ValueError(
                            f"The reward function '{reward_func_name}' returned {len(output_reward_func)} rewards, but "
                            f"{len(prompts)} were expected (one per prompt-completion pair). Make sure the reward "
                            f"function returns exactly one reward per completion."
                        )
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Execute async custom functions in parallel using asyncio.gather
        if async_funcs_info:

            async def _invoke_async(index, func, func_name):
                with profiling_context(self, func_name):
                    output = await func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    output = [r if r is not None else torch.nan for r in output]
                    if len(output) != len(prompts):
                        raise ValueError(
                            f"The reward function '{func_name}' returned {len(output)} rewards, but {len(prompts)} "
                            f"were expected (one per prompt-completion pair). Make sure the reward function returns "
                            f"exactly one reward per completion."
                        )
                    return index, output

            async def _run_async_funcs():
                coros = [_invoke_async(i, func, func_name) for (i, func, func_name) in async_funcs_info]
                return await asyncio.gather(*coros)

            async_results = asyncio.run_coroutine_threadsafe(_run_async_funcs(), self.async_loop).result()
            for idx, output_reward_func in async_results:
                rewards_per_func[:, idx] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx]
                for key, value in reward_kwargs.items()
                if key not in ("trainer_state", "log_extra", "log_metric")
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func

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

            if self.environment_factories is not None:
                # Tools differ per example across environments, so render each prompt with its own tool schema.
                prompt_ids = []
                multimodal_fields = {}
                for prompt, name in zip(prompts, self._batch_environments, strict=True):
                    tokenized = self.processing_class.apply_chat_template(
                        conversation=[prompt],
                        tools=self._env_tools[name] or None,  # `or None`: Llama bug: tool boilerplate for tools=[]
                        chat_template=self.chat_template,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        **self.chat_template_kwargs,
                    )
                    prompt_ids.append(tokenized["input_ids"][0])
                    # For VLMs, the processor returns extra multimodal fields (pixel_values, image_grid_thw, etc.)
                    for k, v in tokenized.items():
                        if k not in ("input_ids", "attention_mask"):
                            multimodal_fields.setdefault(k, []).append(v)
                # Merge the per-prompt fields so the result matches the single batched `apply_chat_template` call used
                # when there are no environments: image fields (pixel_values, image_grid_thw, ...) are flattened over
                # patches/images and concatenated; per-token fields (e.g. token_type_ids) stay batched per prompt.
                multimodal_fields = {
                    k: torch.cat(v) if isinstance(v[0], torch.Tensor) else [row for prompt_v in v for row in prompt_v]
                    for k, v in multimodal_fields.items()
                }
                return prompt_ids, images, multimodal_fields

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

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields, num_generations, has_tool_images=False):
        device = self.accelerator.device

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # Sync weights if training step changed
            if self.state.global_step != self._last_loaded_step:
                if not getattr(getattr(self.vllm_generation, 'llm', None), 'shared_weights', False):
                    with profiling_context(self, 'sync_weights'):
                        self.vllm_generation.sync_weights()
                self._last_loaded_step = self.state.global_step

            # Generate using vLLM with raw token IDs
            _, completion_ids, logprobs, _ = self.vllm_generation.generate(
                prompts=prompt_ids,
                images=images,
                num_generations=num_generations,
                profiler=profiling_context(self, "vLLM.generate"),
            )
            # vLLM returns per-token top-k logprobs; keep only the top-1 (sampled token) logprob
            logprobs = [[lp[0] for lp in seq] for seq in logprobs]

        elif self.use_transformers_continuous_batching:
            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                self._dist.summon_full_params(self.model_wrapped, recurse=False),
            ):
                # Cast to the appropriate dtype based on training configuration
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                if self.args.cast_lm_head_to_fp32:
                    unwrapped_model.lm_head.to(torch.float32)
                all_outputs = unwrapped_model.generate_batch(
                    prompt_ids,
                    generation_config=self.generation_config,
                    continuous_batching_config=self.continuous_batching_config,
                    progress_bar=False,
                )
                unwrapped_model.train()
            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            logprobs = None

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
            if "mm_token_type_ids" in generate_inputs or "image_grid_thw" in generate_inputs:
                mm_token_type_ids = _unsloth_fix_mm_token_type_ids(
                    self.processing_class,
                    generate_inputs["input_ids"],
                    generate_inputs.get("mm_token_type_ids", None),
                )
                if mm_token_type_ids is not None:
                    generate_inputs["mm_token_type_ids"] = mm_token_type_ids

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
            logprobs = None  # not used in this case

        return completion_ids, logprobs

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

    def _tool_call_loop(self, prompts, prompt_ids, completion_ids, completions, logprobs, images, multimodal_fields):
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
                sync_tool_dict = self._sync_tool_dicts[idx_with_tool]
                async_tool_dict = self._async_tool_dicts[idx_with_tool]
                # Append the last assistant message (which triggered tool_calls) to the prompt
                prompt_completion_tool.append(completions[idx_with_tool][-1])
                async_coros = []
                tool_call_results = []
                for tool_call in tool_call_list:
                    tool_call_count += 1
                    if tool_call["type"] == "function":
                        function = tool_call["function"]
                        name = function["name"]
                        try:
                            if name in sync_tool_dict:
                                tool_call_results.append((name, sync_tool_dict[name](**function["arguments"])))
                            elif name in async_tool_dict:
                                async_coros.append((name, async_tool_dict[name](**function["arguments"])))
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

                if async_coros:

                    async def _run_async_tools(async_coros):
                        coros = [coro for _, coro in async_coros]
                        results = await asyncio.gather(*coros, return_exceptions=True)
                        return [(name, result) for (name, _), result in zip(async_coros, results, strict=False)]

                    async_results = asyncio.run_coroutine_threadsafe(
                        _run_async_tools(async_coros), self.async_loop
                    ).result()

                    for name, result in async_results:
                        if isinstance(result, Exception):
                            tool_failure_count += 1
                            tool_call_results.append((name, {"error": str(result)}))
                        else:
                            tool_call_results.append((name, result))

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
            post_tool_ids, post_tool_logprobs = self._generate_single_turn(
                prompt_completion_tool_ids,
                loop_images,
                loop_multimodal_fields,
                num_generations=1,  # each sample has its own history, so generate one completion per sample
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
                    if logprobs is not None:
                        post_tool_logprobs[idx] = post_tool_logprobs[idx][:new_len]

            # Update tool_mask: the tool result should be 0 and the post-tool 1
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_completion_tool_length = len(prompt_completion_tool_ids[idx])
                prompt_length = len(prompt_ids[idx_with_tool])
                completion_length = len(completion_ids[idx_with_tool])
                post_tool_length = len(post_tool_ids[idx])
                tool_length = prompt_completion_tool_length - prompt_length - completion_length
                tool_mask[idx_with_tool] += [0] * tool_length + [1] * post_tool_length
                if logprobs is not None:
                    logprobs[idx_with_tool] += [0.0] * tool_length + post_tool_logprobs[idx]

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

        return tool_mask, completions, completion_ids, logprobs, tool_call_count, tool_failure_count, tool_images

    def _generate(self, prompts: list):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Copy the prompts to avoid modifying the original list
        prompts = copy.deepcopy(prompts)

        if self.rollout_func is not None:
            # Keep vLLM weights in sync for custom rollouts that rely on vLLM utilities.
            if self.use_vllm and self.state.global_step != self._last_loaded_step:
                if not getattr(getattr(self.vllm_generation, 'llm', None), 'shared_weights', False):
                    with profiling_context(self, 'sync_weights'):
                        self.vllm_generation.sync_weights()
                self._last_loaded_step = self.state.global_step

            # Pass prompts to rollout_func preserving structured messages.
            # Chat templating must happen inside rollout_func, at the backend boundary, so that
            # multimodal content (images, typed content blocks) is not lost before rollout logic runs.
            output = self.rollout_func(prompts, self)
            required_keys = {"prompt_ids", "completion_ids", "logprobs"}
            missing_keys = required_keys - output.keys()
            if missing_keys:
                missing_keys_list = sorted(missing_keys)
                raise ValueError(f"rollout_func must return keys {missing_keys_list} in its output dict.")
            extra_fields = {k: v for k, v in output.items() if k not in required_keys}
            prompt_ids, completion_ids, logprobs = output["prompt_ids"], output["completion_ids"], output["logprobs"]
            images = None
            multimodal_fields = {}
        else:
            prompt_ids, images, multimodal_fields = self._tokenize_prompts(prompts)
            num_generations = self.num_generations if mode == "train" else self.num_generations_eval
            completion_ids, logprobs = self._generate_single_turn(
                prompt_ids, images, multimodal_fields, num_generations
            )
            extra_fields = {}

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

        # Extract tool calls from the completions and (possibly) execute them
        tool_images = []
        if self.tools:
            (
                tool_mask,
                completions,
                completion_ids,
                logprobs,
                tool_call_count,
                tool_failure_count,
                tool_images,
            ) = self._tool_call_loop(
                prompts, prompt_ids, completion_ids, completions, logprobs, images, multimodal_fields
            )
            # Merge tool response images into the images list for the forward pass
            if any(imgs for imgs in tool_images):
                if images is None:
                    images = [imgs if imgs else None for imgs in tool_images]
                else:
                    images = [(existing or []) + new for existing, new in zip(images, tool_images, strict=True)]
        else:
            # Support custom env_mask from rollout_func (e.g., for environment feedback masking)
            # Internally treated as tool_mask - marks model tokens (1) vs external tokens (0)
            tool_mask = extra_fields.pop("env_mask", None)

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

        return prompt_ids, completion_ids, tool_mask, completions, logprobs, extra_fields, images, tool_images

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # `prompt` is optional only when an environment owns the data (e.g. a multi-environment routing dataset that
        # carries only an `environment` column); each rollout's `reset()` then supplies it. Default it here rather than
        # writing it back onto the row, so the placeholder stays out of the `reset()` kwargs built below, while every
        # prompt-derived check downstream (conversational detection, multimodal handling) stays consistent. Without an
        # environment, a missing `prompt` is a malformed dataset and must still fail fast.
        if self.environment_factories is not None:
            prompts = [x.get("prompt", [{"role": "user", "content": ""}]) for x in inputs]
        else:
            prompts = [x["prompt"] for x in inputs]
        # Unsloth: Extract per-sample chat_template_kwargs before metadata is lost
        _ct_ = getattr(self.processing_class, 'chat_template', None) or ''
        _sk_ = {'prompt', 'chosen', 'rejected', 'completion', 'messages', 'label',
                'images', 'image', 'videos', 'video', 'audios', 'audio'}
        self._unsloth_batch_chat_kwargs = []
        for _inp_ in inputs:
            _kw_ = {}
            if isinstance(_inp_, dict):
                for _k_ in _inp_.keys() - _sk_:
                    if _k_ in _ct_ and isinstance(_inp_[_k_], str):
                        _kw_[_k_] = _inp_[_k_]
            self._unsloth_batch_chat_kwargs.append(_kw_)
        prompts = _unsloth_grpo_clean_video_prompts(prompts, self)
        # Resolve each example's environment and draw one reusable instance per rollout from the pool, creating more
        # only when this batch needs more concurrent instances of an environment than exist. `_batch_environments`
        # records each example's environment so `_tokenize_prompts` can render the matching tool schema.
        if self.environment_factories is not None:
            self._batch_environments = [x.get("environment") if self._multi_environment else None for x in inputs]
            if self._multi_environment:
                for name in set(self._batch_environments):
                    if name not in self.environment_factories:
                        raise ValueError(
                            f"Example has `environment={name!r}`, which is not among the environments passed to "
                            f"`environment_factory`. Expected one of: {list(self.environment_factories)}."
                        )
            self.environments = []
            pool_cursor = {}  # how many instances of each environment have been handed out so far this batch
            for name in self._batch_environments:
                pool = self._environment_pool[name]
                index = pool_cursor.get(name, 0)
                if index == len(pool):
                    pool.append(self.environment_factories[name]())
                pool_cursor[name] = index + 1
                self.environments.append(pool[index])

        # Build the per-rollout tool dicts for this batch: the standalone tools plus, for each rollout, the methods of
        # its environment. Done here (not at init) because each example's environment, hence its tools, is data-dependent.
        if self.tools:
            self._sync_tool_dicts = []
            self._async_tool_dicts = []
            for i in range(len(inputs)):
                methods = []
                if self.environments:
                    environment = self.environments[i]
                    methods = [
                        getattr(environment, member_name)
                        for member_name, _ in inspect.getmembers(type(environment), predicate=inspect.isfunction)
                        if member_name not in ("reset", "get_reward") and not member_name.startswith("_")
                    ]
                sync_tool_dict, async_tool_dict = {}, {}
                for tool in self._standalone_tools + methods:
                    if inspect.iscoroutinefunction(tool):
                        async_tool_dict[tool.__name__] = tool
                    else:
                        sync_tool_dict[tool.__name__] = tool
                self._sync_tool_dicts.append(sync_tool_dict)
                self._async_tool_dicts.append(async_tool_dict)

        if self.environments:
            for i, (prompt, environment, x) in enumerate(zip(prompts, self.environments, inputs, strict=True)):
                # `environment` is a control field in multi-environment mode, so it is not forwarded to `reset`.
                reset_kwargs = {k: v for k, v in x.items() if k != "environment"} if self._multi_environment else x
                observation = environment.reset(**reset_kwargs)
                if observation is None:
                    continue
                content = prompt[-1]["content"]
                if isinstance(observation, list) and isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                if isinstance(observation, str) and isinstance(content, list):
                    observation = [{"type": "text", "text": observation}]
                # Rebuild the last message rather than mutating it in place, so the input example is left untouched.
                prompts[i] = prompt[:-1] + [{**prompt[-1], "content": content + observation}]

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [_unsloth_grpo_image_cell(example.get("image")) for example in inputs]
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

        dataset_images = images  # preserve dataset images before _generate may overwrite
        (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            completions,
            sampling_per_token_logps_list,
            extra_fields,
            images,
            tool_images,
        ) = self._generate(prompts)

        _unsloth_clear_stateful_mrope(
            self.accelerator.unwrap_model(self.model, keep_fp32_wrapper = False)
        )
        if images is None:
            images = dataset_images  # restore dataset images (rollout_func path returns None)

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
        if sampling_per_token_logps_list is not None:
            # vLLM replaces a NaN token logprob with `None` (see `extract_logprobs`); map it back to NaN so the
            # tensor builds (`torch.tensor([..., None, ...])` raises "Could not infer dtype of NoneType"). The NaN
            # positions are neutralized in the importance-sampling ratio below so they contribute no correction.
            sampling_per_token_logps = [
                torch.tensor([float("nan") if x is None else x for x in logps])
                for logps in sampling_per_token_logps_list
            ]
            sampling_per_token_logps = pad(
                sampling_per_token_logps,
                padding_value=0.0,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            ).to(device=device)
        else:
            sampling_per_token_logps = None
        if tool_mask_list is not None:
            tool_mask = [torch.tensor(mask) for mask in tool_mask_list]
            tool_mask = pad(
                tool_mask, padding_value=1, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            ).to(device=device)
        else:
            tool_mask = None

        # If mask_truncated_completions is enabled, zero out truncated completions for attention and loss masking
        if self.mask_truncated_completions:
            eos_and_pad = [*self.eos_token_ids, self._tokenizer.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            # Mask completion_mask for attention masking
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
            # Also mask tool_mask for consistency in multi-turn training
            if tool_mask is not None:
                tool_mask = tool_mask * (~is_truncated).unsqueeze(1).int()

        loss_mask = completion_mask if tool_mask is None else completion_mask * tool_mask
        num_items_in_batch = self.accelerator.gather(loss_mask.sum()).sum()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        
        max_left_pad = None
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        # Which name carries "this batch has images" moved twice, and at the bottom of the
        # declared window neither name exists: 0.20.0 through 0.23.1 bind has_images, 0.24.0
        # and up bind images, and 0.18.2 / 0.19.1 have no vision path at all, so every batch
        # there is text only. Probing has_images and falling back to images without a third
        # branch made the floor raise NameError out of the except handler and killed training.
        try:
            _unsloth_text_only = not has_images
        except NameError:
            try:
                _unsloth_text_only = images is None
            except NameError:
                _unsloth_text_only = True
        try:
            _unsloth_video_images = images
        except NameError:
            _unsloth_video_images = None
        try:
            _unsloth_video_prompts = prompts
        except NameError:
            _unsloth_video_prompts = []
        _unsloth_video_kwargs = _unsloth_grpo_video_inputs(
            self, _unsloth_video_prompts, _unsloth_video_images
        )
        if _unsloth_video_kwargs is not None:
            _unsloth_text_only = False
        if _unsloth_text_only:
            # Left pad prompt before calculation old and ref hidden states
            left_pad_tokens_per_prompt = calculate_pad_tokens_in_prompt(prompt_completion_ids, logits_to_keep, self.processing_class.pad_token_id)
            max_left_pad = torch.max(left_pad_tokens_per_prompt).item()
        _use_gc = self.model._unsloth_gradient_checkpointing if hasattr(self.model, '_unsloth_gradient_checkpointing') else getattr(self.args, 'gradient_checkpointing', True)
        self.model.for_training(use_gradient_checkpointing=_use_gc)

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
            if self.environment_factories is not None:
                per_prompt_tools = [self._env_tools[name] for name in self._batch_environments]
            else:
                per_prompt_tools = [self.tools] * len(prompts)
            prompts_text = [
                apply_chat_template(
                    {"prompt": prompt}, self.processing_class, tools=tools, **self.chat_template_kwargs
                )["prompt"]
                for prompt, tools in zip(prompts, per_prompt_tools, strict=True)
            ]
            prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}
        if _unsloth_video_kwargs is not None:
            forward_kwargs = {**forward_kwargs, **_unsloth_video_kwargs}

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
        if (
            "mm_token_type_ids" in forward_kwargs
            or "image_grid_thw" in forward_kwargs
            or "video_grid_thw" in forward_kwargs
        ):
            _mm_token_type_ids = _unsloth_fix_mm_token_type_ids(
                self.processing_class,
                prompt_completion_ids,
                forward_kwargs.get("mm_token_type_ids", None),
                completion_ids = completion_ids,
            )
            if _mm_token_type_ids is not None:
                forward_kwargs["mm_token_type_ids"] = _mm_token_type_ids

        # For VLM tool images: build token type IDs from the full prompt_completion_ids.
        # This must happen AFTER the token_type_ids/mm_token_type_ids extension blocks above,
        # because our version already covers the full sequence (images are in the completion,
        # not just the prompt).
        if self.tools and any(imgs for imgs in tool_images) and self._is_vlm:
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

        # When gradient checkpointing is enabled with use_reentrant=True (non default), calling the model inside a
        # torch.no_grad() block triggers a harmless PyTorch warning ("None of the inputs have requires_grad=True").
        # Temporarily disable checkpointing to avoid this warning during inference.
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            _unsloth_rollout_logps = locals().get('sampling_per_token_logps')
            _unsloth_rollout_old = _unsloth_grpo_rollout_logps_as_old(
                self, _unsloth_rollout_logps, completion_mask, locals().get('tool_mask'),
            )
            if _unsloth_rollout_old:
                old_per_token_logps = None
            elif self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size=batch_size,
                    num_images=num_images,
                    num_tiles=num_tiles,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask, spatial_shapes, image_sizes, image_position_ids
                )
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if False and self.use_vllm and self.vllm_importance_sampling_correction:
                mask = completion_mask if tool_mask is None else completion_mask * tool_mask
                per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask
                # Tokens whose sampling logprob was NaN (unavailable from vLLM) get a zero difference, so their
                # importance ratio is exactly 1 (no correction) rather than propagating NaN through the loss.
                per_token_logps_diff = torch.nan_to_num(per_token_logps_diff, nan=0.0)

                sequence_level_is = self.vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]
                if sequence_level_is:
                    per_sequence_logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
                    logps_diff = per_sequence_logps_diff
                else:
                    logps_diff = per_token_logps_diff

                vllm_importance_sampling_ratio = torch.exp(logps_diff)

                # vllm_importance_sampling_ratio.shape:
                #   token_* modes:     (B, T)  (per-token ratio)
                #   sequence_* modes:  (B, 1)  (per-sequence ratio)

                if self.vllm_importance_sampling_mode in ["sequence_truncate", "token_truncate"]:
                    vllm_importance_sampling_ratio = torch.clamp(
                        vllm_importance_sampling_ratio,
                        min=self.vllm_importance_sampling_clip_min,
                        max=self.vllm_importance_sampling_clip_max,
                    )
                elif self.vllm_importance_sampling_mode in ["sequence_mask", "token_mask"]:
                    min_val = (
                        self.vllm_importance_sampling_clip_min
                        if self.vllm_importance_sampling_clip_min is not None
                        else -math.inf
                    )
                    max_val = (
                        self.vllm_importance_sampling_clip_max
                        if self.vllm_importance_sampling_clip_max is not None
                        else math.inf
                    )

                    invalid_mis_mask = (vllm_importance_sampling_ratio < min_val) | (
                        vllm_importance_sampling_ratio > max_val
                    )
                    vllm_importance_sampling_ratio = vllm_importance_sampling_ratio.masked_fill(
                        invalid_mis_mask, value=0.0
                    )
                else:
                    raise ValueError(
                        f"Unknown vLLM importance sampling level: {self.vllm_importance_sampling_mode}. Possible values are 'token_truncate', 'token_mask', 'sequence_truncate', and 'sequence_mask'."
                    )

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        num_tiles=num_tiles,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask, spatial_shapes, image_sizes, image_position_ids
                    )
                else:
                    # When training a PEFT adapter, how we obtain the reference depends on the setup:
                    # - New adapter: disabling adapters yields the base model.
                    # - Re-training an existing adapter: an initial copy is loaded under the name "ref".
                    model = self.accelerator.unwrap_model(self.model)
                    with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
                        ref_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            num_tiles=num_tiles,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask, spatial_shapes, image_sizes, image_position_ids
                        )
            else:
                ref_per_token_logps = None

        # Merge extra_fields from rollout_func into inputs for reward functions
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        _unsloth_reward_locals = locals()
        if images is not None and 'prompts_text' in _unsloth_reward_locals:
            completions_text = _unsloth_reward_locals.get('completions_text')
            if completions_text is None:
                completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            rewards_per_func = self._calculate_rewards(inputs, prompts_text, completions_text, completion_ids_list)
        else:
            rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        del _unsloth_reward_locals
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        # A completion for which every reward function returned None is unscorable. nansum would collapse it to 0,
        # which both biases the per-group baseline and hands the completion a spurious advantage. Mark these rows NaN
        # so they're excluded from the (nan-aware) baseline below; their advantage is forced to 0 afterwards.
        unscorable_mask = torch.isnan(rewards_per_func).all(dim=1)

        if self.multi_objective_aggregation == "sum_then_normalize":
            # Apply weights to each reward function's output and sum
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            rewards[unscorable_mask] = torch.nan
            mean_grouped_rewards = torch.nanmean(rewards.view(-1, num_generations), dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
            if self.scale_rewards in ["group", "none"]:
                # If self.scale_rewards = "none", we'll only use std_rewards to check for zero std for logging
                if num_generations > 1:
                    std_rewards = nanstd(rewards.view(-1, num_generations), dim=1)
                    std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
                else:  # doesn't occur during training, but could occur in eval when num_generations_eval=1
                    std_rewards = torch.zeros_like(rewards)
            elif self.scale_rewards == "batch":
                # Compute global std
                if rewards.numel() > 1:
                    std_rewards = nanstd(rewards).expand_as(rewards)
                else:  # doesn't occur during training, but could occur in eval when num_generations_eval=batch_size=1
                    std_rewards = torch.zeros_like(rewards)
            else:
                raise ValueError(
                    f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
                )

            advantages = rewards - mean_grouped_rewards
            if self.scale_rewards != "none":
                advantages = advantages / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))  # for logging

        elif self.multi_objective_aggregation == "normalize_then_sum":
            grouped = rewards_per_func.view(-1, num_generations, len(self.reward_funcs))
            mean_k = torch.nanmean(grouped, dim=1, keepdim=True)
            std_k = nanstd(grouped, dim=1, keepdim=True) if num_generations > 1 else torch.zeros_like(mean_k)
            reward_k = (grouped - mean_k) / (std_k + 1e-4)
            reward_k = reward_k.view(-1, len(self.reward_funcs))
            rewards = (reward_k * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            rewards[unscorable_mask] = torch.nan
            std_rewards = nanstd(rewards).expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
            advantages = (rewards - torch.nanmean(rewards)) / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))  # for logging

        else:
            raise ValueError(
                f"Invalid multi_objective_aggregation: {self.multi_objective_aggregation}. Must be "
                "'sum_then_normalize' or 'normalize_then_sum'."
            )

        # Unscorable completions (every reward func returned None) carry no learning signal: their reward is NaN here,
        # so zero their advantage to keep them from moving the policy.
        advantages = torch.nan_to_num(advantages, nan=0.0)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        rewards = (rewards_per_func * self.reward_weights.to(rewards_per_func.device).unsqueeze(0)).nansum(dim=1)
        rewards[unscorable_mask] = torch.nan  # exclude unscorable rows from the logged reward stats
        self._metrics[mode]["reward"].append(torch.nanmean(rewards).item())
        self._metrics[mode]["reward_std"].append(nanstd(rewards).item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())
        _unsloth_sample_weight = _unsloth_grpo_sample_weight(
            self, mode, rewards_per_func, completion_mask, locals().get('tool_mask'), process_slice,
        )

        # Log prompts and completions
        self._logs["prompt"].extend(gather_object(prompts))
        self._logs["completion"].extend(gather_object(completions))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        # Flush user-logged extra columns (from log_extra), gathering across processes.
        # Keys must be sorted so that all ranks call gather_object in the same order, otherwise values
        # get mis-attributed across columns (dict insertion order may differ between processes).
        for column in sorted(set(gather_object(list(self._pending_extra_logs)))):
            values = self._pending_extra_logs.get(column, [None] * len(prompts))
            self._logs["extra"][column].extend(gather_object(values))
        self._pending_extra_logs.clear()

        # Flush user-logged metrics (from log_metric), averaging across processes.
        # Keys must be sorted so that all ranks call accelerator.gather in the same order, otherwise values
        # get mis-attributed across metrics (dict insertion order may differ between processes).
        for name in sorted(set(gather_object(list(self._pending_metrics)))):
            values = self._pending_metrics.get(name, [])
            local_stats = torch.tensor([sum(values), len(values)], dtype=torch.float32, device=device)
            total, count = self.accelerator.gather(local_stats).view(-1, 2).sum(dim=0).tolist()
            self._metrics[mode][name].append(total / count)
        self._pending_metrics.clear()

        if images is not None and self.log_multimodal:
            self._logs["images"].extend(gather_object(images))

        if False and self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            mask = completion_mask.bool() if tool_mask is None else (completion_mask * tool_mask).bool()
            # Tokens vLLM could not score carry NaN, so exclude them rather than let them turn the reported
            # divergence into NaN. Counting them as zero instead would understate the divergence.
            delta = delta[mask & ~torch.isnan(delta)]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )
            if sequence_level_is:
                flat_is_ratio = vllm_importance_sampling_ratio.flatten()
            else:
                flat_is_ratio = vllm_importance_sampling_ratio[mask]

            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if _unsloth_rollout_old:
            output["unsloth_rollout_logps_as_old"] = torch.ones(
                completion_ids.size(0), dtype = torch.bool, device = completion_ids.device
            )
            output["sampling_per_token_logps"] = _unsloth_rollout_logps
        if False and self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = vllm_importance_sampling_ratio
        if sampling_per_token_logps is not None:
            output["sampling_per_token_logps"] = sampling_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
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
        try:
            _unsloth_vision_output = _unsloth_grpo_vision_inputs(forward_kwargs)
        except NameError:
            _unsloth_vision_output = {}
        for _vision_key, _vision_value in _unsloth_vision_output.items():
            if _vision_value is not None and _vision_key not in output:
                output[_vision_key] = _vision_value
        if max_left_pad is not None:
            output["max_left_pad"] = torch.tensor(prompt_ids.shape[0] * [max_left_pad]).unsqueeze(-1)
        try:
            if self.use_vllm and getattr(self, "vllm_importance_sampling_correction", False):
                output["sampling_per_token_logps"] = sampling_per_token_logps
        except NameError:
            output["sampling_per_token_logps"] = None
        if tool_mask is not None:
            output["tool_mask"] = tool_mask
        if _unsloth_sample_weight is not None:
            output["unsloth_sample_weight"] = _unsloth_sample_weight
        return output

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs = False,
        num_items_in_batch = None,
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        _vision_inputs = _unsloth_grpo_vision_inputs(inputs)
        pixel_values = _vision_inputs.get("pixel_values", None)
        image_grid_thw = _vision_inputs.get("image_grid_thw", None)
        pixel_attention_mask = _vision_inputs.get("pixel_attention_mask", None)
        image_sizes = _vision_inputs.get("image_sizes", None)
        num_images = _vision_inputs.get("num_images", None)
        # Transformers 5.x needs token_type_ids/mm_token_type_ids for some vision models.
        token_type_ids = _vision_inputs.get("token_type_ids", None)
        mm_token_type_ids = _vision_inputs.get("mm_token_type_ids", None)
        video_grid_thw = _vision_inputs.get("video_grid_thw", None)
        num_items_in_batch = inputs.get("num_items_in_batch", None)
        sampling_per_token_logps = inputs.get("sampling_per_token_logps", None)
        tool_mask = inputs.get("tool_mask", None)
        _rollout_logps_as_old = inputs.get("unsloth_rollout_logps_as_old", None) is not None
        _loss_delta = getattr(self.args, "delta", None)
        if _rollout_logps_as_old and _loss_delta is None:
            # verl's dual-clip default (clip_ratio_c = 3.0).
            _loss_delta = 3.0
        current_gradient_accumulation_steps = _unsloth_grpo_accumulation_steps(self)
        num_processes = self.accelerator.num_processes

        input_ids = torch.cat([prompt_ids, completion_ids], dim = 1)
        bsz, qlen = input_ids.shape
        attention_mask = torch.cat([prompt_mask, completion_mask], dim = 1)
        if (
            mm_token_type_ids is not None
            or image_grid_thw is not None
            or video_grid_thw is not None
        ):
            mm_token_type_ids = _unsloth_fix_mm_token_type_ids(
                self.processing_class,
                input_ids,
                mm_token_type_ids,
                completion_ids = completion_ids,
            )
            _vision_inputs["mm_token_type_ids"] = mm_token_type_ids
        # Only the keys the processor produced: an older zoo must not see unknown kwargs.
        _vision_inputs = {k: v for k, v in _vision_inputs.items() if v is not None}
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        _input_ids = input_ids
        _logits_to_keep = logits_to_keep

        get_logps_func = (
            lambda model,
            input_ids,
            attention_mask,
            logits_to_keep,
            batch_size = None,
            compute_entropy = False,
            compute_efficient = False: (
                self._get_per_token_logps(
                    model, input_ids, attention_mask, logits_to_keep, compute_efficient
                )
                if hasattr(self, "_get_per_token_logps")
                else self._get_per_token_logps_and_entropies(
                    model,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    compute_entropy,
                    compute_efficient,
                )[0]
            )
        )  # logps

        per_token_logps = get_logps_func(
            model, input_ids, attention_mask, logits_to_keep, compute_efficient = True
        )
        # KL divergence between model and reference: _prepare_inputs no longer returns reference log probs. See trl grpo_trainer.py#L1328.
        ref_logps = inputs.get("ref_per_token_logps", None)
        # x - x.detach() preserves gradients from x.
        advantages = inputs["advantages"]
        old_logps = inputs.get("old_per_token_logps", None)

        input_ids = input_ids[:, -logits_to_keep:]

        model_config = _unsloth_get_model_config(model)
        # The old and reference logps come from _get_per_token_logps_and_entropies and the gradient logps from here, so both must read the transforms alike or the importance ratio compares two different policies.
        if detect_logit_transforms is not None:
            # model_config, not model: see _get_per_token_logps_and_entropies.
            _transforms = detect_logit_transforms(model_config)
            logit_softcapping = _transforms["logit_softcapping"]
            logit_scale_multiply = _transforms["logit_scale_multiply"]
            logit_scale_divide = _transforms["logit_scale_divide"]
        else:
            logit_softcapping = _unsloth_get_final_logit_softcapping(model)  # Gemma
            logit_scale_multiply, logit_scale_divide = _unsloth_resolve_logit_scales(model_config)

        max_left_pad = inputs.get("max_left_pad", 0)
        # importance_sampling_level is absent before TRL 0.20.0 (token level).
        importance_sampling_level = getattr(
            self,
            "importance_sampling_level",
            getattr(self.args, "importance_sampling_level", "token"),
        )
        if per_token_logps is not None:
            loss_mask = completion_mask
            if tool_mask is not None:
                if tool_mask.shape != completion_mask.shape:
                    raise ValueError(
                        "tool_mask/env_mask must have the same shape as completion_mask"
                    )
                loss_mask = completion_mask * tool_mask.to(
                    device = completion_mask.device,
                    dtype = completion_mask.dtype,
                )
            if _rollout_logps_as_old:
                old_logps, sampling_per_token_logps = sampling_per_token_logps, None
            (
                loss,
                completion_length,
                mean_kl,
                delta,
                flat_is_ratio,
                coef_1,
                completion_mask,
            ) = grpo_compute_loss_slow(
                ref_logps,
                per_token_logps,
                old_logps,
                sampling_per_token_logps,
                input_ids,
                loss_mask,
                self.beta,
                advantages,
                pixel_values = pixel_values,
                image_grid_thw = image_grid_thw,
                loss_type = self.args.loss_type,
                importance_sampling_level = importance_sampling_level,
                epsilon_low = self.epsilon_low,
                epsilon_high = self.epsilon_high,
                max_completion_length = self.args.max_completion_length,
                delta = _loss_delta,
                temperature = self.args.temperature,
                max_left_pad = max_left_pad,
                logit_softcapping = logit_softcapping,
                logit_scale_multiply = logit_scale_multiply,
                logit_scale_divide = logit_scale_divide,
                num_items_in_batch = num_items_in_batch,
                current_gradient_accumulation_steps = current_gradient_accumulation_steps,
                num_processes = num_processes,
            )
        else:
            # With beta = 0 and num_iterations = 1 the no-grad path (and its zoo gate) never runs, and an older
            # grpo_accumulated_loss silently drops pixel_values. Body-local: copied out without imports (#6960).
            if pixel_values is not None and not getattr(
                self, "_unsloth_grpo_vision_zoo_checked", False
            ):
                _grpo_vision_chunks = None
                try:
                    from unsloth_zoo.rl_replacements import (
                        grpo_vision_chunks as _grpo_vision_chunks,
                    )
                except Exception:
                    pass
                if _grpo_vision_chunks is None:
                    raise RuntimeError(
                        "Unsloth: vision GRPO needs an unsloth_zoo build that exports "
                        "grpo_vision_chunks, the shared multimodal key tuple and chunker "
                        "used by both GRPO logprob paths. Please upgrade unsloth_zoo to "
                        "2026.9.5 or newer: pip install -U unsloth_zoo"
                    )
                self._unsloth_grpo_vision_zoo_checked = True

            def _unsloth_requires_multi_image_zoo(value):
                if value is None:
                    return False
                if isinstance(value, torch.Tensor):
                    counts = value.detach().cpu().reshape(-1).tolist()
                else:
                    counts = list(value)
                return any(int(n) != 1 for n in counts)

            if _unsloth_requires_multi_image_zoo(num_images) and not getattr(
                self, "_unsloth_grpo_zoo_checked", False
            ):
                _supports_num_images = (
                    "num_images" in inspect.signature(grpo_accumulated_loss).parameters
                )
                if not _supports_num_images:
                    # Probe by import: the grep below cries "upgrade unsloth_zoo" falsely.
                    try:
                        from unsloth_zoo.rl_replacements import grpo_vision_chunks
                        _supports_num_images = grpo_vision_chunks is not None
                    except Exception:
                        pass
                if not _supports_num_images:
                    try:
                        _zoo_src = inspect.getsource(grpo_accumulated_loss)
                    except (TypeError, OSError):
                        _zoo_src = ""
                    _supports_num_images = "num_images" in _zoo_src
                if not _supports_num_images:
                    raise RuntimeError(
                        "Multi-image GRPO requires an unsloth_zoo build whose "
                        "grpo_accumulated_loss handles num_images. Please upgrade "
                        "unsloth_zoo (see https://github.com/unslothai/unsloth-zoo/pull/613)."
                    )
                self._unsloth_grpo_zoo_checked = True
            if tool_mask is not None and not getattr(
                self, "_unsloth_grpo_tool_mask_zoo_checked", False
            ):
                _supports_tool_mask = (
                    "tool_mask" in inspect.signature(grpo_accumulated_loss).parameters
                )
                if not _supports_tool_mask:
                    try:
                        _zoo_src = inspect.getsource(grpo_accumulated_loss)
                    except (TypeError, OSError):
                        _zoo_src = ""
                    _supports_tool_mask = "tool_mask" in _zoo_src
                if not _supports_tool_mask:
                    raise RuntimeError(
                        "env_mask/tool_mask GRPO requires an unsloth_zoo build whose "
                        "grpo_accumulated_loss handles tool_mask. Please upgrade "
                        "unsloth_zoo."
                    )
                self._unsloth_grpo_tool_mask_zoo_checked = True
            _grpo_accumulated_loss_kwargs = {}
            _sample_weight = inputs.get("unsloth_sample_weight", None)
            if _sample_weight is not None:
                if "sample_weight" not in inspect.getsource(grpo_compute_loss):
                    raise RuntimeError(
                        "UNSLOTH_GRPO_ZERO_VARIANCE_EXCLUDE / UNSLOTH_GRPO_LOSS_AGGREGATION need a "
                        "newer unsloth_zoo. Please upgrade unsloth_zoo."
                    )
                _grpo_accumulated_loss_kwargs["sample_weight"] = _sample_weight
            if tool_mask is not None:
                _grpo_accumulated_loss_kwargs["tool_mask"] = tool_mask
            if _rollout_logps_as_old:
                _grpo_accumulated_loss_kwargs["rollout_logps_as_old"] = True
            if hasattr(self.args, "loss_type"):
                (
                    loss,
                    completion_length,
                    mean_kl,
                    delta,
                    flat_is_ratio,
                    coef_1,
                    completion_mask,
                ) = grpo_accumulated_loss(
                    trainer = self,
                    input_ids = _input_ids,
                    logits_to_keep = logits_to_keep,
                    completion_mask = completion_mask,
                    advantages = advantages,
                    old_logps = old_logps,
                    ref_logps = ref_logps,
                    n_chunks = self.args.unsloth_num_chunks,
                    loss_type = self.args.loss_type,
                    importance_sampling_level = importance_sampling_level,
                    epsilon_low = self.epsilon_low,
                    epsilon_high = self.epsilon_high,
                    max_completion_length = self.args.max_completion_length,
                    delta = _loss_delta,
                    temperature = self.args.temperature,
                    max_left_pad = max_left_pad,
                    logit_softcapping = logit_softcapping,
                    logit_scale_multiply = logit_scale_multiply,
                    logit_scale_divide = logit_scale_divide,
                    attention_mask = attention_mask,
                    num_items_in_batch = num_items_in_batch,
                    current_gradient_accumulation_steps = current_gradient_accumulation_steps,
                    num_processes = num_processes,
                    sampling_per_token_logps = sampling_per_token_logps,
                    **_vision_inputs,
                    **_grpo_accumulated_loss_kwargs,
                )
            else:
                # For backwards compatibility with trl 0.15.2 and maybe 0.17.
                loss, completion_length, mean_kl, coef_1, completion_mask = grpo_accumulated_loss(
                    trainer = self,
                    input_ids = _input_ids,
                    logits_to_keep = logits_to_keep,
                    completion_mask = completion_mask,
                    advantages = advantages,
                    old_logps = old_logps,
                    ref_logps = ref_logps,
                    n_chunks = self.args.unsloth_num_chunks,
                    temperature = self.args.temperature,
                    logit_softcapping = logit_softcapping,
                    logit_scale_multiply = logit_scale_multiply,
                    logit_scale_divide = logit_scale_divide,
                    attention_mask = attention_mask,
                    **_vision_inputs,
                    **_grpo_accumulated_loss_kwargs,
                )
        if "train" in self._metrics:
            # model.training like TRL; should_evaluate is False under eval_on_start and evaluate().
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["completion_length"].append(completion_length.item())
            self._metrics[mode]["kl"].append(mean_kl.item())
        else:
            self._metrics["completion_length"].append(completion_length.item())
            self._metrics["kl"].append(mean_kl.item())

        if delta is not None and (
            (self.use_vllm and getattr(self, "vllm_importance_sampling_correction", False))
            or _rollout_logps_as_old
        ):
            delta, flat_is_ratio = _unsloth_grpo_is_metric_values(
                delta,
                flat_is_ratio,
                completion_mask,
                (not _rollout_logps_as_old)
                and _unsloth_grpo_is_sequence_level(
                    getattr(self.args, "vllm_importance_sampling_mode", None)
                ),
            )
            mean_delta = (
                torch.mean(delta)
                if delta.numel() > 0
                else torch.tensor(0.0, device = self.model.device)
            )
            max_delta = (
                torch.max(delta)
                if delta.numel() > 0
                else torch.tensor(0.0, device = self.model.device)
            )
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device = self.model.device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device = self.model.device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device = self.model.device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                self.accelerator.gather(min_importance_sampling_ratio)
                .nan_to_num(nan = float("inf"))
                .min()
                .item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                self.accelerator.gather(max_importance_sampling_ratio)
                .nan_to_num(nan = float("-inf"))
                .max()
                .item()
            )

        completion_token_count = completion_mask.sum().clamp(min = 1.0)

        def masked_batch_mean(x):
            # a width-1 tensor is sequence-level only when the completion mask spans multiple tokens
            if x.shape[1] == 1 and completion_mask.shape[1] != 1:
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(
                gathered_clip_ratio.nanmean().item()
            )
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather(cispo_clip_ratio)
            self._metrics[mode]["cispo_clip_ratio"].append(
                gathered_cispo_clip_ratio.nanmean().item()
            )

        return loss

    @staticmethod
    def get_off_policy_mask(
        advantages: torch.Tensor,
        per_token_logps: torch.Tensor,
        sampling_per_token_logps: torch.Tensor,
        mask: torch.Tensor,
        off_policy_threshold: float,
    ) -> torch.Tensor:
        """
        Computes the Off-Policy Sequence Mask from DeepSeek-V3.2 paper. Returns a (B, 1) tensor where 1.0 indicates
        "Keep" and 0.0 indicates "Drop".
        """
        # forward KL div: log(pi_old) - log(pi_theta)
        kl_div = sampling_per_token_logps - per_token_logps.detach()
        # Tokens vLLM could not score carry NaN and provide no KL evidence. Left in, the sequence mean is NaN,
        # `avg_seq_kl <= off_policy_threshold` is then False, and a negative-advantage sequence would be dropped
        # on the strength of a single unscorable token.
        kl_div = torch.nan_to_num(kl_div, nan=0.0)
        # Sequence-level Mean KL (ignoring prompt+padding)
        seq_kl_sum = (kl_div * mask).sum(dim=1, keepdim=True)
        avg_seq_kl = seq_kl_sum / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        # Keep if (Advantage >= 0) OR (KL <= delta)
        is_pos_adv = advantages >= 0
        is_low_kl = avg_seq_kl <= off_policy_threshold
        return (is_pos_adv | is_low_kl).to(dtype=mask.dtype)  # (B, 1)

    @staticmethod
    @torch.no_grad()
    def get_gamma_weights(
        advantages: torch.Tensor,
        log_ratio_per_token: torch.Tensor,
        mask: torch.Tensor,
        importance_sampling_ratio: torch.Tensor | None,  # (B, T)
        k_pos: float = 2.0,
        lambda_pos: float = 3.0,
        k_neg: float = 3.0,
        lambda_neg: float = 2.0,
    ) -> torch.Tensor:
        """
        Computes the Gamma weights for the VESPO loss. For reference:
            φ(w) = e^λ × w^k × e^{-λw} is the gamma weighting (normalized so φ(1)=1)
                with w = sequence-level importance sampling ratio
        note: we will compute φ(w) in log space

        φ(w) is detached via @torch.no_grad(), only acts as gradient scaling coefficient

        VESPO loss = -φ(w) × A × log_prob, gradient naturally gives φ(w) × A × ∇log π
        """
        # reducing clamp range directly to log(1e-8) ~ -18.42, to avoid recomputing log_w=log(w.clamp(min=1e-8)) later
        # This is solely for matching truthfully the original implementation, otherwise keeping -20 could be fine.
        lower_clamp = math.log(1e-8)

        # Sequence-level log ratio Σ log(π_θ/π_old) (not a mean like for `log_importance_weights`)
        log_ratio_clamped = torch.clamp(log_ratio_per_token, -20.0, 20.0)
        seq_log_ratio = torch.sum(log_ratio_clamped * mask, dim=-1, keepdim=True)  # (B, 1)

        # Apply token-level TIS or MIS correction (in log space)
        if importance_sampling_ratio is not None:
            log_is_ratio = torch.clamp(torch.log(importance_sampling_ratio), lower_clamp, 20.0)
            # log(w) = log(π_θ/π_old) + log(π_old/π_sampler)
            seq_log_ratio += torch.sum(log_is_ratio, dim=-1, keepdim=True)

        log_w_seq = torch.clamp(seq_log_ratio, lower_clamp, 20.0)
        w_seq = torch.exp(log_w_seq)

        # compute k and lambda based on advantage sign
        is_nonneg_adv = advantages >= 0
        k_seq = torch.where(is_nonneg_adv, k_pos, k_neg)
        lambda_seq = torch.where(is_nonneg_adv, lambda_pos, lambda_neg).clamp(min=1e-4)

        # log(φ(w)) = λ + k × log(w) - λ × w
        log_phi = lambda_seq + k_seq * log_w_seq - lambda_seq * w_seq
        phi_seq = torch.exp(log_phi).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        return phi_seq  # (B, 1)

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies, aux_loss = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            compute_aux_loss=self.aux_loss_enabled,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            spatial_shapes=inputs.get("spatial_shapes"),
            num_tiles=inputs.get("num_tiles"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
            image_position_ids=inputs.get("image_position_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the loss
        advantages = inputs["advantages"]
        # In the base GRPO implementation, advantages are expected to have shape (B,). To support subclasses that
        # provide advantages with shape (B, T), we *conditionally* unsqueeze the tensor.
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
        # old_per_token_logps == per_token_logps. In this case we can skip its computation
        # (see _generate_and_score_completions) and instead use per_token_logps.detach().
        # The exception is when using vLLM, where we always compute old_per_token_logps
        # for importance sampling
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        if self.off_policy_mask_threshold is not None:
            # OPSM should use inference-time logprobs to detect both sources of off-policyness:
            # 1. Drift from gradient updates (always present)
            # 2. Drift from training-inference mismatch (when using vLLM)
            # When using vLLM, prioritize sampling_per_token_logps, otherwise use old_per_token_logps
            sampling_per_token_logps = inputs.get("sampling_per_token_logps", old_per_token_logps)

            off_policy_mask = self.get_off_policy_mask(
                advantages=advantages,
                per_token_logps=per_token_logps,
                sampling_per_token_logps=sampling_per_token_logps,
                mask=mask,
                off_policy_threshold=self.off_policy_mask_threshold,
            )

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            # Importance sampling correction for the KL divergence
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1

        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)
        if self.loss_type == "cispo":
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            # Two-sided clipping
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        elif self.loss_type == "sapo":
            temperatures = torch.where(advantages > 0, self.args.sapo_temperature_pos, self.args.sapo_temperature_neg)
            soft_coef_1 = torch.sigmoid(temperatures * (coef_1 - 1)) * 4 / temperatures
            per_token_loss = -soft_coef_1 * advantages
        elif self.loss_type == "vespo":
            phi_seq = self.get_gamma_weights(
                advantages=advantages,
                log_ratio_per_token=log_ratio,
                mask=mask,
                importance_sampling_ratio=inputs.get("importance_sampling_ratio"),
                k_pos=self.args.vespo_k_pos,
                lambda_pos=self.args.vespo_lambda_pos,
                k_neg=self.args.vespo_k_neg,
                lambda_neg=self.args.vespo_lambda_neg,
            )
            per_token_loss = -phi_seq * advantages * per_token_logps
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.off_policy_mask_threshold is not None:
            per_token_loss = per_token_loss * off_policy_mask

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction and self.loss_type != "vespo":
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        mode = "train" if self.model.training else "eval"
        if self.loss_type in ["grpo", "sapo"]:
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
            policy_loss = loss.detach()
            loss = loss / normalizer
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
            policy_loss = loss.detach()
            loss = loss / normalizer
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
            policy_loss = loss.detach()
            loss = loss / normalizer
        elif self.loss_type in ["cispo", "dapo", "vespo"]:
            # `num_items_in_batch` spans the generation batch, so rescale it to one accumulation window
            normalizer = inputs["num_items_in_batch"].clamp(min=1.0) / self.accelerator.num_processes
            if mode == "train":  # in eval, the batch is neither split across steps nor accumulated
                normalizer = normalizer * self.current_gradient_accumulation_steps / self.args.steps_per_generation
            loss = (per_token_loss * mask).sum() / normalizer
            policy_loss = loss.detach()
        elif self.loss_type == "luspo":
            # `per_token_loss` is (B, 1) only in the recommended sequence-level setup; importance_sampling_level=
            # "token" (the config default), the KL term, token-level vLLM IS ratios, and the entropy mask all
            # broadcast it to (B, T), so mask before aggregating.
            loss = (per_token_loss * mask).sum(-1).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            policy_loss = loss.detach()
            loss = loss / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Entropy bonus: add entropy regularization to encourage exploration. _entropy_bonus_enabled is set
        # whenever a non-zero static coef is set OR adaptive mode is enabled (adaptive stays enabled even when
        # entropy_coef has been decremented to entropy_coef_min so it can recover once entropy drops again).
        if self._entropy_bonus_enabled:
            # When top_entropy_quantile < 1.0, entropy_mask restricts policy gradients to high-entropy
            # tokens. Use the same effective mask for the entropy bonus so it acts on the same tokens.
            effective_mask = mask if entropy_mask is None else mask * entropy_mask
            # Entropy bonus = mean per-token entropy H (the documented objective L = L_policy - coef * H), so
            # H does not depend on how each loss type normalizes its policy term. The bonus is a mean over the
            # tokens it acts on (effective_mask), scaled only for gradient accumulation, never by a loss-type-
            # specific policy normalizer. (The adaptive controller below tracks a window-global token-weighted mean,
            # which can differ from this per-micro-batch mean when token counts vary across micro-batches.)
            accumulation_factor = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            entropy_loss = (
                (entropies * effective_mask).sum() / effective_mask.sum().clamp(min=1.0) / accumulation_factor
            )

            # Apply the coefficient and gating from the end of the previous optimizer step, so that every
            # micro-batch in the current accumulation window applies the same entropy bonus. The adaptive
            # update below only takes effect on the next step.
            if self.use_adaptive_entropy:
                apply_coef = self.entropy_coef if self._last_world_entropy <= self.args.entropy_target else 0.0
            else:
                apply_coef = self.entropy_coef

            loss = loss - apply_coef * entropy_loss

            self._metrics[mode]["policy_loss"].append(self.accelerator.gather(policy_loss).nanmean().item())

            # Adaptive update. Gated on train mode so evaluation cannot mutate the entropy controller state.
            if self.use_adaptive_entropy and mode == "train":
                # Accumulate the entropy sum and active-token count of every micro-batch into a running window
                # buffer, so the controller measures the exact window-global entropy rather than just the last
                # micro-batch (which would be a 1 / gradient_accumulation_steps subsample).
                stats = torch.stack([(entropies * effective_mask).sum(), effective_mask.sum()]).detach()
                if self._entropy_window_stats is None:
                    self._entropy_window_stats = stats
                else:
                    self._entropy_window_stats = self._entropy_window_stats + stats
                # At the optimizer-step boundary, reduce the window totals across ranks (sum and token count
                # jointly, for a true global mean unbiased when ranks have different completion lengths),
                # update the coefficient for the next step, then reset the window buffer.
                if self.accelerator.sync_gradients:
                    window_stats = self.accelerator.reduce(self._entropy_window_stats, reduction="sum")
                    world_entropy = (window_stats[0] / window_stats[1].clamp(min=1.0)).item()
                    if world_entropy <= self.args.entropy_target:
                        self.entropy_coef = min(
                            self.entropy_coef + self.args.entropy_coef_delta, self.args.entropy_coef_max
                        )
                    else:
                        self.entropy_coef = max(
                            self.entropy_coef - self.args.entropy_coef_delta, self.args.entropy_coef_min
                        )
                    self._last_world_entropy = world_entropy
                    self._entropy_window_stats = None

            # Log entropy_coef on train optimizer-step boundaries (constant for static control; updated just
            # above for adaptive control). sync_gradients is always True in eval (no accumulation context).
            if mode == "train" and self.accelerator.sync_gradients:
                self._metrics[mode]["entropy_coef"].append(self.entropy_coef)

        # The policy loss above is scaled for gradient accumulation (HF auto-scaling is off here), so scale aux too
        if self.aux_loss_enabled:
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss + self.router_aux_loss_coef * aux_loss / normalizer
            self._metrics[mode]["aux_loss"].append(self.accelerator.gather_for_metrics(aux_loss).mean().item())

        # Log the metrics
        def masked_seq_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence": already one value per sequence
                return x.squeeze(1)
            return (x * mask).sum(-1) / mask.sum(-1)

        def global_masked_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence": one value per sequence
                local_sum, local_count = x.sum(), torch.tensor(float(x.shape[0]), device=x.device)
            else:
                local_sum, local_count = (x * mask).sum(), mask.sum().float()
            totals = self.accelerator.reduce(torch.stack([local_sum, local_count]), reduction="sum")
            return (totals[0] / totals[1].clamp(min=1.0)).item()

        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(global_masked_mean(per_token_kl))

        self._metrics[mode]["entropy"].append(global_masked_mean(entropies))

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            # Compute the clipped probability ratios
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped
            self._metrics[mode]["clip_ratio/low_mean"].append(global_masked_mean(is_low_clipped.float()))
            self._metrics[mode]["clip_ratio/high_mean"].append(global_masked_mean(is_high_clipped.float()))
            self._metrics[mode]["clip_ratio/region_mean"].append(global_masked_mean(is_region_clipped.float()))
            gathered_low_clip = self.accelerator.gather(masked_seq_mean(is_low_clipped.float()))
            self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather(masked_seq_mean(is_high_clipped.float()))
            self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            self._metrics[mode]["cispo_clip_ratio"].append(global_masked_mean(is_cispo_clipped.float()))
        elif self.loss_type == "vespo":
            self._metrics[mode]["vespo/phi_seq_mean"].append(global_masked_mean(phi_seq))

        return loss

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
            # Filter out NaN values before averaging. A reward function that returns None for all samples
            # in a batch produces NaN for that batch's metric. With logging_steps > 1, a naive sum()/len()
            # would let a single NaN contaminate valid data from other batches. Only return None when no
            # valid values remain (e.g. JSON loggers crash on float NaN).
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
                    self._logs["rewards"],
                    self._logs["advantages"],
                    self.state.global_step,
                    self.num_completions_to_print,
                    extra=dict(self._logs["extra"]),
                )

            logging_backends = []
            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                logging_backends.append(wandb)
            if self.args.report_to and "trackio" in self.args.report_to:
                logging_backends.append(trackio)

            table = {
                "step": [self.state.global_step] * len(self._logs["prompt"]),
                "prompt": [strip_images_from_messages(messages) for messages in self._logs["prompt"]],
                "completion": [strip_images_from_messages(messages) for messages in self._logs["completion"]],
                **self._logs["rewards"],
                **self._logs["extra"],
                "advantage": self._logs["advantages"],
            }

            df_base = pd.DataFrame(table)
            df_base.to_parquet(
                os.path.join(
                    self.args.output_dir,
                    "completions",
                    f"completions_{self.state.global_step:05d}.parquet",
                )
            )

            images_raw = self._logs["images"] or []

            for logging_backend in logging_backends:
                if images_raw:
                    images = []
                    for image_list in self._logs["images"]:
                        if image_list:
                            images.append([logging_backend.Image(image) for image in image_list])
                        else:
                            images.append([])
                    df = pd.concat(
                        [df_base, pd.Series(images, name="image")],
                        axis=1,
                        copy=False,
                    )
                else:
                    df = df_base

                if self.log_unique_prompts:
                    # Prompts are conversations (lists of dicts), so compare them by their text
                    df = df[~df["prompt"].astype(str).duplicated()]

                logging_backend.log({"completions": logging_backend.Table(dataframe=df)})

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
        if self.use_adaptive_entropy and self.args.should_save:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            output_dir = os.path.join(self._get_output_dir(trial=trial), checkpoint_folder)
            with open(os.path.join(output_dir, "entropy_ctrl_state.json"), "w") as f:
                json.dump({"entropy_coef": self.entropy_coef, "last_world_entropy": self._last_world_entropy}, f)

    def _load_optimizer_and_scheduler(self, checkpoint):
        super()._load_optimizer_and_scheduler(checkpoint)
        if self.use_adaptive_entropy and checkpoint is not None:
            path = os.path.join(checkpoint, "entropy_ctrl_state.json")
            if os.path.exists(path):
                with open(path) as f:
                    state = json.load(f)
                self.entropy_coef = state["entropy_coef"]
                self._last_world_entropy = state["last_world_entropy"]
class UnslothGRPOTrainer(_UnslothGRPOTrainer):
    """
    
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
    Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    >>> from trl import GRPOTrainer
    >>> from trl.rewards import accuracy_reward
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    >>> trainer = GRPOTrainer(
    ...     model="Qwen/Qwen2.5-0.5B-Instruct",
    ...     reward_funcs=accuracy_reward,
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
        reward_funcs (`RewardFunc | list[RewardFunc]`, *optional*):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. May be omitted when the reward is supplied
            by the environment through `environment_factory` (see below). Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. Custom reward
                   functions can be either synchronous or asynchronous and can also return `None` when the reward is
                   not applicable to those samples. This is useful for multi-task training where different reward
                   functions apply to different types of samples. When a reward function returns `None` for a sample,
                   that reward function is excluded from the reward calculation for that sample. For more details, see
                   [Using a custom reward
                  function](#using-a-custom-reward-function).

                  The trainer's state is also passed to the reward function. The trainer's state is an instance of
                  [`~transformers.TrainerState`] and can be accessed by accessing the `trainer_state` argument to the
                  reward function's signature.
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`], *optional*):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).

            May be omitted only when an `environment_factory` is provided and the environment owns (or procedurally
            generates) the data, returning the prompt from its `reset()` method. In that case, `max_steps` must be set
            to define the training length.

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
        reward_processing_classes ([`~transformers.PreTrainedTokenizerBase`] or `list[PreTrainedTokenizerBase]`, *optional*):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using
            [`~transformers.AutoTokenizer.from_pretrained`]. For elements in `reward_funcs` that are custom reward
            functions (not [`~transformers.PreTrainedModel`]), the corresponding entries in `reward_processing_classes`
            are ignored.
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
            A list of callable tool functions (sync or async) that the model can invoke during generation. Each tool
            should be a standard Python function with properly type-hinted arguments and return values, and a
            Google-style docstring describing its purpose, arguments, and return value. For more details, see:
            https://huggingface.co/docs/transformers/en/chat_extras#passing-tools. The model uses the function's name,
            type hints, and docstring to determine how to call it. Ensure that the model's chat template supports tool
            use and that it has been fine-tuned for tool calling.
        rollout_func (`RolloutFunc`, *optional*):
            Function to use for generating completions. It receives the list of prompts allocated to the current
            process and the trainer instance. It must return a dict with `"prompt_ids"`, `"completion_ids"`, and
            `"logprobs"` fields, and can optionally return `"logprob_token_ids"` (same shape as `"logprobs"`). Any
            other fields are forwarded to the reward functions. The function receives the raw per-process prompt slice
            with no duplication; it is responsible for returning the correct number of completions per prompt (see
            `num_generations` / `num_generations_eval` on the trainer). This feature is experimental and may change or
            be removed at any time without prior notice.
        environment_factory (`EnvironmentFactory` or `dict[str, EnvironmentFactory]`, *optional*):
            A callable that creates and returns an environment instance, or a dictionary mapping environment names to
            such callables. The environment class should define methods that can be invoked as tools during generation.
            Each method should comply with the same requirements as the `tools` described above. The environment must
            also implement a callable `reset` method that can be used to reset state between generations. The `reset`
            method should return either `None` or a string: when it returns a string, that string is appended to the
            last user message before generation. The environment may also define a `get_reward` method taking no
            argument and returning a `float`: when present, the environment owns the reward, and `get_reward` is called
            once per completed rollout to score it from the environment's internal state. It acts as an additional
            reward source (with weight 1, logged under the environment's class name) alongside `reward_funcs`, which
            then becomes optional.

            With a single callable, every example uses the same environment, with one instance per rollout so their
            interactions stay isolated. With a dictionary, each example must carry an `environment` field selecting its
            environment by name, and only that environment's tools are exposed in its prompt — letting a single run mix
            tasks (e.g. a coding environment and a game). This feature is experimental and may change or be removed at
            any time without prior notice.
    
    """
    def __init__(
        self,
        model,
        reward_funcs = None,
        args = None,
        train_dataset = None,
        eval_dataset = None,
        processing_class = None,
        reward_processing_classes = None,
        callbacks = None,
        quantization_config = None,
        peft_config = None,
        tools = None,
        rollout_func = None,
        environment_factory = None,
        **kwargs
    ):
        if args is None: args = UnslothGRPOConfig()
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
        if not isinstance(reward_funcs, list): _reward_funcs = [reward_funcs]
        else: _reward_funcs = reward_funcs
        for reward_func in _reward_funcs:
            try:
                reward_func_name = reward_func.__name__
                if True:
                    other_metrics.append(f'rewards/{reward_func_name}/mean')
                if True:
                    other_metrics.append(f'rewards/{reward_func_name}/std')
                if False:
                    other_metrics.append(f'rewards/{reward_func_name}')
            except: pass
        
        from unsloth_zoo.logging_utils import PatchRLStatistics
        PatchRLStatistics('grpo_trainer', other_metrics)
        
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
            reward_funcs = reward_funcs,
            args = args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            processing_class = processing_class,
            reward_processing_classes = reward_processing_classes,
            callbacks = callbacks,
            quantization_config = quantization_config,
            peft_config = peft_config,
            tools = tools,
            rollout_func = rollout_func,
            environment_factory = environment_factory,**kwargs)
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
        if getattr(self, 'vllm_generation', None) is not None:
            self.vllm_generation._unsloth_vllm_sampling_params = getattr(getattr(self, 'args', None), 'vllm_sampling_params', None)
        pass
        if os.environ.get('UNSLOTH_GRPO_ROUTING_REPLAY', '0') == '1' and hasattr(model, 'vllm_engine'):
            try:
                from unsloth_zoo.routing_replay import install_routing_replay
            except ImportError:
                print('Unsloth: UNSLOTH_GRPO_ROUTING_REPLAY=1 needs a newer unsloth_zoo: pip install -U unsloth_zoo')
            else:
                from unsloth.models._routing_replay import routing_replay_hook_snapshot, prepare_routing_replay
                # TRL scores the PEFT reference through self.model.disable_adapter(), so prepare that object.
                _unsloth_replay_model = getattr(self, 'model', None)
                _unsloth_replay_model = model if _unsloth_replay_model is None else _unsloth_replay_model
                _unsloth_replay_hooks_before = routing_replay_hook_snapshot(_unsloth_replay_model)
                install_routing_replay(model, model.vllm_engine, getattr(self, 'processing_class', None))
                # Pick the router weight formulas outside compilation; PEFT reference scoring routes naturally.
                prepare_routing_replay(_unsloth_replay_model, _unsloth_replay_hooks_before)
        pass
        
pass


if hasattr(logger, "addFilter"):
    import logging
    class HideLoggingMessage(logging.Filter):
        def __init__(self, text): self.text = text
        def filter(self, x): return not (self.text in x.getMessage())
    pass
    logger.addFilter(HideLoggingMessage("`use_cache=True`"))

