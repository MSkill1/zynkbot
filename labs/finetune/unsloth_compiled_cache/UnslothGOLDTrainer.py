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
from trl.experimental.gold.gold_trainer import (Any, AutoConfig, AutoProcessor, AutoTokenizer, BaseImageProcessor, Callable, DataCollator, DataCollatorForChatML, DataCollatorForVisionLanguageChatML, DataLoader, Dataset, DistributedType, EvalPrediction, F, FeatureExtractionMixin, GOLDConfig, GOLDTrainer, GenerationConfig, IterableDataset, Optional, PeftConfig, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, RepeatSampler, SFTTrainer, Table, TrainerCallback, ULDLoss, VLLMGeneration, Version, broadcast_object_list, build_teacher_inputs_from_texts, create_model_from_path, defaultdict, deque, disable_dropout_in_model, dist, empty_cache, encode_with_byte_offsets, gather_object, get_config_model_id, global_then_local_main_first, identity, is_conversational, is_datasets_available, is_rich_available, is_vllm_available, maybe_convert_to_chatml, nn, pack_dataset, pad, pad_byte_offsets, partial, piece_byte_len, prepare_deepspeed, prepare_multimodal_messages, print_prompt_completions_sample_uld, profiling_decorator, random, seed_worker, split_tensor_dict, takewhile, textwrap, torch, transformers, unwrap_model_for_generation, warnings, AutoConfig, AutoProcessor, AutoTokenizer, BaseImageProcessor, Callable, DataCollator, DataCollatorForChatML, DataCollatorForVisionLanguageChatML, Dataset, EvalPrediction, F, FeatureExtractionMixin, GOLDConfig, GenerationConfig, Optional, PeftConfig, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, SFTTrainer, TrainerCallback, ULDLoss, VLLMGeneration, create_model_from_path, defaultdict, deque, disable_dropout_in_model, dist, get_config_model_id, identity, is_vllm_available, nn, pad, prepare_deepspeed, torch, warnings)


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
class UnslothGOLDConfig(GOLDConfig):
    """
    
    Configuration class for [`experimental.gold.GOLDTrainer`].

    This class includes only the parameters that are specific to GOLD training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] and [`SFTConfig`] documentation.

    Args:
        > Parameters that control generation and the training loop

        temperature (`float`, *optional*, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        top_p (`float`, *optional*, defaults to `0.95`):
            If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to
            `top_p` or higher are kept for generation.
        top_k (`int`, *optional*, defaults to `0`):
            Number of highest probability vocabulary tokens to keep for top-k-filtering. If `0`, top-k-filtering is
            disabled and all tokens are considered.
        lmbda (`float`, *optional*, defaults to `0.5`):
            Lambda parameter that controls the student data fraction (i.e., the proportion of on-policy
            student-generated outputs).
        beta (`float`, *optional*, defaults to `0.5`):
            Interpolation coefficient between `0.0` and `1.0` of the Generalized Jensen-Shannon Divergence loss. When
            beta is `0.0`, the loss is the KL divergence. When beta is `1.0`, the loss is the Inverse KL Divergence.
        max_completion_length (`int`, *optional*, defaults to `128`):
            Maximum number of tokens to generate per completion.
        teacher_model_name_or_path (`str`, *optional*):
            Model name or path of the teacher model. If `None`, the teacher model will be the same as the model being
            trained.
        teacher_model_revision (`str` or `None`, *optional*, defaults to `None`):
            Model revision of the teacher model (e.g., branch name, tag, or commit hash). If `None`, the default
            revision is used.
        teacher_model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the teacher model
            from a string. The `revision` value is also used when loading the teacher processing class.
        teacher_tokenizer_name_or_path (`str`, *optional*):
            Tokenizer name or path for the teacher model. If None when using ULD loss, will use the same tokenizer as
            the student model (not recommended for cross-tokenizer distillation).
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model.
        seq_kd (`bool`, *optional*, defaults to `False`):
            Seq_kd parameter that controls whether to perform Sequence-Level KD (can be viewed as supervised FT on
            teacher-generated output).
        num_generations (`int`, *optional*, defaults to `1`):
            Number of generations per prompt. Each prompt is repeated this many times in the generation batch.
        generation_batch_size (`int` or `None`, *optional*, defaults to `None`):
            Number of unique prompts per worker per optimizer step. If `None`, it is computed from
            `(per_device_train_batch_size * gradient_accumulation_steps) // num_generations`.
        > Parameters that control the ULD loss

        use_uld_loss (`bool`, *optional*, defaults to `False`):
            Whether to use Universal Logit Distillation (ULD) loss instead of Generalized Jensen-Shannon Divergence
            loss.
        use_extended_uld (`bool`, *optional*, defaults to `True`):
            Whether to enable extended ULD alignment that uses tokenizers to align and merge token probabilities across
            student and teacher tokenizations. When `True`, the trainer will compute token mappings and merge
            probabilities for split tokens; when `False`, ULD will use simple positional truncation like in the
            original ULD paper. Set to `False` for SentencePiece or other non-ByteLevel tokenizers.
        uld_token_merge_strategy (`str`, *optional*, defaults to `"observed"`):
            Strategy used to align answer logits and merge token probabilities in the ULD loss. With `"observed"`, the
            answer logits are sliced at the answer positions and split tokens (when `use_extended_uld=True`) are merged
            by multiplying the marginal distribution at the first position by the scalar conditional probabilities of
            the actual later tokens. With `"bayesian"`, the answer-logit slice is shifted one position earlier so that
            `probs[k]` predicts `token_ids[k]` (chain rule), and split tokens are merged using the last position's full
            distribution, conditioned on the actual prefix tokens, multiplied by the scalar probabilities of the
            earlier tokens. The logit-alignment shift applies whether or not `use_extended_uld` is enabled.
        uld_use_hybrid_loss (`bool`, *optional*, defaults to `False`):
            Whether to use a hybrid loss that combines ULD loss and JSD loss. When `True`, the final loss is a
            combination of JSD for known token mappings and ULD for unknown token mappings.
        uld_hybrid_matched_weight (`float` or `None`, *optional*):
            Weight for the matched token loss component when using hybrid ULD + JSD loss. This weight scales the JSD
            loss computed over tokens that have a direct mapping between student and teacher tokenizations. If `None`,
            uses adaptive weighting based on vocabulary overlap. Must be set together with
            `uld_hybrid_unmatched_weight` (both `None` or both `float`).
        uld_hybrid_unmatched_weight (`float` or `None`, *optional*):
            Weight for the unmatched token loss component when using hybrid ULD + JSD loss. This weight scales the ULD
            loss computed over tokens that do not have a direct mapping between student and teacher tokenizations. If
            `None`, uses adaptive weighting based on vocabulary overlap. Must be set together with
            `uld_hybrid_matched_weight` (both `None` or both `float`).
        uld_crossentropy_weight (`float`, *optional*, defaults to `0.0`):
            Weight for the cross-entropy loss component in ULD loss. If 0, only ULD distillation loss is used.
        uld_distillation_weight (`float`, *optional*, defaults to `1.0`):
            Weight for the distillation loss component in ULD loss.
        uld_student_temperature (`float`, *optional*, defaults to `1.0`):
            Temperature for student logits in ULD loss computation.
        uld_teacher_temperature (`float`, *optional*, defaults to `1.0`):
            Temperature for teacher logits in ULD loss computation.
        uld_skip_student_eos (`bool`, *optional*, defaults to `True`):
            Whether to skip EOS token for student in ULD loss computation.
        uld_skip_teacher_eos (`bool`, *optional*, defaults to `True`):
            Whether to skip EOS token for teacher in ULD loss computation.
        > Parameters that control vLLM integration

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions from the student model. Requires `vllm` to be installed.
        vllm_mode (`str`, *optional*, defaults to `"colocate"`):
            Mode for student vLLM integration. Either `"server"` (connect to a running vLLM server) or `"colocate"`
            (run vLLM in the same process).
        vllm_server_host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host of the vLLM server for the student model (if `vllm_mode="server"`).
        vllm_server_port (`int`, *optional*, defaults to `8001`):
            Port of the vLLM server for the student model (if `vllm_mode="server"`).
        vllm_server_timeout (`float`, *optional*, defaults to `240.0`):
            Timeout for connecting to the student vLLM server (if `vllm_mode="server"`).
        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            GPU memory utilization for the colocated student vLLM engine (if `vllm_mode="colocate"`). It is recommended
            to set this to a low value if the student and teacher models share the same GPU.
        vllm_tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Tensor parallel size for the colocated student vLLM engine (if `vllm_mode="colocate"`).
        vllm_structured_outputs_regex (`str`, *optional*):
            Regex for vLLM structured outputs for the student model.
        vllm_server_base_url (`str`, *optional*):
            Base URL for the vLLM server (e.g., `"http://localhost:8001"`). If provided, `vllm_server_host` and
            `vllm_server_port` are ignored.
        vllm_group_port (`int`, *optional*, defaults to `51216`):
            Port for the vLLM weight-update group (NCCL communicator). Unless the port is occupied, there is no need to
            change it.
        vllm_max_model_length (`int`, *optional*):
            Maximum model sequence length for the colocated vLLM engine when `vllm_mode="colocate"`. Defaults to the
            model's maximum context length.
        vllm_model_impl (`str`, *optional*, defaults to `"vllm"`):
            Model implementation backend to use in vLLM. Use `"vllm"` (default) or `"transformers"`.
        vllm_sync_frequency (`int`, *optional*, defaults to `1`):
            Frequency (in training steps) to synchronize student model weights to vLLM engine. Set to 1 to sync after
            every step.
        vllm_enable_sleep_mode (`bool`, *optional*, defaults to `False`):
            Enable vLLM sleep mode to offload student weights/cache during the optimizer step. Keeps GPU memory usage
            low, but waking the engine adds host–device transfer latency.

        > Parameters that control logging

        log_completions (`bool`, *optional*, defaults to `False`):
            Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is installed,
            it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`.
        log_completions_steps (`int`, *optional*, defaults to `100`):
            Number of steps between logging (prompt, completion) pairs. Only used if `log_completions` is set to
            `True`.
        num_completions_to_print (`int` or `None`, *optional*):
            Number of completions to print with `rich`. If `None`, all completions are logged.
        wandb_log_unique_prompts (`bool`, *optional*, defaults to `True`):
            Whether to log the unique prompts to wandb. This will create a new run for each unique prompt.
        callbacks (`list[str]`, *optional*, defaults to `[]`):
            The callbacks to run during training.

    > [!NOTE]
    > These parameters have default values different from [`~transformers.TrainingArguments`]:
    > - `learning_rate`: Defaults to `1e-7` instead of `5e-5`.
    
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
        top_p = 0.95,
        top_k = 0,
        lmbda = 0.5,
        beta = 0.5,
        max_completion_length = 128,
        teacher_model_name_or_path = None,
        teacher_model_revision = None,
        teacher_model_init_kwargs = None,
        teacher_tokenizer_name_or_path = None,
        disable_dropout = True,
        seq_kd = False,
        num_generations = 1,
        generation_batch_size = None,
        use_uld_loss = False,
        uld_token_merge_strategy = 'observed',
        use_extended_uld = True,
        uld_use_hybrid_loss = False,
        uld_hybrid_matched_weight = None,
        uld_hybrid_unmatched_weight = None,
        uld_crossentropy_weight = 0.0,
        uld_distillation_weight = 1.0,
        uld_student_temperature = 1.0,
        uld_teacher_temperature = 1.0,
        uld_skip_student_eos = True,
        uld_skip_teacher_eos = True,
        use_vllm = False,
        vllm_mode = 'colocate',
        vllm_server_base_url = None,
        vllm_server_host = '0.0.0.0',
        vllm_server_port = 8001,
        vllm_server_timeout = 240.0,
        vllm_group_port = 51216,
        vllm_gpu_memory_utilization = 0.9,
        vllm_tensor_parallel_size = 1,
        vllm_max_model_length = None,
        vllm_model_impl = 'vllm',
        vllm_structured_outputs_regex = None,
        vllm_sync_frequency = 1,
        vllm_enable_sleep_mode = False,
        log_completions = False,
        log_completions_steps = 100,
        num_completions_to_print = None,
        wandb_log_unique_prompts = True,
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
            top_p = top_p,
            top_k = top_k,
            lmbda = lmbda,
            beta = beta,
            max_completion_length = max_completion_length,
            teacher_model_name_or_path = teacher_model_name_or_path,
            teacher_model_revision = teacher_model_revision,
            teacher_model_init_kwargs = teacher_model_init_kwargs,
            teacher_tokenizer_name_or_path = teacher_tokenizer_name_or_path,
            disable_dropout = disable_dropout,
            seq_kd = seq_kd,
            num_generations = num_generations,
            generation_batch_size = generation_batch_size,
            use_uld_loss = use_uld_loss,
            uld_token_merge_strategy = uld_token_merge_strategy,
            use_extended_uld = use_extended_uld,
            uld_use_hybrid_loss = uld_use_hybrid_loss,
            uld_hybrid_matched_weight = uld_hybrid_matched_weight,
            uld_hybrid_unmatched_weight = uld_hybrid_unmatched_weight,
            uld_crossentropy_weight = uld_crossentropy_weight,
            uld_distillation_weight = uld_distillation_weight,
            uld_student_temperature = uld_student_temperature,
            uld_teacher_temperature = uld_teacher_temperature,
            uld_skip_student_eos = uld_skip_student_eos,
            uld_skip_teacher_eos = uld_skip_teacher_eos,
            use_vllm = use_vllm,
            vllm_mode = vllm_mode,
            vllm_server_base_url = vllm_server_base_url,
            vllm_server_host = vllm_server_host,
            vllm_server_port = vllm_server_port,
            vllm_server_timeout = vllm_server_timeout,
            vllm_group_port = vllm_group_port,
            vllm_gpu_memory_utilization = vllm_gpu_memory_utilization,
            vllm_tensor_parallel_size = vllm_tensor_parallel_size,
            vllm_max_model_length = vllm_max_model_length,
            vllm_model_impl = vllm_model_impl,
            vllm_structured_outputs_regex = vllm_structured_outputs_regex,
            vllm_sync_frequency = vllm_sync_frequency,
            vllm_enable_sleep_mode = vllm_enable_sleep_mode,
            log_completions = log_completions,
            log_completions_steps = log_completions_steps,
            num_completions_to_print = num_completions_to_print,
            wandb_log_unique_prompts = wandb_log_unique_prompts,**kwargs)
        super().__init__(**_unsloth_filter_config_init_kwargs(GOLDConfig, _unsloth_config_arguments, mirrored_from = __class__))
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

class _UnslothGOLDTrainer(SFTTrainer):
    _tag_names = ["trl", "gold"]
    _name = "GOLD"
    _paper = {
        "title": "Unlocking On-Policy Distillation for Any Model Family",
        # docstyle-ignore
        "citation": textwrap.dedent(
            """\
            @misc{patino2025unlocking,
                title        = {{Unlocking On-Policy Distillation for Any Model Family}},
                author       = {Carlos Miguel Patiño and Kashif Rasul and Quentin Gallouédec and Ben Burtenshaw and Sergio Paniego and Vaibhav Srivastav and Thibaud Frere and Ed Beeching and Lewis Tunstall and Leandro von Werra and Thomas Wolf},
                year         = 2025,
                url          = {https://huggingface.co/spaces/HuggingFaceH4/general-on-policy-logit-distillation},
            }"""
        ),
    }

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        teacher_model: PreTrainedModel | nn.Module | str = None,
        args: GOLDConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: (
            PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin | None
        ) = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: (Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None) = None,
        peft_config: Optional["PeftConfig"] = None,
    ):

        if hasattr(model, 'vllm_engine') and hasattr(args, 'use_vllm'):
            _unsloth_fsdp_cfg = getattr(args, 'fsdp_config', None)
            if getattr(args, 'fsdp', None) and isinstance(_unsloth_fsdp_cfg, dict) and str(_unsloth_fsdp_cfg.get('version', _unsloth_fsdp_cfg.get('fsdp_version', 2))) == '2':
                raise NotImplementedError('Unsloth: `fast_inference = True` (vLLM) is not supported with FSDP2.\nLoad the model with `fast_inference = False` to train with FSDP2, or launch without FSDP2 to use vLLM.')
            if (getattr(args, 'use_vllm', False) == False):
                args.use_vllm = True
            if getattr(args, 'top_k', -1) is None or getattr(args, 'top_k', -1) == 0:
                args.top_k = -1
        self.model_name_or_path = model if isinstance(model, str) else model.config._name_or_path
        self.model_revision = (args.model_init_kwargs or {}).get("revision") if isinstance(model, str) else None
        teacher_revision = (
            (args.teacher_model_init_kwargs or {}).get("revision", args.teacher_model_revision)
            if isinstance(teacher_model, str)
            else None
        )
        dataset_sample = next(iter(train_dataset)) if train_dataset is not None else {}
        if processing_class is None:
            model_id = model if isinstance(model, str) else get_config_model_id(model.config)
            processing_class = AutoProcessor.from_pretrained(
                model_id, revision=self.model_revision, trust_remote_code=args.trust_remote_code
            )
            # simplified logic from SFTTrainer
        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            self._tokenizer = processing_class.tokenizer
            self._is_vlm = True
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        else:
            self._tokenizer = processing_class
            self._is_vlm = False

        self.pad_token_id = self._tokenizer.pad_token_id

        # VLM distillation: only VLM-to-VLM is supported. Both student and teacher must be
        # VLMs so that both receive images and multimodal inputs.
        self._teacher_processor = None
        self._is_cross_architecture_vlm = False
        if self._is_vlm:
            if isinstance(teacher_model, str):
                # Teacher not yet instantiated -- validate it's a VLM
                teacher_proc = AutoProcessor.from_pretrained(
                    teacher_model, revision=teacher_revision, trust_remote_code=args.trust_remote_code
                )
                if not isinstance(teacher_proc, ProcessorMixin):
                    raise ValueError(
                        "VLM distillation requires both student and teacher to be vision-language models. "
                        "The student has a `ProcessorMixin` but the teacher does not."
                    )
                teacher_model_type = AutoConfig.from_pretrained(
                    teacher_model, trust_remote_code=args.trust_remote_code
                ).model_type
            else:
                # Teacher already instantiated — check if it looks like a VLM by checking for a vision config
                if teacher_model.config.vision_config is None:
                    raise ValueError(
                        "VLM distillation requires both student and teacher to be vision-language models. "
                        "The student has a `ProcessorMixin` but the teacher model does not appear to be a VLM "
                        "(missing `vision_config`)."
                    )
                teacher_model_type = teacher_model.config.model_type

            # Check for cross-architecture VLM distillation
            student_model_type = (
                AutoConfig.from_pretrained(model, trust_remote_code=args.trust_remote_code).model_type
                if isinstance(model, str)
                else model.config.model_type
            )
            is_cross_architecture = student_model_type and teacher_model_type != student_model_type
            self._is_cross_architecture_vlm = is_cross_architecture
            if is_cross_architecture:
                warnings.warn(
                    f"Cross-architecture VLM distillation detected: student is '{student_model_type}', "
                    f"teacher is '{teacher_model_type}'. Images will be processed separately through each "
                    "model's processor, which may increase memory usage and computation time."
                )
            if is_cross_architecture or args.use_uld_loss:
                self._teacher_processor = (
                    teacher_proc
                    if isinstance(teacher_model, str)
                    else AutoProcessor.from_pretrained(
                        teacher_model.config._name_or_path,
                        revision=teacher_revision,
                        trust_remote_code=args.trust_remote_code,
                    )
                )
        if self._is_cross_architecture_vlm and not args.use_uld_loss:
            raise ValueError(
                "Cross-architecture VLM distillation (student and teacher have different `model_type`) is not "
                "supported with the standard JSD loss because the models require different image token formats "
                "and tokenizers. Please set `use_uld_loss=True` in your GOLDConfig to enable cross-tokenizer "
                "alignment via ULD loss."
            )
        self._is_vision_dataset = "image" in dataset_sample or "images" in dataset_sample
        if self._is_vision_dataset and not self._is_vlm:
            raise ValueError(
                "The dataset appears to be vision-related (contains 'image' or 'images' keys), but the provided "
                "model does not seem to be a vision-language model. Please check your model and dataset."
            )
        if self._is_vlm and args.max_length is not None and args.truncation_mode == "keep_end":
            raise ValueError(
                "truncation_mode='keep_end' is not supported for vision-language models. Image tokens reside "
                "inside the prompt portion of the sequence; depending on the example, keep_end may silently "
                "drop them, causing pixel_values to be forwarded to the model with no corresponding visual "
                "tokens in input_ids. Use truncation_mode='keep_start' (the default) or set max_length=None."
            )

        # Respect a user-provided data_collator for text; otherwise, pick the right collator based on modality.
        # For VLMs, always use identity collator to preserve raw PIL images in the dataloader.
        # Raw images are needed for: [1] vLLM generation, [2] cross-architecture teacher processing.
        # A separate _vlm_collator is stored for on-the-fly collation inside _fill_buffer.
        if self._is_vision_dataset and data_collator is not None:
            raise ValueError(
                "Passing a custom data collator is not supported for VLM training. GOLD manages its own collation "
                "to preserve raw images for generation and teacher processing; leave `data_collator=None`."
            )
        self._vlm_collator = None
        if data_collator is None:
            if self._is_vision_dataset:
                self._vlm_collator = DataCollatorForVisionLanguageChatML(
                    processor=processing_class,
                    max_length=args.max_length,
                )
                data_collator = identity
            else:
                data_collator = DataCollatorForChatML(tokenizer=self._tokenizer, max_length=args.max_length)

        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the GOLDConfig, but your teacher_model is already instantiated."
            )
        else:
            teacher_model_init_kwargs = args.teacher_model_init_kwargs
            teacher_model_init_kwargs["dtype"] = (
                teacher_model_init_kwargs["dtype"]
                if teacher_model_init_kwargs["dtype"] in ["auto", None]
                else getattr(torch, teacher_model_init_kwargs["dtype"])
            )

        if args.use_uld_loss and args.teacher_tokenizer_name_or_path is None:
            if isinstance(teacher_model, str):
                args.teacher_tokenizer_name_or_path = teacher_model
            elif teacher_model.config._name_or_path:
                args.teacher_tokenizer_name_or_path = teacher_model.config._name_or_path
            else:
                raise ValueError(
                    "`teacher_tokenizer_name_or_path` must be set when using ULD loss with a pre-instantiated teacher model."
                )

        # The teacher revision pins a commit in the teacher model's repo, so it only applies to a tokenizer served from
        # that same repo. ULD's cross-tokenizer setup points `teacher_tokenizer_name_or_path` at a different repo, where
        # that commit does not exist.
        teacher_tokenizer_revision = teacher_revision if args.teacher_tokenizer_name_or_path == teacher_model else None

        if isinstance(teacher_model, str):
            init_kwargs = dict(teacher_model_init_kwargs)
            if args.teacher_model_revision is not None:
                init_kwargs.setdefault("revision", args.teacher_model_revision)
            init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
            # Distributed training requires device_map=None ["auto" fails]
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                init_kwargs["device_map"] = None
            teacher_model = create_model_from_path(teacher_model, **init_kwargs)
        self.use_uld_loss = args.use_uld_loss
        self.teacher_tokenizer = None
        if args.use_uld_loss and self._teacher_processor is not None:
            self.teacher_tokenizer = self._teacher_processor.tokenizer
            if self.teacher_tokenizer.pad_token is None:
                self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
        elif args.use_uld_loss and args.teacher_tokenizer_name_or_path is not None:
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(
                args.teacher_tokenizer_name_or_path,
                revision=teacher_tokenizer_revision,
                trust_remote_code=args.trust_remote_code,
            )
            if self.teacher_tokenizer.pad_token is None:
                self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token

        # Hybrid ULD loss configuration is handled in ULDLoss class

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
        )

        if args.disable_dropout:
            disable_dropout_in_model(self.model)
        if not args.use_uld_loss:
            teacher_model.resize_token_embeddings(self.model.config.get_text_config().vocab_size)

        if self.is_deepspeed_enabled:
            self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.seq_kd = args.seq_kd
        self.num_generations = args.num_generations

        # Track per-step loss statistics for on/off-policy batches [used in logging]
        self._on_policy_loss_total = 0.0
        self._off_policy_loss_total = 0.0
        self._on_policy_step_equiv = 0.0
        self._off_policy_step_equiv = 0.0

        # Buffering for rollouts across gradient accumulation steps
        self._buffered_inputs = None
        self._buffered_on_policy = None
        self._buffered_text_logs = None
        self._step = 0

        # Hybrid ULD matched/unmatched accumulators [logged every step when ULD hybrid is used]
        self._matched_sum = 0.0
        self._unmatched_sum = 0.0
        self._matched_step_eq = 0.0
        self._unmatched_step_eq = 0.0

        self.uld_loss_fn = None
        if self.use_uld_loss:
            self.uld_loss_fn = ULDLoss(
                config=args,
                student_tokenizer=self._tokenizer,
                teacher_tokenizer=self.teacher_tokenizer,
                device=self.accelerator.device,
            )

        generation_kwargs = {
            "max_new_tokens": args.max_completion_length,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "do_sample": True,
            "top_k": args.top_k,
            "pad_token_id": self.pad_token_id,
        }
        self.generation_config = GenerationConfig(**generation_kwargs)
        # Keep training-specific generation kwargs to overwrite model's original generation config
        self.generation_kwargs = generation_kwargs
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.log_completion_steps = args.log_completions_steps
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.gradient_accumulation_steps
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
            "advantages": deque(maxlen=maxlen),
        }

        self.use_vllm = args.use_vllm
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and use_vllm is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )
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
                max_model_length=args.vllm_max_model_length or args.max_length,
                max_num_seqs=args.per_device_train_batch_size * args.gradient_accumulation_steps,
                enable_sleep_mode=args.vllm_enable_sleep_mode,
                model_impl=args.vllm_model_impl,
                trust_remote_code=args.trust_remote_code,
                repetition_penalty=getattr(args, "repetition_penalty", 1.0),
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=getattr(args, "min_p", 0.0),
                max_completion_length=args.max_completion_length,
                logprobs=None,
            )
            self.vllm_sync_frequency = args.vllm_sync_frequency
            self._last_vllm_sync_step = -self.vllm_sync_frequency

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        required_columns = [
            "prompt",
            "completion",
            "prompts",
            "prompt_attention_mask",
            "messages",
            "chat_template_kwargs",
            "tools",
            "original_prompt_text",
            "original_completion_text",
            "byte_offsets",
            "completion_mask",
            "images",
            "image",
            "pixel_values",
            "image_grid_thw",
            "image_position_ids",
            "pixel_attention_mask",
            "image_sizes",
            "spatial_shapes",
            "token_type_ids",
            "mm_token_type_ids",
        ]
        if self._signature_columns is None:
            self._signature_columns = required_columns
        else:
            for column in required_columns:
                if column not in self._signature_columns:
                    self._signature_columns.append(column)

    def _get_train_sampler(self, dataset=None):
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size * self.accelerator.num_processes,
            repeat_count=self.args.gradient_accumulation_steps,
            shuffle=True,
            seed=self.args.seed,
        )

    def get_train_dataloader(self):
        """
        Override Trainer.get_train_dataloader to load one generation batch per optimizer window.

        The dataloader yields local batches of size `per_device_train_batch_size * gradient_accumulation_steps`. The
        `RepeatSampler` (with `repeat_count=gradient_accumulation_steps`) ensures each generation batch is sampled
        `gradient_accumulation_steps` times so Trainer's loop iterates the correct number of times. Only the first
        batch in each window triggers `_fill_buffer`; the rest are ignored by `_prepare_inputs`.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.gradient_accumulation_steps,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            )
            if self.args.dataloader_num_workers > 0:
                dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    @profiling_decorator
    def _prepare_inputs(self, generation_batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        if not self.model.training:
            # Evaluation is off-policy (no generation): the student never samples, both models are forwarded over
            # the dataset's ground-truth prompt+completion and the distillation loss is taken over the completion.
            # For text the collated tensor dict is consumed directly. For VLMs the identity collator yields raw
            # dicts (to preserve PIL images for the train-time on-policy path), so collate them here -- mirroring
            # the off-policy slice construction in _fill_buffer -- including the raw images/prompts the cross-arch
            # / ULD teacher processor needs.
            if self._vlm_collator is not None:
                pending_slice = {"_gold_vlm_lazy_examples": list(generation_batch)}
                if self._teacher_processor is not None:
                    raw_images, raw_prompts = self._extract_images_and_prompts(list(generation_batch))
                    pending_slice["_gold_vlm_raw_images"] = raw_images
                    pending_slice["_gold_vlm_raw_prompts"] = raw_prompts
                return self._materialize_vlm_slice(pending_slice)
            return generation_batch

        buffer_steps = self.args.gradient_accumulation_steps
        if self._step % buffer_steps == 0 or self._buffered_inputs is None:
            self._fill_buffer(generation_batch, buffer_steps)

        slice_idx = self._step % buffer_steps
        inputs = self._buffered_inputs[slice_idx]
        if isinstance(inputs, dict):
            if "_gold_vlm_on_policy_raw_examples" in inputs:
                inputs, text_logs = self._generate_on_policy_vlm_slice(inputs)
                self._buffered_inputs[slice_idx] = inputs
                self._buffered_text_logs[slice_idx] = text_logs
            elif "_gold_vlm_lazy_examples" in inputs:
                inputs = self._materialize_vlm_slice(inputs)
                self._buffered_inputs[slice_idx] = inputs
        self._step += 1
        return inputs

    def _generate_on_policy_vlm_slice(
        self, pending_slice: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor | Any], tuple[list[str], list[str]]]:
        """Generate and collate one non-vLLM on-policy VLM slice immediately before it is consumed."""
        raw_examples = pending_slice["_gold_vlm_on_policy_raw_examples"]
        generation_examples = []
        for example in raw_examples:
            generation_example = dict(example)
            completion = generation_example.get("completion")
            generation_example["completion"] = (
                [{"role": "assistant", "content": [{"type": "text", "text": ""}]}]
                if isinstance(completion, list)
                else ""
            )
            generation_examples.append(generation_example)
        collated = self._vlm_collator(generation_examples)
        collated = {
            k: (v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v) for k, v in collated.items()
        }

        with unwrap_model_for_generation(
            self.model, self.accelerator, generation_kwargs=self.generation_kwargs
        ) as unwrapped_model:
            (
                new_input_ids,
                new_attention_mask,
                new_labels,
                prompt_texts,
                completion_texts,
            ) = self.generate_on_policy_outputs(
                unwrapped_model,
                collated,
                self.generation_config,
            )

        updated_slice = dict(collated)
        updated_slice["input_ids"] = new_input_ids
        updated_slice["attention_mask"] = new_attention_mask
        updated_slice["labels"] = new_labels
        # Rebuild sequence-length-dependent keys to match new input_ids shape
        new_seq_len = new_input_ids.shape[1]
        prompt_seq_len = collated["prompts"].shape[1]
        for k in self._SEQUENCE_KEYS:
            if k in updated_slice:
                sequence_dtype = updated_slice[k].dtype
                prompt_part = self._get_prompt_sequence_key(collated, k)
                comp_part = torch.zeros(
                    new_input_ids.shape[0],
                    new_seq_len - prompt_seq_len,
                    dtype=sequence_dtype,
                    device=new_input_ids.device,
                )
                updated_slice[k] = torch.cat([prompt_part, comp_part], dim=1)
        if "original_prompt_text" not in updated_slice:
            updated_slice["original_prompt_text"] = prompt_texts
        # Keep special tokens (e.g. EOS) so the teacher inputs are built from text covering the same generated
        # content as the supervised `labels != -100` tokens. The alignment itself remains byte-offset based.
        updated_slice["original_completion_text"] = completion_texts
        self._maybe_add_completion_byte_offsets(updated_slice)
        if self._teacher_processor is not None:
            updated_slice["_raw_images"] = pending_slice["_gold_vlm_raw_images"]
            updated_slice["_raw_prompts"] = pending_slice["_gold_vlm_raw_prompts"]

        return updated_slice, (prompt_texts, completion_texts)

    def _materialize_vlm_slice(self, pending_slice: dict[str, Any]) -> dict[str, torch.Tensor | Any]:
        """Collate one pending VLM slice immediately before it is consumed."""
        slice_inputs = self._vlm_collator([dict(example) for example in pending_slice["_gold_vlm_lazy_examples"]])
        slice_inputs = {
            k: (v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v) for k, v in slice_inputs.items()
        }

        if self.use_uld_loss and self.teacher_tokenizer is not None:
            slice_inputs = self._ensure_original_text_fields(slice_inputs)
            if "original_prompt_text" not in slice_inputs or "original_completion_text" not in slice_inputs:
                raise ValueError(
                    "Off-policy batch missing 'original_prompt_text' or 'original_completion_text' fields. "
                    "When using ULD loss with cross-tokenizer alignment, datasets must be prepared with "
                    "_prepare_dataset_with_original_text(). Ensure your dataset includes these fields."
                )

        if self._teacher_processor is not None:
            slice_inputs["_raw_images"] = pending_slice["_gold_vlm_raw_images"]
            slice_inputs["_raw_prompts"] = pending_slice["_gold_vlm_raw_prompts"]

        return slice_inputs

    @staticmethod
    def _build_sequence_batch(
        new_input_ids: torch.Tensor,
        prompt_lengths: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build attention mask and labels from full sequences and prompt lengths."""
        prompt_lengths = prompt_lengths.to(device=new_input_ids.device, dtype=torch.long)
        positions = torch.arange(new_input_ids.shape[1], device=new_input_ids.device).unsqueeze(0)
        completion_mask = positions >= prompt_lengths.unsqueeze(1)

        new_attention_mask = attention_mask.to(device=new_input_ids.device, dtype=new_input_ids.dtype)

        new_labels = torch.full_like(new_input_ids, -100)
        new_labels[completion_mask & new_attention_mask.bool()] = new_input_ids[
            completion_mask & new_attention_mask.bool()
        ]

        return new_attention_mask, new_labels

    def _extract_images_and_prompts(self, examples: list[dict]) -> tuple[list | None, list]:
        """
        Extract per-example images and build prompts with multimodal messages, mirroring GRPOTrainer.

        Returns `(images, prompts)` where `images` is a per-example list (entries may be `None`), or `None` when the
        batch carries no images, and `prompts` are the prepared multimodal messages.
        """
        if "images" in examples[0]:
            images = [example.get("images") for example in examples]
        elif "image" in examples[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in examples]
        else:
            images = None
        if images is not None and all(img_list is None or img_list == [] for img_list in images):
            images = None

        prompts = [example["prompt"] for example in examples]
        if images is not None:
            prompts = [
                prepare_multimodal_messages(prompt, images=img_list)
                for prompt, img_list in zip(prompts, images, strict=True)
            ]
        return images, prompts

    def _decode_completion_texts_from_labels(self, slice_inputs: dict[str, torch.Tensor | Any]) -> list[str] | None:
        """Decode completion text from labels when raw text is absent."""
        labels = slice_inputs.get("labels")
        if labels is None or not isinstance(labels, torch.Tensor):
            return None

        labels_cpu = labels.detach().cpu()
        decoded_completion_tokens: list[list[int]] = []
        for row in labels_cpu:
            token_ids = row[row != -100].tolist()
            decoded_completion_tokens.append(token_ids)

        return self.processing_class.batch_decode(
            decoded_completion_tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _ensure_original_text_fields(
        self, slice_inputs: dict[str, torch.Tensor | Any]
    ) -> dict[str, torch.Tensor | Any]:
        """Populate original prompt/completion text fields when missing."""
        if "original_prompt_text" in slice_inputs and "original_completion_text" in slice_inputs:
            return slice_inputs

        prompts = slice_inputs.get("prompts")
        if prompts is None or not isinstance(prompts, torch.Tensor):
            return slice_inputs

        prompt_texts = self.processing_class.batch_decode(
            prompts,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        completion_texts = self._decode_completion_texts_from_labels(slice_inputs)
        if completion_texts is None:
            return slice_inputs

        updated_slice = dict(slice_inputs)
        updated_slice["original_prompt_text"] = prompt_texts
        updated_slice["original_completion_text"] = completion_texts
        return updated_slice

    @staticmethod
    def _get_prompt_sequence_key(inputs: dict[str, torch.Tensor | Any], key: str) -> torch.Tensor:
        """Align a sequence-length-dependent key with the left-padded prompt tensor."""
        values = inputs[key]
        prompts = inputs["prompts"]
        prompt_attention_mask = inputs.get("prompt_attention_mask")

        if prompt_attention_mask is None:
            return values[:, : prompts.shape[1]]

        prompt_values = values.new_zeros(prompts.shape)
        for i, mask in enumerate(prompt_attention_mask.bool()):
            prompt_length = int(mask.sum().item())
            if prompt_length:
                prompt_values[i, mask] = values[i, :prompt_length]
        return prompt_values

    _SEQUENCE_KEYS = ("token_type_ids", "mm_token_type_ids")
    _MODEL_INPUT_RESERVED_KEYS = frozenset(
        (
            "input_ids",
            "attention_mask",
            "labels",
            "prompts",
            "prompt_attention_mask",
            "completion_mask",
            "assistant_masks",
            "original_prompt_text",
            "original_completion_text",
            "byte_offsets",
        )
    )

    def _get_model_forward_kwargs(
        self, inputs: dict[str, torch.Tensor | Any], exclude: tuple[str, ...] = ()
    ) -> dict[str, torch.Tensor]:
        reserved_keys = self._MODEL_INPUT_RESERVED_KEYS | set(exclude)
        return {
            k: v
            for k, v in inputs.items()
            if k not in reserved_keys and not k.startswith("_") and isinstance(v, torch.Tensor)
        }

    def _maybe_add_completion_byte_offsets(self, updated_slice: dict[str, torch.Tensor | Any]) -> None:
        """Attach completion-relative byte offsets to on-policy ULD batches.

        Derived from the sampled ids via ``piece_byte_len`` (no decode→re-encode round-trip).
        """
        if not (
            self.use_uld_loss
            and self.teacher_tokenizer is not None
            and self.uld_loss_fn is not None
            and self.uld_loss_fn.use_extended_uld
        ):
            return

        new_input_ids = updated_slice["input_ids"]
        new_labels = updated_slice["labels"]
        seq_len = new_input_ids.shape[1]

        # convert_ids_to_tokens is a tokenizer method; VLM processors expose it via `.tokenizer`.
        tokenizer = self._tokenizer

        rows: list[list[tuple[int, int]]] = []
        for row_ids, row_labels in zip(new_input_ids.cpu().tolist(), new_labels.cpu().tolist(), strict=True):
            offs: list[tuple[int, int]] = [(0, 0)] * seq_len
            cumulative = 0
            for pos, (tid, label) in enumerate(zip(row_ids, row_labels, strict=True)):
                if label == -100:
                    continue
                nb = piece_byte_len(tokenizer.convert_ids_to_tokens([tid])[0])
                offs[pos] = (cumulative, cumulative + nb)
                cumulative += nb
            rows.append(offs)
        updated_slice["byte_offsets"] = torch.tensor(rows, dtype=torch.long, device=new_input_ids.device)

    @profiling_decorator
    def _fill_buffer(
        self,
        generation_batch: dict[str, torch.Tensor | Any] | list[dict],
        buffer_steps: int,
    ):
        # The batch is split into `buffer_steps` equal chunks via floor division (here and in
        # `split_tensor_dict` below). A batch size that is not divisible by `buffer_steps` would
        # silently drop examples or produce empty chunks, so fail fast instead.
        batch_len = (
            len(generation_batch)
            if isinstance(generation_batch, list)
            else next(t for t in generation_batch.values() if t is not None).shape[0]
        )
        if batch_len % buffer_steps != 0:
            raise ValueError(
                "The generation batch size must be divisible by gradient_accumulation_steps. Set "
                "dataloader_drop_last=True or adjust per_device_train_batch_size / gradient_accumulation_steps."
            )

        if self._vlm_collator is not None:
            # Identity collator path: generation_batch is list[dict] with raw PIL images.
            # Split into chunks via list slicing, then collate on-the-fly per slice.
            chunk_size = len(generation_batch) // buffer_steps
            raw_slices = [generation_batch[i * chunk_size : (i + 1) * chunk_size] for i in range(buffer_steps)]
            slices = None  # not used in this path
        else:
            raw_slices = None  # not used in this path
            slices = split_tensor_dict(generation_batch, buffer_steps)

        if self.accelerator.is_main_process:
            on_policy_flags = [random.random() <= self.lmbda for _ in range(buffer_steps)]
        else:
            on_policy_flags = [False] * buffer_steps

        on_policy_flags = broadcast_object_list(on_policy_flags, from_process=0)
        on_policy_indices = [i for i, flag in enumerate(on_policy_flags) if flag]

        self._buffered_inputs = [None] * buffer_steps
        self._buffered_on_policy = on_policy_flags
        self._buffered_text_logs = [None] * buffer_steps

        for i, flag in enumerate(on_policy_flags):
            if not flag:
                if self._vlm_collator is not None:
                    # Extract raw images and prompts BEFORE collation, since the collator
                    # mutates examples in place (pops "image", overwrites "prompt").
                    slice_inputs = {"_gold_vlm_lazy_examples": raw_slices[i]}
                    if self._teacher_processor is not None:
                        raw_images, raw_prompts = self._extract_images_and_prompts(raw_slices[i])
                        slice_inputs["_gold_vlm_raw_images"] = raw_images
                        slice_inputs["_gold_vlm_raw_prompts"] = raw_prompts
                    self._buffered_inputs[i] = slice_inputs
                    continue

                slice_inputs = slices[i]

                if (
                    self.use_uld_loss
                    and self.teacher_tokenizer is not None
                    and ("original_prompt_text" not in slice_inputs or "original_completion_text" not in slice_inputs)
                ):
                    raise ValueError(
                        "Off-policy batch missing 'original_prompt_text' or 'original_completion_text' fields. "
                        "Use the default DataCollatorForChatML (or a collator that emits these fields) so the "
                        "teacher tokenizer has source text to align against."
                    )
                if (
                    self.use_uld_loss
                    and self.teacher_tokenizer is not None
                    and self.uld_loss_fn.use_extended_uld
                    and "byte_offsets" not in slice_inputs
                ):
                    raise ValueError(
                        "Off-policy batch missing `byte_offsets`. Use the default DataCollatorForChatML or set "
                        "`use_extended_uld=False`."
                    )

                self._buffered_inputs[i] = slice_inputs

        if on_policy_indices:
            if self._vlm_collator is not None:
                self._generate_on_policy_vlm_raw(raw_slices, on_policy_indices)
            else:
                self._generate_on_policy_for_slices(slices, on_policy_indices)

    @profiling_decorator
    def _generate_on_policy_for_slices(
        self, slices: list[dict[str, torch.Tensor | Any]], on_policy_indices: list[int]
    ):
        prompt_ids_list = []
        local_slice_indices = []
        # Budget the prompt to max_length - max_new_tokens before generation, keeping the END of the
        # prompt (the generation marker), so the student generates from and trains on the same context
        # and prompt + completion fit in max_length.
        max_completion_length = self.generation_config.max_new_tokens
        prompt_max_length = max(1, self.args.max_length - max_completion_length) if self.args.max_length else None
        for slice_idx in on_policy_indices:
            slice_inputs = slices[slice_idx]
            prompt_attention_mask = slice_inputs.get("prompt_attention_mask")
            for prompt_idx, prompt in enumerate(slice_inputs["prompts"]):
                if prompt_attention_mask is not None:
                    prompt = prompt[prompt_attention_mask[prompt_idx].bool()]
                prompt_ids = prompt.tolist()
                if prompt_max_length is not None and len(prompt_ids) > prompt_max_length:
                    prompt_ids = prompt_ids[-prompt_max_length:]
                prompt_ids_list.append(prompt_ids)
                local_slice_indices.append(slice_idx)

        prompts_text = self.processing_class.batch_decode(
            prompt_ids_list,
            skip_special_tokens=False,
        )

        if not self.use_vllm:
            self._generate_non_vllm_for_slices(slices, on_policy_indices)
            return

        if (
            self.state.global_step != self._last_vllm_sync_step
            and self.state.global_step >= self._last_vllm_sync_step + self.vllm_sync_frequency
        ):
            self.vllm_generation.sync_weights()
            self._last_vllm_sync_step = self.state.global_step

        _, completion_ids, _, _ = self.vllm_generation.generate(
            prompts=prompt_ids_list,
            images=None,
            num_generations=self.num_generations,
        )

        self._process_completions_to_buffer(
            slices,
            on_policy_indices,
            local_slice_indices,
            completion_ids,
            prompt_ids_list,
            prompts_text,
            self.generation_config.max_new_tokens,
        )

    def _generate_non_vllm_for_slices(self, slices: list[dict[str, torch.Tensor | Any]], on_policy_indices: list[int]):
        """Fallback generation without vLLM (uses model.generate per slice)."""
        with unwrap_model_for_generation(
            self.model,
            self.accelerator,
            generation_kwargs=self.generation_kwargs,
        ) as unwrapped_model:
            max_completion_length = self.generation_config.max_new_tokens
            prompt_max_length = max(1, self.args.max_length - max_completion_length) if self.args.max_length else None
            pad_token_id = self.processing_class.pad_token_id if self.processing_class.pad_token_id is not None else 0
            for slice_idx in on_policy_indices:
                slice_inputs = slices[slice_idx]
                if prompt_max_length is not None:
                    # Budget per real token (keep-end), mirroring the vLLM path so both keep the same tokens.
                    # Trigger on the real-token count, not the padded width, and rebuild only when a row
                    # overflows, so no pad columns are carried into the budgeted prompt.
                    prompts = slice_inputs["prompts"]
                    prompt_attention_mask = slice_inputs.get("prompt_attention_mask")
                    budgeted_rows = []
                    needs_budget = False
                    for row_idx, prompt in enumerate(prompts):
                        real_ids = (
                            prompt[prompt_attention_mask[row_idx].bool()]
                            if prompt_attention_mask is not None
                            else prompt
                        )
                        if real_ids.shape[0] > prompt_max_length:
                            real_ids = real_ids[-prompt_max_length:]
                            needs_budget = True
                        budgeted_rows.append(real_ids)
                    if needs_budget:
                        slice_inputs = dict(slice_inputs)
                        slice_inputs["prompts"] = pad(budgeted_rows, padding_side="left", padding_value=pad_token_id)
                        slice_inputs["prompt_attention_mask"] = pad(
                            [torch.ones(row.shape[0], dtype=torch.long, device=row.device) for row in budgeted_rows],
                            padding_side="left",
                            padding_value=0,
                        )
                result = self.generate_on_policy_outputs(
                    unwrapped_model,
                    slice_inputs,
                    self.generation_config,
                )
                (
                    new_input_ids,
                    new_attention_mask,
                    new_labels,
                    prompt_texts,
                    completion_texts,
                ) = result

                updated_slice = dict(slice_inputs)
                updated_slice["input_ids"] = new_input_ids
                updated_slice["attention_mask"] = new_attention_mask
                updated_slice["labels"] = new_labels
                updated_slice["original_prompt_text"] = prompt_texts
                updated_slice["original_completion_text"] = completion_texts
                self._maybe_add_completion_byte_offsets(updated_slice)

                self._buffered_inputs[slice_idx] = updated_slice
                self._buffered_text_logs[slice_idx] = (prompt_texts, completion_texts)

    def _generate_on_policy_vlm_raw(self, raw_slices: list[list[dict]], on_policy_indices: list[int]):
        """On-policy generation from raw VLM examples, preserving PIL images for vLLM."""
        all_prompt_ids = []
        all_images = []
        all_prompts = []
        all_raw_examples = []
        local_slice_indices = []
        slice_raw_data = {}
        max_completion_length = self.generation_config.max_new_tokens
        prompt_max_length = max(1, self.args.max_length - max_completion_length) if self.args.max_length else None

        for slice_idx in on_policy_indices:
            raw_examples = raw_slices[slice_idx]

            images, prompts = self._extract_images_and_prompts(raw_examples)

            prompts = [
                [
                    (
                        {**msg, "content": [{"type": "text", "text": msg["content"]}]}
                        if isinstance(msg.get("content"), str)
                        else msg
                    )
                    for msg in prompt
                ]
                for prompt in prompts
            ]

            if not self.use_vllm:
                slice_raw_data[slice_idx] = (raw_examples, images, prompts, None)
                continue

            # TODO: add self.tools support
            # Workaround for a bug in transformers 5.3.0 where some processors (e.g. Qwen2.5-VL) crash on
            # batched unpadded input (transformers#44514).
            # Fixed in transformers 5.4.0 (transformers#44563).
            needs_padding_workaround = Version("5.3.0") <= Version(transformers.__version__) < Version("5.4.0")
            tokenized = self.processing_class.apply_chat_template(
                conversation=prompts,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **({"padding": True} if needs_padding_workaround else {}),
            )
            if needs_padding_workaround:
                # Unpad input_ids: remove padding tokens using attention_mask to get per-sequence lists
                prompt_ids_list = [
                    [tok for tok, m in zip(ids, mask, strict=True) if m]
                    for ids, mask in zip(tokenized["input_ids"], tokenized["attention_mask"], strict=True)
                ]
            else:
                prompt_ids_list = tokenized["input_ids"]
            if prompt_max_length is not None:
                # Do not truncate VLM prompts: a token-level slice can cut through the expanded image-token block,
                # desyncing input_ids from the image features (garbage generations), and the buffered training
                # examples are rebuilt from the untruncated rows, so the completion would be sampled under a
                # different prompt than the one used for the loss. Fail fast instead.
                for prompt_ids in prompt_ids_list:
                    if len(prompt_ids) > prompt_max_length:
                        raise ValueError(
                            f"A tokenized VLM prompt has {len(prompt_ids)} tokens (including expanded image "
                            f"tokens), exceeding the prompt budget of max_length - max_completion_length = "
                            f"{self.args.max_length} - {max_completion_length} = {prompt_max_length}. Increase `max_length` or set it to `None`."
                        )

            slice_raw_data[slice_idx] = (raw_examples, images, prompts, prompt_ids_list)

            for i, example in enumerate(raw_examples):
                all_prompt_ids.append(prompt_ids_list[i])
                all_images.append(images[i] if images is not None else None)
                all_prompts.append(prompts[i])
                all_raw_examples.append(example)
                local_slice_indices.append(slice_idx)

        if not self.use_vllm:
            for slice_idx in on_policy_indices:
                raw_examples, images, prompts, _ = slice_raw_data[slice_idx]
                has_images = images is not None and any(img is not None for img in images)
                pending_slice = {"_gold_vlm_on_policy_raw_examples": raw_examples}
                if self._teacher_processor is not None:
                    pending_slice["_gold_vlm_raw_images"] = images if has_images else None
                    pending_slice["_gold_vlm_raw_prompts"] = prompts
                self._buffered_inputs[slice_idx] = pending_slice
            return

        all_prompts_text = self.processing_class.batch_decode(all_prompt_ids, skip_special_tokens=True)
        if (
            self.state.global_step != self._last_vllm_sync_step
            and self.state.global_step >= self._last_vllm_sync_step + self.vllm_sync_frequency
        ):
            self.vllm_generation.sync_weights()
            self._last_vllm_sync_step = self.state.global_step

        if any(img is not None for img in all_images):
            generate_images = all_images
        else:
            generate_images = None
        _, completion_ids, _, _ = self.vllm_generation.generate(
            prompts=all_prompt_ids,
            images=generate_images,
            num_generations=self.num_generations,
        )

        all_completion_texts = []
        for comp_ids in completion_ids:
            if len(comp_ids) > max_completion_length:
                comp_ids = comp_ids[:max_completion_length]
            all_completion_texts.append(
                self.processing_class.decode(
                    comp_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )

        # Redistribute completions to slices. The RepeatSampler has already duplicated examples
        # `num_generations` times, so completions align 1:1 with the sampled input entries.
        slice_completions = {idx: [] for idx in on_policy_indices}
        slice_raw = {idx: [] for idx in on_policy_indices}
        slice_images = {idx: [] for idx in on_policy_indices}
        slice_prompts = {idx: [] for idx in on_policy_indices}
        slice_prompts_text = {idx: [] for idx in on_policy_indices}

        for i, slice_idx in enumerate(local_slice_indices):
            slice_completions[slice_idx].append(all_completion_texts[i])
            slice_raw[slice_idx].append(all_raw_examples[i])
            slice_images[slice_idx].append(all_images[i])
            slice_prompts[slice_idx].append(all_prompts[i])
            slice_prompts_text[slice_idx].append(all_prompts_text[i])

        for slice_idx in on_policy_indices:
            completion_texts = slice_completions[slice_idx]
            raw_for_slice = slice_raw[slice_idx]
            images_for_slice = slice_images[slice_idx]
            prompts_for_slice = slice_prompts[slice_idx]

            # Build synthetic examples: original prompt + generated completion
            synthetic_examples = []
            for i, example in enumerate(raw_for_slice):
                synthetic = dict(example)
                # Wrap as content blocks so VLM chat templates (e.g. SmolVLM) that index
                # `message.content[0]` can render the synthetic assistant turn.
                synthetic["completion"] = [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": completion_texts[i]}],
                    }
                ]
                synthetic_examples.append(synthetic)

            has_images = any(img is not None for img in images_for_slice)
            pending_slice = {
                "_gold_vlm_lazy_examples": synthetic_examples,
            }
            if self._teacher_processor is not None:
                pending_slice["_gold_vlm_raw_images"] = images_for_slice if has_images else None
                pending_slice["_gold_vlm_raw_prompts"] = prompts_for_slice
            self._buffered_inputs[slice_idx] = pending_slice
            self._buffered_text_logs[slice_idx] = (
                slice_prompts_text[slice_idx],
                completion_texts,
            )

    def _process_completions_to_buffer(
        self,
        slices: list[dict[str, torch.Tensor | Any]],
        on_policy_indices: list[int],
        local_slice_indices: list[int],
        completion_ids: list,
        prompt_ids_list: list[list[int]],
        prompts_text: list[str],
        max_completion_length: int,
    ):
        """
        Process vLLM completions and update buffered inputs for on-policy slices.
        """
        device = self.accelerator.device
        pad_token_id = self.processing_class.pad_token_id if self.processing_class.pad_token_id is not None else 0

        slice_completions = {idx: [] for idx in on_policy_indices}
        slice_prompt_ids = {idx: [] for idx in on_policy_indices}
        slice_prompts = {idx: [] for idx in on_policy_indices}

        for i, slice_idx in enumerate(local_slice_indices):
            slice_completions[slice_idx].append(completion_ids[i])
            slice_prompt_ids[slice_idx].append(prompt_ids_list[i])
            slice_prompts[slice_idx].append(prompts_text[i])

        for slice_idx in on_policy_indices:
            slice_inputs = slices[slice_idx]
            completion_ids_for_slice = slice_completions[slice_idx]
            prompt_ids_for_slice = slice_prompt_ids[slice_idx]
            prompt_txts = slice_prompts[slice_idx]

            # Prompts are already budgeted to max_length - max_new_tokens before generation by the
            # callers, so no truncation happens here; just build tensors and left-pad.
            prompt_tensors = []
            prompt_attention_masks = []
            for prompt_ids in prompt_ids_for_slice:
                prompt_tensor = torch.tensor(prompt_ids, device=device, dtype=torch.long)
                prompt_tensors.append(prompt_tensor)
                prompt_attention_masks.append(torch.ones(len(prompt_ids), device=device, dtype=torch.long))

            prompt_ids = pad(prompt_tensors, padding_side="left", padding_value=pad_token_id)
            prompt_attention_mask = pad(prompt_attention_masks, padding_side="left", padding_value=0)

            # Decode the prompt (already budgeted by the callers) so the teacher conditions on the same
            # context the student saw. `clean_up_tokenization_spaces=False` matches the completion decode
            # below so byte counts stay aligned.
            prompt_txts_with_special = self.processing_class.batch_decode(
                [ids.tolist() for ids in prompt_tensors],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            completion_ids_tensors = [torch.tensor(ids, device=device) for ids in completion_ids_for_slice]
            completion_ids_for_text: list[list[int]] = []
            padded_completion_ids_list = []
            completion_attention_masks = []
            for completion_tensor in completion_ids_tensors:
                if len(completion_tensor) > max_completion_length:
                    truncated_completion_tensor = completion_tensor[:max_completion_length]
                    padded_completion_ids_list.append(truncated_completion_tensor)
                    completion_ids_for_text.append(truncated_completion_tensor.tolist())
                    completion_attention_masks.append(
                        torch.ones(
                            len(truncated_completion_tensor),
                            device=device,
                            dtype=torch.long,
                        )
                    )
                elif len(completion_tensor) < max_completion_length:
                    padding_needed = max_completion_length - len(completion_tensor)
                    padded_tensor = torch.cat(
                        [
                            completion_tensor,
                            torch.full(
                                (padding_needed,),
                                pad_token_id,
                                device=device,
                                dtype=completion_tensor.dtype,
                            ),
                        ]
                    )
                    padded_completion_ids_list.append(padded_tensor)
                    completion_ids_for_text.append(completion_tensor.tolist())
                    completion_attention_masks.append(
                        torch.cat(
                            [
                                torch.ones(
                                    len(completion_tensor),
                                    device=device,
                                    dtype=torch.long,
                                ),
                                torch.zeros(padding_needed, device=device, dtype=torch.long),
                            ]
                        )
                    )
                else:
                    padded_completion_ids_list.append(completion_tensor)
                    completion_ids_for_text.append(completion_tensor.tolist())
                    completion_attention_masks.append(
                        torch.ones(len(completion_tensor), device=device, dtype=torch.long)
                    )

            completion_ids_padded = torch.stack(padded_completion_ids_list)
            completion_attention_mask = torch.stack(completion_attention_masks)

            new_input_ids = torch.cat([prompt_ids, completion_ids_padded], dim=1)
            new_attention_mask = torch.cat([prompt_attention_mask, completion_attention_mask], dim=1)
            prompt_lengths = torch.full((prompt_ids.shape[0],), prompt_ids.shape[1], device=device)
            new_attention_mask, new_labels = self._build_sequence_batch(
                new_input_ids,
                prompt_lengths,
                attention_mask=new_attention_mask,
            )

            completion_texts = self.processing_class.batch_decode(
                completion_ids_for_text,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            updated_slice = dict(slice_inputs)
            updated_slice["input_ids"] = new_input_ids
            updated_slice["attention_mask"] = new_attention_mask
            updated_slice["labels"] = new_labels
            # Update prompts to match the new padding width so prompt_length stays consistent
            updated_slice["prompts"] = prompt_ids
            updated_slice["prompt_attention_mask"] = prompt_attention_mask
            updated_slice["original_prompt_text"] = prompt_txts_with_special
            updated_slice["original_completion_text"] = completion_texts
            self._maybe_add_completion_byte_offsets(updated_slice)

            self._buffered_inputs[slice_idx] = updated_slice
            self._buffered_text_logs[slice_idx] = (prompt_txts, completion_texts)

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: (PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin),
        args,
        packing: bool,
        formatting_func: Callable[[dict], str] | None,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        """Preserve original text fields for ULD when needed."""
        # For VLM datasets, skip dataset preparation entirely — the VLM collator handles tokenization
        # and image processing on the fly, similar to how SFTTrainer skips prep for vision datasets.
        if self._is_vision_dataset:
            return dataset

        column_names = list(next(iter(dataset)).keys())
        is_processed = "input_ids" in column_names

        if packing and self.use_uld_loss and self.teacher_tokenizer is not None:
            raise ValueError(
                "Packing is not supported with cross-tokenizer ULD because byte-offset alignment is defined per "
                "prompt/completion example."
            )

        if not is_processed or (self.use_uld_loss and self.teacher_tokenizer is not None):
            return self._prepare_dataset_with_original_text(
                dataset, processing_class, args, packing, formatting_func, dataset_name
            )

        return super()._prepare_dataset(dataset, processing_class, args, packing, formatting_func, dataset_name)

    def _prepare_dataset_with_original_text(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: (PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin),
        args,
        packing: bool,
        formatting_func: Callable[[dict], str] | None,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        """
        Prepare dataset while preserving original text for cross-tokenizer distillation.
        """
        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with global_then_local_main_first():
            # Apply the formatting function if any
            if formatting_func is not None:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

                def _func(example):
                    return {"text": formatting_func(example)}

                dataset = dataset.map(_func, batched=False, **map_kwargs)

            # Convert the dataset to ChatML if needed
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Converting {dataset_name} dataset to ChatML"
            column_names = next(iter(dataset)).keys()
            dataset = dataset.map(
                maybe_convert_to_chatml,
                remove_columns=("conversations" if "conversations" in column_names else None),
                **map_kwargs,
            )

            # Tokenize the dataset while preserving original text
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset (preserving original text)"

            def tokenize_with_original_text(
                example, processing_class, dataset_text_field, max_length, use_extended_uld
            ):
                """Emit input_ids, attention_mask, byte_offsets, completion_mask, and the original prompt/completion
                text.
                """
                result = {}

                if "prompt" in example:  # prompt-completion case
                    if is_conversational(example):
                        prompt_text = processing_class.apply_chat_template(
                            example["prompt"],
                            add_generation_prompt=True,
                            tokenize=False,
                            **example.get("chat_template_kwargs", {}),
                        )
                        full_text = processing_class.apply_chat_template(
                            example["prompt"] + example["completion"],
                            tokenize=False,
                            **example.get("chat_template_kwargs", {}),
                        )
                        prompt_text = "".join(
                            x
                            for x, _ in takewhile(
                                lambda x: x[0] == x[1],
                                zip(prompt_text, full_text, strict=False),
                            )
                        )
                        completion_text = full_text[len(prompt_text) :]
                    else:
                        prompt_text = example["prompt"]
                        completion_text = example["completion"]
                        full_text = prompt_text + completion_text
                    result["original_prompt_text"] = prompt_text
                    result["original_completion_text"] = completion_text
                elif is_conversational(example):
                    messages = example["messages"]
                    assistant_indices = [idx for idx, msg in enumerate(messages) if msg["role"] == "assistant"]
                    if assistant_indices:
                        completion_idx = assistant_indices[-1]
                        prompt_messages = messages[:completion_idx]
                        full_messages = messages[: completion_idx + 1]
                        if prompt_messages:
                            prompt_text = processing_class.apply_chat_template(
                                prompt_messages,
                                add_generation_prompt=True,
                                tokenize=False,
                                **example.get("chat_template_kwargs", {}),
                            )
                        else:
                            prompt_text = ""
                        full_text = processing_class.apply_chat_template(
                            full_messages,
                            add_generation_prompt=False,
                            tokenize=False,
                            **example.get("chat_template_kwargs", {}),
                        )
                        prompt_text = "".join(
                            x
                            for x, _ in takewhile(
                                lambda x: x[0] == x[1],
                                zip(prompt_text, full_text, strict=False),
                            )
                        )
                        completion_text = full_text[len(prompt_text) :]
                        result["original_prompt_text"] = prompt_text
                        result["original_completion_text"] = completion_text
                    else:
                        full_text = processing_class.apply_chat_template(
                            messages,
                            tokenize=False,
                            **example.get("chat_template_kwargs", {}),
                        )
                        prompt_text = ""
                        result["original_prompt_text"] = ""
                        result["original_completion_text"] = full_text
                else:
                    text = example.get(dataset_text_field, example.get("text", ""))
                    prompt_text = ""
                    completion_text = text
                    full_text = text
                    result["original_prompt_text"] = ""
                    result["original_completion_text"] = text

                if use_extended_uld:
                    # Single backend call: ids and char-derived byte offsets from the same encoding,
                    # so input_ids[i] is described by full_offs[i] without any boundary slop.
                    backend = processing_class.backend_tokenizer
                    [(input_ids, full_offs)] = encode_with_byte_offsets(backend, [full_text], add_special_tokens=False)
                    prompt_byte_len = len(prompt_text.encode("utf-8"))
                    # A token straddling the boundary is pulled into the completion here: extended ULD
                    # re-splits it against the teacher's own byte offsets via `_align_by_byte_offsets`.
                    completion_start = next(
                        (idx for idx, (_, e) in enumerate(full_offs) if e > prompt_byte_len),
                        len(input_ids),
                    )
                    # Completion-relative: prompt positions zeroed, completion offsets shifted to
                    # the assistant content's first byte (matches build_teacher_inputs_from_texts).
                    byte_offsets = [(0, 0)] * completion_start + [
                        (max(0, s - prompt_byte_len), e - prompt_byte_len) for s, e in full_offs[completion_start:]
                    ]
                else:
                    # Works with slow tokenizers too (e.g. SentencePiece): no `backend_tokenizer` needed.
                    encoding = processing_class(full_text, add_special_tokens=False, return_offsets_mapping=True)
                    input_ids = encoding["input_ids"]
                    # Same boundary rule as extended ULD: a leading space at the seam is normally attached to
                    # the following (completion) token by the tokenizer, so matching on `start` instead would
                    # drop that token's content from the completion entirely.
                    completion_start = next(
                        (idx for idx, (_, end) in enumerate(encoding["offset_mapping"]) if end > len(prompt_text)),
                        len(input_ids),
                    )
                    byte_offsets = [(0, 0)] * len(input_ids)

                append_eos = (
                    not is_conversational(example)
                    and processing_class.eos_token_id is not None
                    and (not input_ids or input_ids[-1] != processing_class.eos_token_id)
                )
                if append_eos:
                    input_ids.append(processing_class.eos_token_id)
                    if use_extended_uld:
                        completion_byte_len = len(completion_text.encode("utf-8"))
                        byte_offsets.append((completion_byte_len, completion_byte_len))
                    else:
                        byte_offsets.append((0, 0))

                # Keep the last `max_length` tokens (the completion end). `completion_mask` tracks the
                # boundary so it survives truncation without re-tokenizing the prompt.
                if max_length is not None and len(input_ids) > max_length:
                    drop = len(input_ids) - max_length
                    input_ids = input_ids[drop:]
                    byte_offsets = byte_offsets[drop:]
                    completion_start = max(0, completion_start - drop)
                    # If truncation ate into the completion, rebase the kept completion offsets so they're
                    # relative to the new (truncated) `original_completion_text` the teacher will re-encode.
                    if completion_start < len(byte_offsets):
                        base = byte_offsets[completion_start][0]
                        if base > 0:
                            byte_offsets = byte_offsets[:completion_start] + [
                                (s - base, e - base) for s, e in byte_offsets[completion_start:]
                            ]
                    # Resync the strings the teacher will re-encode with the ids the student kept.
                    decode = partial(
                        processing_class.decode,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    result["original_prompt_text"] = decode(input_ids[:completion_start])
                    completion_ids = input_ids[completion_start:]
                    if append_eos:
                        completion_ids = completion_ids[:-1]
                    result["original_completion_text"] = decode(completion_ids)

                result["input_ids"] = input_ids
                result["attention_mask"] = [1] * len(input_ids)
                result["byte_offsets"] = byte_offsets
                result["completion_mask"] = [0] * completion_start + [1] * (len(input_ids) - completion_start)
                return result

            dataset = dataset.map(
                tokenize_with_original_text,
                fn_kwargs={
                    "processing_class": processing_class,
                    "dataset_text_field": args.dataset_text_field,
                    "max_length": args.max_length,
                    "use_extended_uld": args.use_extended_uld,
                },
                **map_kwargs,
            )

            # Pack if requested. Truncation already happened in `tokenize_with_original_text`, keeping
            # the completion end — so the generic front-truncating `truncate_dataset` is intentionally
            # not applied here (it would drop the completion).
            if packing:
                if args.max_length is None:
                    raise ValueError("When packing is enabled, `max_length` can't be `None`.")
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Packing {dataset_name} dataset"

                columns_to_keep = [
                    "input_ids",
                    "original_prompt_text",
                    "original_completion_text",
                ]
                existing_columns = set(dataset.column_names)
                columns_to_select = [col for col in columns_to_keep if col in existing_columns]

                dataset = dataset.select_columns(columns_to_select)
                dataset = pack_dataset(dataset, args.max_length, args.packing_strategy, map_kwargs)

        return dataset

    @staticmethod
    def generalized_jsd_loss(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        reduction="batchmean",
        logits_are_probs=False,
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

        if logits_are_probs:
            student_log_probs = torch.log(student_logits.clamp_min(1e-8))
            teacher_log_probs = torch.log(teacher_logits.clamp_min(1e-8))
        else:
            # Apply temperature scaling to logits before computing probabilities
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
                torch.stack(
                    [
                        student_log_probs + torch.log1p(-beta),
                        teacher_log_probs + torch.log(beta),
                    ]
                ),
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

    def _build_teacher_vlm_inputs(
        self, completion_texts: list[str], raw_images: list, raw_prompts: list
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Build image-aware teacher inputs for cross-architecture VLM ULD distillation.

        The teacher prompt is rendered through the teacher's own processor so image placeholders and pixel tensors
        match the teacher model, while completion tokens carry byte offsets relative to the original
        ``completion_texts`` — the same coordinate system the student uses — so cross-tokenizer byte-offset alignment
        stays valid. Mirrors ``build_teacher_inputs_from_texts`` but injects the teacher's multimodal prompt and
        returns the teacher's forward kwargs (``pixel_values``, ...).

        Returns ``(input_ids, labels, attention_mask, byte_offsets, forward_kwargs)``.
        """
        pad_token_id = self.teacher_tokenizer.pad_token_id
        eos_token_id = self.teacher_tokenizer.eos_token_id

        teacher_prompt_texts = self._teacher_processor.apply_chat_template(
            raw_prompts, tokenize=False, add_generation_prompt=True
        )
        teacher_prompt_processed = self._teacher_processor(
            images=raw_images,
            text=teacher_prompt_texts,
            padding=True,
            return_tensors="pt",
        )
        if self.uld_loss_fn.use_extended_uld:
            # Only the extended path needs the fast tokenizer's byte offsets; positional ULD works with slow
            # (e.g. SentencePiece) tokenizers that have no `backend_tokenizer`.
            backend = self.teacher_tokenizer.backend_tokenizer
            completion_encs = encode_with_byte_offsets(backend, completion_texts, add_special_tokens=False)
        else:
            completion_ids = self.teacher_tokenizer(completion_texts, add_special_tokens=False)["input_ids"]
            completion_encs = [(ids, [(0, 0)] * len(ids)) for ids in completion_ids]

        sequences: list[torch.Tensor] = []
        attention_masks: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        offsets_list: list[list[tuple[int, int]]] = []
        sequence_kwargs = defaultdict(list)

        for row, ((enc_ids, enc_offs), completion_text) in enumerate(
            zip(completion_encs, completion_texts, strict=True)
        ):
            prompt_mask = teacher_prompt_processed["attention_mask"][row].bool()
            prompt_ids = teacher_prompt_processed["input_ids"][row][prompt_mask].tolist()
            # Remove trailing EOS from prompt so completions can extend cleanly
            if eos_token_id is not None and prompt_ids and prompt_ids[-1] == eos_token_id:
                prompt_ids = prompt_ids[:-1]

            completion_ids = list(enc_ids)
            completion_offs = list(enc_offs)
            content_len = len(completion_text.encode("utf-8"))

            sequence = list(prompt_ids) + completion_ids
            offsets = [(0, 0)] * len(prompt_ids) + completion_offs
            if eos_token_id is not None:
                sequence.append(eos_token_id)
                offsets.append((content_len, content_len) if self.uld_loss_fn.use_extended_uld else (0, 0))

            seq_tensor = torch.tensor(sequence, dtype=torch.long)
            sequences.append(seq_tensor)
            attention_masks.append(torch.ones_like(seq_tensor))
            offsets_list.append(offsets)

            labels = seq_tensor.clone()
            labels[: len(prompt_ids)] = -100
            labels_list.append(labels)

            # Sequence-aligned multimodal keys (e.g. token_type_ids for Gemma): keep the prompt values
            # and zero-fill the completion span.
            for key in self._SEQUENCE_KEYS:
                if key in teacher_prompt_processed:
                    prompt_values = teacher_prompt_processed[key][row][prompt_mask][: len(prompt_ids)]
                    completion_values = torch.zeros(len(sequence) - len(prompt_ids), dtype=prompt_values.dtype)
                    sequence_kwargs[key].append(torch.cat((prompt_values, completion_values)))

        teacher_input_ids = pad(
            sequences,
            padding_side="right",
            padding_value=pad_token_id if pad_token_id is not None else 0,
        )
        teacher_attention_mask = pad(attention_masks, padding_side="right", padding_value=0).bool()
        teacher_labels = pad(labels_list, padding_side="right", padding_value=-100)

        target_len = teacher_input_ids.size(1)
        teacher_byte_offsets = torch.stack(
            [pad_byte_offsets(offs, target_len, padding_side="right") for offs in offsets_list],
            dim=0,
        )

        # Multimodal forward kwargs from the teacher processor (pixel_values, image_grid_thw, ...).
        forward_kwargs = {
            k: v.to(self.accelerator.device)
            for k, v in self._get_model_forward_kwargs(teacher_prompt_processed, exclude=self._SEQUENCE_KEYS).items()
        }
        for key, values in sequence_kwargs.items():
            forward_kwargs[key] = pad(values, padding_side="right", padding_value=0).to(self.accelerator.device)

        return (
            teacher_input_ids,
            teacher_labels,
            teacher_attention_mask,
            teacher_byte_offsets,
            forward_kwargs,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract multimodal fields (pixel_values, image_grid_thw, ...) for student forward passes.
        # Standard JSD reuses these for the teacher (same-family VLM); cross-tokenizer ULD rebuilds
        # teacher inputs separately below.
        student_forward_kwargs = self._get_model_forward_kwargs(inputs)
        if self.use_uld_loss and self.teacher_tokenizer is not None:
            # Both DataCollatorForChatML and the on-policy generation path attach these
            # fields, so cross-tokenizer ULD never has to round-trip through batch_decode.
            prompt_texts = inputs["original_prompt_text"]
            completion_texts = inputs["original_completion_text"]

            if self._teacher_processor is not None:
                # VLM teacher: render the prompt through the teacher's own processor so image
                # placeholders and pixel tensors match the teacher model.
                if "_raw_images" not in inputs or "_raw_prompts" not in inputs:
                    raise ValueError(
                        "VLM ULD distillation requires `_raw_images` and `_raw_prompts` in the batch so teacher "
                        "inputs can be rendered with the teacher processor. Use the default GOLD VLM data collator "
                        "or ensure your custom collator preserves these fields."
                    )
                (
                    teacher_input_ids,
                    teacher_labels,
                    teacher_attention_mask,
                    teacher_completion_byte_offsets,
                    teacher_forward_kwargs,
                ) = self._build_teacher_vlm_inputs(completion_texts, inputs["_raw_images"], inputs["_raw_prompts"])
            else:
                teacher_forward_kwargs = {}
                (
                    teacher_input_ids,
                    teacher_labels,
                    teacher_attention_mask,
                    teacher_completion_byte_offsets,
                ) = build_teacher_inputs_from_texts(
                    self.teacher_tokenizer,
                    prompt_texts,
                    completion_texts,
                    use_extended_uld=self.uld_loss_fn.use_extended_uld,
                )

            teacher_input_ids = teacher_input_ids.to(self.accelerator.device)
            teacher_labels = teacher_labels.to(self.accelerator.device)
            teacher_attention_mask = teacher_attention_mask.to(self.accelerator.device)

            outputs_student = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                **student_forward_kwargs,
            )

            self.teacher_model.eval()
            with torch.no_grad():
                outputs_teacher = self.teacher_model(
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_attention_mask,
                    **teacher_forward_kwargs,
                )
        else:
            outputs_student = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **student_forward_kwargs,
            )

            self.teacher_model.eval()
            with torch.no_grad():
                outputs_teacher = self.teacher_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **student_forward_kwargs,
                )

            # Standard causal shift: logits at position i predict the token at i + 1. The `labels != -100` mask
            # inside `generalized_jsd_loss` already excludes prompt (and padding) positions, so we do not slice by
            # prompt length. Slicing by `inputs["prompts"].shape[1]` (the batch-max prompt width) would drop real
            # completion tokens for samples whose prompt is shorter than the batch maximum, since `labels` is
            # padded to the full-sequence width independently of `prompts`.
            shifted_student_logits = outputs_student.logits[:, :-1, :]
            shifted_teacher_logits = outputs_teacher.logits[:, :-1, :]
            shifted_labels = inputs["labels"][:, 1:]
            loss = self.generalized_jsd_loss(
                student_logits=shifted_student_logits,
                teacher_logits=shifted_teacher_logits,
                labels=shifted_labels,
                beta=self.beta,
                temperature=self.temperature,
                num_items_in_batch=num_items_in_batch,
            )

        if self.use_uld_loss and self.teacher_tokenizer is not None:
            student_labels = inputs["labels"]

            student_byte_offsets = inputs.get("byte_offsets")
            if self.uld_loss_fn.use_extended_uld and student_byte_offsets is None:
                raise ValueError("Input batches must include `byte_offsets` when `use_extended_uld=True`.")

            loss = self.uld_loss_fn(
                student_logits=outputs_student.logits,
                teacher_logits=outputs_teacher.logits,
                student_labels=student_labels,
                teacher_labels=teacher_labels,
                student_input_ids=inputs["input_ids"],
                teacher_input_ids=teacher_input_ids,
                student_byte_offsets=student_byte_offsets,
                teacher_byte_offsets=teacher_completion_byte_offsets,
            )

            if hasattr(self.uld_loss_fn, "last_matched_loss") and hasattr(self.uld_loss_fn, "last_unmatched_loss"):
                ga = max(1, int(self.args.gradient_accumulation_steps))
                step_eq = 1.0 / ga
                matched_val = (
                    self.uld_loss_fn.last_matched_loss.item()
                    if self.uld_loss_fn.last_matched_loss is not None
                    else 0.0
                )
                unmatched_val = (
                    self.uld_loss_fn.last_unmatched_loss.item()
                    if self.uld_loss_fn.last_unmatched_loss is not None
                    else 0.0
                )

                self._matched_sum += matched_val
                self._unmatched_sum += unmatched_val
                self._matched_step_eq += step_eq
                self._unmatched_step_eq += step_eq

        empty_cache()

        return (loss, outputs_student) if return_outputs else loss

    def generate_on_policy_outputs(self, model, inputs, generation_config):
        # Generate output with respect to the prompt only
        # Drop sequence-aligned multimodal keys (token_type_ids, mm_token_type_ids): generate recomputes them from
        # `input_ids` as it extends the sequence, and some models (e.g. Qwen3-VL) reject them as unknown
        # `generate` kwargs.
        generate_kwargs = self._get_model_forward_kwargs(inputs, exclude=self._SEQUENCE_KEYS)
        generated_outputs = model.generate(
            input_ids=inputs["prompts"],
            attention_mask=inputs["prompt_attention_mask"],
            generation_config=generation_config,
            return_dict_in_generate=True,
            **generate_kwargs,
        )
        # Get the generated token IDs
        generated_tokens = generated_outputs.sequences

        batch_size = generated_tokens.size(0)
        device = generated_tokens.device

        prompt_mask = inputs["prompt_attention_mask"]

        # model.generate() returns full sequences (prompt + completion), so completions start
        # after the full padded prompt width.
        prompt_length = inputs["prompts"].shape[1]
        prompt_lengths = torch.full((batch_size,), prompt_length, dtype=torch.long, device=device)
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

        new_input_ids = generated_tokens
        new_attention_mask, new_labels = self._build_sequence_batch(
            new_input_ids, prompt_lengths, attention_mask=new_attention_mask
        )

        prompt_texts = []
        completion_texts = []
        for idx in range(batch_size):
            length = int(prompt_lengths[idx].item())
            prompt_tokens = inputs["prompts"][idx]
            prompt_tokens = prompt_tokens[prompt_mask[idx].bool()]
            prompt_texts.append(
                self.processing_class.decode(
                    prompt_tokens.tolist(),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
            completion_tokens = new_input_ids[idx, length:]
            completion_tokens = completion_tokens[new_labels[idx, length:] != -100]
            completion_texts.append(
                self.processing_class.decode(
                    completion_tokens.tolist(),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )

        return (
            new_input_ids,
            new_attention_mask,
            new_labels,
            prompt_texts,
            completion_texts,
        )

    # During eval, Trainer calls prediction_step. The inherited SFT prediction_step indexes the raw inputs before
    # collation, which breaks the VLM identity-collator path (inputs is a list of raw examples). We override it to
    # collate via _prepare_inputs and force compute_loss, evaluating the off-policy distillation loss over the
    # ground-truth completion (no generation).
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: list[str] | None = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    @profiling_decorator
    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        """
        Perform a training step for the General Online Logit Distillation (GOLD) model.

        This method implements the on-policy learning approach described in the GOLD blog post. With probability
        `self.lmbda`, it generates new responses using the student model, which are then used for training instead of
        the offline original inputs.
        """
        buffer_steps = self.args.gradient_accumulation_steps

        loss = super().training_step(model, inputs, num_items_in_batch)

        slice_idx = (self._step - 1) % buffer_steps

        on_policy = False
        if self._buffered_on_policy is not None and slice_idx < len(self._buffered_on_policy):
            on_policy = self._buffered_on_policy[slice_idx]

        if on_policy and self._buffered_text_logs is not None and self._buffered_text_logs[slice_idx] is not None:
            prompt_texts, completion_texts = self._buffered_text_logs[slice_idx]
            self._textual_logs["prompt"].extend(gather_object(prompt_texts))
            self._textual_logs["completion"].extend(gather_object(completion_texts))

        loss_scalar = float(loss.detach())
        step_equiv = 1.0 / self.args.gradient_accumulation_steps

        if on_policy:
            self._on_policy_loss_total += loss_scalar
            self._on_policy_step_equiv += step_equiv
        else:
            self._off_policy_loss_total += loss_scalar
            self._off_policy_step_equiv += step_equiv
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        if mode == "train":
            device = self.accelerator.device if hasattr(self.accelerator, "device") else torch.device("cpu")
            vec = torch.tensor(
                [
                    self._on_policy_loss_total,
                    self._off_policy_loss_total,
                    self._on_policy_step_equiv,
                    self._off_policy_step_equiv,
                    self._matched_sum,
                    self._unmatched_sum,
                    self._matched_step_eq,
                    self._unmatched_step_eq,
                ],
                dtype=torch.float32,
                device=device,
            )

            if (
                getattr(self.accelerator, "distributed_type", DistributedType.NO) != DistributedType.NO
                and dist.is_available()
                and dist.is_initialized()
            ):
                dist.all_reduce(vec, op=dist.ReduceOp.SUM)

            (
                on_sum,
                off_sum,
                on_eq,
                off_eq,
                matched_sum,
                unmatched_sum,
                matched_eq,
                unmatched_eq,
            ) = vec.tolist()

            if on_eq > 0:
                logs["on_policy_loss"] = round(on_sum / on_eq, 4)
            if off_eq > 0:
                logs["off_policy_loss"] = round(off_sum / off_eq, 4)

            if matched_eq > 0:
                logs["matched_loss"] = round(matched_sum / matched_eq, 4)
            if unmatched_eq > 0:
                logs["unmatched_loss"] = round(unmatched_sum / unmatched_eq, 4)

            self._on_policy_loss_total = self._off_policy_loss_total = 0.0
            self._on_policy_step_equiv = self._off_policy_step_equiv = 0.0
            self._matched_sum = self._unmatched_sum = 0.0
            self._matched_step_eq = self._unmatched_step_eq = 0.0

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs.update(metrics)
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if (
            self.accelerator.is_main_process
            and self.log_completions
            and ((self.state.global_step % self.log_completion_steps) == 0)
        ):
            if is_rich_available():
                print_prompt_completions_sample_uld(
                    self._textual_logs["prompt"],
                    self._textual_logs["completion"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [self.state.global_step] * len(self._textual_logs["prompt"]),
                    "prompt": self._textual_logs["prompt"],
                    "completion": self._textual_logs["completion"],
                }
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                if self.num_completions_to_print and len(df) > 0:
                    df = df.sample(n=self.num_completions_to_print, random_state=42)
                wandb.log({"completions": wandb.Table(dataframe=df)})
class UnslothGOLDTrainer(_UnslothGOLDTrainer):
    """
    
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
        **kwargs
    ):
        if args is None: args = UnslothGOLDConfig()
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
        PatchRLStatistics('gold_trainer', other_metrics)
        
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
