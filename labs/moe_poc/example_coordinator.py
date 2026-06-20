"""
ZynkCluster Coordinator (Illustrative Implementation)

Orchestrates distributed inference across multiple expert nodes. This is the
"brain" of ZynkCluster that decides which experts to activate and combines
their outputs.

This is ILLUSTRATIVE ONLY - demonstrates the architecture without being
executable on current hardware.

Deployment Model: Expert-level MoE distribution over consumer LAN devices,
using ZynkSync device-pairing infrastructure as the coordination layer.

MoE parallel execution is an established model property (see Mixtral paper).
The contribution here is the deployment architecture: mapping experts onto
separate consumer devices over WiFi/LAN, rather than running on one machine
or requiring datacenter interconnects (as DeepSpeed-MoE does).

Status: Non-functional example (design phase)
Hardware Required: 2+ GPU-enabled devices running expert node servers
Purpose: Demonstrate expert-level distribution over local network

Author: Matt Skillman
Date: January 2026
License: See main repository
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import requests
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
from typing import List, Dict, Tuple

# Hugging Face for tokenization and model utilities
from transformers import AutoTokenizer

# ZynkSync integration (would be imported from main codebase)
# from zynk_sync import get_paired_devices

# ==============================================================================
# CONFIGURATION
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CLUSTER COORDINATOR CLASS
# ==============================================================================

class ZynkClusterCoordinator:
    """
    Coordinates distributed MoE inference across multiple devices.

    MoE models like Mixtral already activate only k experts per token per layer
    (k=2 for Mixtral 8x7B — fixed by training). ZynkCluster maps those experts
    onto separate devices so they execute in parallel over the local network,
    rather than routing everything through one machine.

    Compare with Petals (layer-based, sequential):
    ----------------------------------------------
    1. Send tokens to Device A → Process layers 0-10 → Get output
    2. Send output to Device B → Process layers 11-20 → Get output
    3. Send output to Device C → Process layers 21-30 → Get output
    Total time = Time(A) + Time(B) + Time(C) + 3× network latency
    All devices involved for every token.

    ZynkCluster (expert-based, parallel over LAN):
    -----------------------------------------------
    1. Router: "This query needs Expert 2 + Expert 6" (fixed by model)
    2. Send hidden states to Device A (Expert 2) | SIMULTANEOUS over LAN
       Send hidden states to Device B (Expert 6) |
    3. Wait for both to complete (takes max(A, B) time, not sum)
    4. Combine outputs
    Total time = max(Time(Expert 2), Time(Expert 6)) + 1× network latency
    Only devices hosting the selected experts are involved.
    """

    def __init__(self, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", top_k_experts=2):
        """
        Initialize the cluster coordinator.

        Parameters:
        -----------
        model_name : str
            HuggingFace model identifier
            Must match the model loaded on expert nodes
        top_k_experts : int
            Number of experts to activate per token per layer.
            Must match the model's trained routing config — not tunable without
            retraining. ZynkCluster is not Mixtral-specific; any open MoE model works.
            Known values: Mixtral 8x7B/8x22B: k=2, DBRX: k=4, OLMoE: k=8,
            DeepSeek-V2: k=6, DeepSeek-V3: k=8, Grok-1: k=2.
            Optimal cluster size = k (one device per active expert).
        """
        self.model_name = model_name
        self.top_k_experts = top_k_experts  # Model-architecture-dependent — not tunable:
                                             # Mixtral 8x7B/8x22B: k=2
                                             # DBRX: k=4, OLMoE: k=8
                                             # DeepSeek-V2: k=6, DeepSeek-V3: k=8
                                             # Must match the model's trained routing config
        self.nodes = {}  # {device_id: {ip, expert_ids, status}}
        self.router = None  # Would load lightweight router model (~100MB)
        self.tokenizer = None

        logger.info("=" * 70)
        logger.info("ZynkCluster Coordinator Initializing")
        logger.info("=" * 70)
        logger.info(f"Model: {model_name}")

        # Load tokenizer (small, can run on coordinator)
        logger.info("Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("✅ Tokenizer loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load tokenizer: {e}")
            raise

        # In production, would also load router model here
        # self.router = load_router_model(model_name)
        logger.info("Router model: [Placeholder - would load ~100MB router]")

        logger.info("=" * 70)

    def discover_cluster_via_zynksync(self):
        """
        Integration with ZynkSync device discovery.

        This demonstrates how ZynkCluster leverages existing Zynkbot infrastructure.

        In production, this would:
        --------------------------
        1. Query zynk_devices table for paired devices
        2. Filter for devices with moe_capable=True
        3. Query each device's /expert/list endpoint
        4. Build cluster map: {device_id: {ip, expert_ids, gpu_memory, etc}}

        SQL Query (Pseudocode):
        -----------------------
        SELECT device_id, device_ip, hosted_experts, gpu_memory_gb
        FROM zynk_devices
        WHERE owner_user_id = ? AND moe_capable = TRUE AND status = 'online'

        For each device:
        ----------------
        GET http://{device_ip}:5001/expert/list
        → Returns: {device_id, hosted_experts, model_name, status}

        Key Insight:
        ------------
        The same device pairing infrastructure built for ZynkSync (memory sync)
        is reused for compute clustering. No new networking stack needed!
        """

        logger.info("🔍 Discovering cluster via ZynkSync...")

        # Simulated cluster discovery
        # In production, would query actual ZynkSync database + APIs
        self.nodes = {
            'device_a_rtx3090': {
                'ip': '192.168.1.100',
                'port': 5001,
                'expert_ids': [0, 1, 2, 3],
                'status': 'online',
                'gpu_memory_gb': 24,
                'model_name': self.model_name
            },
            'device_b_rtx3060': {
                'ip': '192.168.1.101',
                'port': 5001,
                'expert_ids': [4, 5, 6, 7],
                'status': 'online',
                'gpu_memory_gb': 12,
                'model_name': self.model_name
            }
        }

        logger.info(f"✅ Discovered {len(self.nodes)} devices:")
        for device_id, info in self.nodes.items():
            logger.info(f"   • {device_id}: Experts {info['expert_ids']} "
                       f"({info['gpu_memory_gb']}GB GPU)")

        logger.info("")
        return self.nodes

    def execute_distributed_layer(self, layer_idx, hidden_states):
        """
        Executes one MoE layer distributed across cluster devices.

        Standard MoE layer processing (what the model already does):
        -------------------------------------------------------------
        1. Router scores all experts for this input
        2. Select top-k experts (k fixed by model architecture)
        3. Execute selected experts
        4. Combine outputs with weighted sum

        ZynkCluster's role:
        -------------------
        Step 3 happens across separate LAN devices simultaneously rather than
        on a single machine. The model's routing logic is unchanged — we just
        dispatch the work to wherever each expert lives on the network.

        Parameters:
        -----------
        layer_idx : int
            Which transformer layer to process
            Example: 0 (first layer), 15 (middle layer)

        hidden_states : torch.Tensor
            Input hidden states from previous layer
            Shape: [batch_size, sequence_length, hidden_size]
            Example: [1, 128, 4096] for Mixtral

        Returns:
        --------
        torch.Tensor : Combined expert outputs
            Same shape as input: [batch_size, sequence_length, hidden_size]
        """

        logger.info(f"\n{'='*70}")
        logger.info(f"Processing Layer {layer_idx} with Distributed Experts")
        logger.info(f"{'='*70}")

        # ==================================================================
        # STEP 1: ROUTER SELECTION (Sparse Activation)
        # ==================================================================

        logger.info(f"[Step 1] Router selecting experts...")

        # In production, this would be actual router neural network (~100MB model)
        # Router learns which experts are best for each type of input
        # For demonstration, we simulate with random selection

        # router_logits = self.router(hidden_states)  # Production code
        router_logits = torch.rand(8)  # Simulated: 8 experts, random scores

        # Select top-k experts — k is determined by the model's trained architecture,
        # NOT a tunable parameter. Changing k without retraining produces wrong outputs
        # because the model's weights expect exactly k expert outputs to combine.
        #
        # Production note: if the same expert is replicated across multiple devices
        # (e.g., Expert 2 on both Device A and Device C), latency-aware routing would
        # select whichever copy has the lowest current response time. That's a device
        # selection optimization on top of this expert selection step.
        top_k = torch.topk(router_logits, k=self.top_k_experts)
        expert_ids = top_k.indices.tolist()
        expert_weights = top_k.values.tolist()

        logger.info(f"[Router] Selected Experts: {expert_ids}")
        logger.info(f"[Router] Weights: {[f'{w:.3f}' for w in expert_weights]}")

        # This is SPARSE ACTIVATION: Only top_k/total experts needed
        # Compare to Petals: ALL layers needed sequentially
        total_experts = len(router_logits)
        logger.info(f"[Router] Sparse activation: {self.top_k_experts}/{total_experts} experts "
                   f"({100*self.top_k_experts//total_experts}% active)")

        # ==================================================================
        # STEP 2: MAP EXPERTS TO DEVICES
        # ==================================================================

        logger.info(f"\n[Step 2] Mapping experts to devices...")

        execution_plan = []
        for expert_id in expert_ids:
            device_id = self._find_device_hosting_expert(expert_id)
            device_info = self.nodes[device_id]

            execution_plan.append({
                'expert_id': expert_id,
                'device_id': device_id,
                'device_ip': device_info['ip'],
                'device_port': device_info['port']
            })

            logger.info(f"[Map] Expert {expert_id} → {device_id} "
                       f"({device_info['ip']}:{device_info['port']})")

        # ==================================================================
        # STEP 3: PARALLEL EXECUTION
        # ==================================================================
        # THIS IS THE PROOF OF CONCEPT FOR PARALLEL EXECUTION
        # ==================================================================

        logger.info(f"\n[Step 3] Executing experts IN PARALLEL...")
        logger.info(f"{'─'*70}")

        def execute_remote_expert(plan):
            """
            Send an expert execution request to a remote LAN device and return
            its output. One instance of this runs per selected expert, all
            dispatched simultaneously via ThreadPoolExecutor.
            """
            expert_id = plan['expert_id']
            device_id = plan['device_id']
            device_ip = plan['device_ip']
            device_port = plan['device_port']

            url = f"http://{device_ip}:{device_port}/expert/execute"

            logger.info(f"[{device_id}] Sending Expert {expert_id} request...")

            try:
                start_time = time.time()

                # Send HTTP POST request to expert node
                response = requests.post(
                    url,
                    json={
                        'layer_idx': layer_idx,
                        'expert_id': expert_id,
                        'hidden_states': hidden_states.tolist()  # Serialize tensor
                    },
                    timeout=60.0  # 60 second timeout
                )

                elapsed = time.time() - start_time

                if response.status_code == 200:
                    result = response.json()
                    output_tensor = torch.tensor(result['output'])

                    logger.info(f"[{device_id}] ✅ Expert {expert_id} completed "
                               f"in {elapsed*1000:.1f}ms (reported: {result['elapsed_ms']:.1f}ms)")

                    return {
                        'expert_id': expert_id,
                        'device_id': device_id,
                        'output': output_tensor,
                        'elapsed_ms': elapsed * 1000,
                        'success': True
                    }
                else:
                    logger.error(f"[{device_id}] ❌ Expert {expert_id} failed: "
                                f"HTTP {response.status_code}")
                    return {
                        'expert_id': expert_id,
                        'device_id': device_id,
                        'success': False,
                        'error': f"HTTP {response.status_code}"
                    }

            except requests.exceptions.Timeout:
                elapsed = time.time() - start_time
                logger.error(f"[{device_id}] ⏱️  Expert {expert_id} timed out "
                            f"after {elapsed:.1f}s")
                return {
                    'expert_id': expert_id,
                    'device_id': device_id,
                    'success': False,
                    'error': 'Timeout'
                }

            except Exception as e:
                logger.error(f"[{device_id}] ❌ Expert {expert_id} error: {e}")
                return {
                    'expert_id': expert_id,
                    'device_id': device_id,
                    'success': False,
                    'error': str(e)
                }

        # Dispatch all expert requests simultaneously over LAN
        # The parallelism comes from the model's MoE structure (experts are independent);
        # ZynkCluster's role is routing those requests to the right network devices.
        parallel_start = time.time()

        with ThreadPoolExecutor(max_workers=len(execution_plan)) as executor:
            # Submit all expert execution tasks simultaneously
            futures = [
                executor.submit(execute_remote_expert, plan)
                for plan in execution_plan
            ]

            # Wait for all to complete
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        parallel_elapsed = (time.time() - parallel_start) * 1000

        # ==================================================================
        # STEP 4: ANALYZE PARALLELISM PROOF
        # ==================================================================

        logger.info(f"\n{'='*70}")
        logger.info(f"DISTRIBUTED EXECUTION RESULTS")
        logger.info(f"{'='*70}")

        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]

        if failed_results:
            logger.error(f"❌ {len(failed_results)} expert(s) failed:")
            for r in failed_results:
                logger.error(f"   • Expert {r['expert_id']} on {r['device_id']}: "
                           f"{r['error']}")

        if not successful_results:
            raise RuntimeError("All expert executions failed")

        # Calculate timing analysis
        individual_times = [r['elapsed_ms'] for r in successful_results]
        sequential_time = sum(individual_times)
        max_time = max(individual_times)

        logger.info(f"\n📊 Timing Analysis:")
        logger.info(f"{'─'*70}")
        for result in successful_results:
            logger.info(f"   Expert {result['expert_id']} ({result['device_id']}): "
                       f"{result['elapsed_ms']:.1f}ms")

        logger.info(f"{'─'*70}")
        logger.info(f"   Actual wall-clock time: {parallel_elapsed:.1f}ms")
        logger.info(f"   Sequential would take: {sequential_time:.1f}ms")
        logger.info(f"   Speedup: {sequential_time/parallel_elapsed:.2f}x")
        logger.info(f"{'─'*70}")

        # Theoretical vs actual
        logger.info(f"\n💡 Analysis:")
        logger.info(f"   Theoretical minimum: {max_time:.1f}ms (slowest expert)")
        logger.info(f"   Actual parallel time: {parallel_elapsed:.1f}ms")
        logger.info(f"   Overhead: {parallel_elapsed - max_time:.1f}ms "
                   f"(network latency, thread coordination)")

        if parallel_elapsed < sequential_time * 0.8:
            logger.info(f"\n✅ SUCCESS: Parallel execution is {sequential_time/parallel_elapsed:.1f}x "
                       f"faster than sequential!")
        else:
            logger.warning(f"\n⚠️  Parallel execution not significantly faster - "
                          f"check network latency")

        logger.info(f"{'='*70}\n")

        # ==================================================================
        # STEP 5: COMBINE EXPERT OUTPUTS
        # ==================================================================

        logger.info(f"[Step 4] Combining expert outputs...")

        # Weighted sum of expert outputs
        # This is standard MoE combination strategy
        combined_output = torch.zeros_like(hidden_states)

        for result, weight in zip(successful_results, expert_weights[:len(successful_results)]):
            combined_output += weight * result['output']

        logger.info(f"[Combine] ✅ Outputs combined with weighted sum")
        logger.info(f"{'='*70}\n")

        return combined_output

    def _find_device_hosting_expert(self, expert_id):
        """
        Find which device hosts a specific expert.

        Parameters:
        -----------
        expert_id : int
            Expert index (0-7 for Mixtral 8x7B)

        Returns:
        --------
        str : Device ID that hosts this expert

        Raises:
        -------
        ValueError : If no device hosts this expert
        """
        for device_id, info in self.nodes.items():
            if expert_id in info['expert_ids']:
                return device_id

        raise ValueError(f"No device hosts expert {expert_id}. "
                        f"Cluster configuration error!")

    def generate(self, prompt, max_tokens=50):
        """
        Generate text using distributed inference.

        This demonstrates end-to-end generation with ZynkCluster.

        In production, this would:
        --------------------------
        1. Tokenize prompt
        2. For each token to generate:
           a. Embed current tokens
           b. Process through distributed MoE layers
           c. Generate next token
           d. Append to sequence
        3. Return generated text

        Parameters:
        -----------
        prompt : str
            Input text
        max_tokens : int
            Maximum tokens to generate

        Returns:
        --------
        str : Generated text
        """

        logger.info("\n" + "="*70)
        logger.info("DISTRIBUTED GENERATION")
        logger.info("="*70)
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Max tokens: {max_tokens}")
        logger.info("="*70 + "\n")

        # Tokenize
        logger.info("[Tokenize] Converting text to tokens...")
        tokens = self.tokenizer.encode(prompt, return_tensors='pt')
        logger.info(f"[Tokenize] Input tokens: {tokens.shape[1]}")

        # Simplified generation loop (production would be more complex)
        logger.info(f"\n[Generate] Processing with distributed experts...\n")

        for i in range(max_tokens):
            # Create dummy hidden states (production would use actual embeddings)
            batch_size = 1
            seq_len = tokens.shape[1] + i
            hidden_size = 4096  # Mixtral hidden size; other models differ (e.g., DBRX: 6144, OLMoE: 2048)
            hidden_states = torch.randn(batch_size, seq_len, hidden_size)

            # Process through one distributed layer (demonstration)
            logger.info(f"[Token {i+1}/{max_tokens}] Processing layer 0...")
            output_states = self.execute_distributed_layer(
                layer_idx=0,
                hidden_states=hidden_states
            )

            logger.info(f"[Token {i+1}/{max_tokens}] ✅ Generated\n")

            # In production, would:
            # - Process through all 32 layers
            # - Apply final layer norm
            # - Project to vocabulary
            # - Sample next token
            # - Append to sequence

        generated_text = f"[Simulated generation - {max_tokens} tokens produced]"
        logger.info("="*70)
        logger.info("GENERATION COMPLETE")
        logger.info("="*70)
        logger.info(f"Output: {generated_text}")
        logger.info("="*70 + "\n")

        return generated_text

# ==============================================================================
# COMPARISON WITH PETALS
# ==============================================================================

def compare_with_petals():
    """
    Visual comparison of ZynkCluster vs Petals architectures.

    This demonstrates why ZynkCluster's approach is fundamentally different.
    """
    print("\n" + "="*70)
    print("ARCHITECTURE COMPARISON: ZynkCluster vs Petals")
    print("="*70)

    print("""
┌─────────────────────────────────────────────────────────────┐
│  PETALS (Sequential Layer Processing)                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Device A (Layers 0-10)                                      │
│      ↓  (50ms + network latency)                             │
│  Device B (Layers 11-20)                                     │
│      ↓  (50ms + network latency)                             │
│  Device C (Layers 21-31)                                     │
│      ↓  (50ms + network latency)                             │
│                                                              │
│  Total: ~150ms + 3× network overhead                         │
│  Active: ALL devices for EVERY token                         │
│  Bottleneck: Slowest device blocks entire pipeline          │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  ZYNKCLUSTER (Parallel Expert Execution)                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Router: "Select Expert 2, Expert 6"                         │
│                                                              │
│  Device A (Expert 2)  ←─── PARALLEL ───→  Device B (Expert 6)│
│      ↓ 45ms                                    ↓ 50ms        │
│                                                              │
│  Combine results                                             │
│                                                              │
│  Total: ~50ms + 1× network overhead                          │
│  Active: ONLY selected experts (2/8 = 25%)                   │
│  Efficient: Only necessary computation performed             │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Comparison:
────────────────────────────────────────────────────────────
  Latency:     max(expert times) + 1 hop  vs  sum(layer times) + N hops
  Devices:     only experts selected by router  vs  all devices every token
  Scalability: graceful degradation if one device fails
  Power:       only active-expert devices do work
  Network:     no chain of sequential hops

Note: The expert parallelism is a property of MoE architecture (not
invented here). ZynkCluster's contribution is the deployment model:
routing those experts to consumer devices over LAN via ZynkSync.
""")

    print("="*70 + "\n")

# ==============================================================================
# MAIN DEMONSTRATION
# ==============================================================================

if __name__ == "__main__":
    """
    Demonstration of ZynkCluster distributed inference.

    This shows what a full inference cycle would look like, even though
    we can't execute it without appropriate hardware.
    """

    print("\n" + "="*70)
    print("ZYNKCLUSTER PROOF OF CONCEPT DEMONSTRATION")
    print("="*70)
    print("Status: Illustrative code (non-executable without 2+ GPU devices)")
    print("Purpose: Demonstrate expert-level distribution over LAN using ZynkSync infrastructure")
    print("="*70 + "\n")

    # Show architectural comparison first
    compare_with_petals()

    # Initialize coordinator
    logger.info("Initializing coordinator...")
    coordinator = ZynkClusterCoordinator(
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )

    # Discover cluster
    coordinator.discover_cluster_via_zynksync()

    # Demonstrate distributed layer processing
    logger.info("\nDemonstrating distributed layer processing...")
    logger.info("(This would execute if expert node servers were running)\n")

    # Create dummy input
    batch_size = 1
    seq_len = 128
    hidden_size = 4096
    dummy_hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # Process one layer (demonstration)
    try:
        output = coordinator.execute_distributed_layer(
            layer_idx=0,
            hidden_states=dummy_hidden_states
        )
        logger.info("✅ Layer processing demonstration complete")
    except Exception as e:
        logger.error(f"❌ Layer processing failed (expected - no servers running): {e}")

    # Final summary
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("""
This code illustrates ZynkCluster's deployment model for MoE inference.

What This Demonstrates:
───────────────────────
1. MoE models activate only k experts per token — k is fixed by the model's
   trained architecture (Mixtral: k=2, DBRX: k=4, OLMoE: k=8, DeepSeek-V2:
   k=6). This code uses Mixtral as the reference; any open MoE model works
2. ZynkCluster maps those experts to separate LAN devices and dispatches
   requests simultaneously, rather than processing on one machine
3. ZynkSync's existing device discovery and pairing infrastructure is
   reused as the coordination layer — no separate cluster manager needed
4. Petals distributes layers sequentially; this distributes experts in
   parallel — a consequence of targeting MoE models specifically

Next Steps:
───────────
1. Set up 2+ GPU-enabled devices for testing
2. Implement full Python prototype
3. Measure actual latency on LAN vs single-device baseline
4. Port to Rust for integration with Tauri backend

Hardware Requirements:
──────────────────────
- 2+ devices with NVIDIA/AMD GPUs
- Gigabit+ network connection
- VRAM requirements vary by model: Mixtral 8x7B needs ~13GB per node /
  24GB+ total; other models scale with their parameter counts and k values

Community collaboration welcome!
Contact: matt@containai.ai
""")
    print("="*70 + "\n")
