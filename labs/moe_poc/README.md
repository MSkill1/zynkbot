# ZynkCluster: Distributed MoE Proof of Concept

**Status:** Design Phase (Illustrative Code Examples)
**Purpose:** Demonstrate architectural approach to distributed Mixture of Experts inference
**Requirements:** 2+ devices with GPU support for testing (NVIDIA CUDA or AMD ROCm)

---

## Overview

This directory contains architectural examples for distributed Mixture of Experts (MoE) inference. The code shown here is **illustrative only** - it demonstrates the intended architecture but is not currently executable without appropriate hardware.

### What the Code Examples Actually Implement

The following table distinguishes what the example code genuinely does from what is designed but not yet built:

| Component | Status | Notes |
|-----------|--------|-------|
| HTTP routing between coordinator and nodes | ✅ Implemented | Flask endpoints, parallel dispatch via ThreadPoolExecutor |
| Node server with `/expert/execute`, `/expert/list`, `/health` | ✅ Implemented | Functional HTTP API structure |
| Parallel expert request dispatch | ✅ Implemented | Concurrent requests, waits for all to complete |
| Router model (selects which experts to activate) | 🔲 Placeholder | Currently uses `torch.rand(8)` — real router model not yet built |
| Expert-only loading on each node | 🔲 Placeholder | Code loads the full model; comment marks it as illustrative |
| ZynkSync device discovery | 🔲 Designed | Import is present but commented out; device list is hardcoded |
| Efficient tensor serialization | 🔲 Designed | Currently uses `tolist()` → JSON; Protocol Buffers intended |

This is accurate as of the current code. The architecture is sound; the implementation is illustrative.

### What ZynkCluster Is (and Isn't)

**Mixture of Experts** architecture and the parallel nature of expert execution are established research concepts — see the papers in the References section. ZynkCluster does not claim to have invented these ideas.

**What this project proposes** is a specific deployment model that hasn't been packaged together before:

- **Expert-level distribution over consumer WiFi/LAN** — existing tools like Petals distribute model *layers* sequentially; ZynkCluster distributes individual *experts* so they can execute in parallel. The number of active experts per token (k) is fixed by each model's training: Mixtral uses k=2, DBRX uses k=4, OLMoE uses k=8. ZynkCluster maps those active experts onto separate devices. Optimal cluster size matches the model's k value.
- **Consumer hardware, not datacenter clusters** — prior work on distributed MoE (e.g., DeepSpeed-MoE) targets high-speed datacenter interconnects (InfiniBand). ZynkCluster targets ordinary WiFi/Gigabit Ethernet between devices people already own.
- **Built on existing device-pairing infrastructure** — rather than a standalone cluster manager, ZynkCluster extends ZynkSync's device discovery, authentication, and networking. The pairing system built for memory sync becomes the foundation for compute clustering.
- **Privacy-first, no cloud** — the cluster stays entirely on your local network.

```
Petals (Sequential layers):
Device A → Device B → Device C (all devices process every token)
Total latency: sum of all device times + network overhead

ZynkCluster (Parallel experts, consumer LAN):
Router selects Expert 2 + Expert 6
Device A (Expert 2) ←─ Parallel ─→ Device B (Expert 6)
Total latency: max(Device A, Device B) + 1× network hop
```

---

## Why This Matters

### Current Problem with Distributed Inference

**Petals** enables running large models across multiple devices, but uses layer-by-layer sequential processing:
- Every token must flow through ALL devices in order
- One slow device bottlenecks the entire pipeline
- High cumulative latency (sum of all network hops)

**Datacenter MoE frameworks** (DeepSpeed-MoE, etc.) require specialized high-speed interconnects — not viable on consumer hardware.

### ZynkCluster's Approach

**Mixture of Experts models** naturally divide into independent sub-models:
- Each query activates only k experts per layer — both the total number of experts and k are fixed by the model's trained architecture and weights. You cannot change them. Mixtral 8x7B has 8 experts with k=2; DBRX has 16 experts with k=4; OLMoE has 64 experts with k=8; DeepSeek-V2 has 160 experts with k=6. Choosing a model determines your cluster requirements.
- Those k active experts are independent and can execute simultaneously on different devices
- Only devices hosting active experts consume power/resources per query
- Natural fault tolerance (graceful degradation if one device fails)
- ZynkCluster is not Mixtral-specific — any open MoE model can be used. The cluster design is driven by whichever model you choose.
- **Minimum useful cluster size = k** — you need at least k devices to fully parallelize active expert calls. With fewer than k devices, some expert calls must be sequential. Hosting multiple experts per device is valid but reduces parallelism benefit.

---

## Hardware Requirements

### Minimum for Testing

- **2+ devices with GPU support** (NVIDIA CUDA or AMD ROCm)
- **~13GB VRAM per node** for Mixtral 8x7B (4 experts × ~3.25GB each)
- **Gigabit network** between devices
- Combined requirements: 24GB+ VRAM total for Mixtral 8x7B; 56GB+ for Mixtral 8x22B

CPU-only nodes are technically supported but impractical — expert inference on CPU takes 30-60 seconds vs ~50ms on GPU.

### Current Status

Code examples demonstrate architecture without execution. Proof of concept requires 2+ GPU-enabled devices. Community collaboration welcome for hardware testing.

---

## Why Python First, Then Rust?

Zynkbot is intentionally **Rust-based** for:
- Performance (native speed)
- Safety (memory safety, type safety)
- Mobile deployment (smaller binaries)

However, distributed MoE requires Python prototyping because:

1. **ML Library Ecosystem:** Hugging Face Transformers, PyTorch, and MoE implementations are Python-only
2. **Model Loading:** Mixtral model loading uses Python tooling
3. **Expert Extraction:** No Rust equivalent for MoE expert extraction exists yet
4. **Rapid Prototyping:** Python allows faster iteration on complex ML architectures

**Implementation Path:**
1. ✅ Design architecture (this document)
2. ⏸️ Python proof of concept (pending hardware validation)
3. ⏳ Test on 2+ GPU systems (community collaboration)
4. ⏳ Port to Rust using Candle or llama.cpp bindings
5. ⏳ Integrate with ZynkSync device discovery

---

## Architecture Components

### 1. Expert Node Server (`example_node_server.py`)

Each device in the cluster runs a node server that:
- Hosts a subset of experts (e.g., Experts 0-3)
- Exposes HTTP/gRPC API for expert execution
- Integrates with ZynkSync device registry
- Handles expert forward passes

**Key Operations:**
```python
# Execute specific expert on hidden states
POST /expert/execute
{
  "layer_idx": 0,
  "expert_id": 2,
  "hidden_states": [[...], [...], ...]
}
→ Returns expert output for combination
```

### 2. Cluster Coordinator (`example_coordinator.py`)

Orchestrates distributed inference:
- Discovers cluster devices via ZynkSync (designed — currently hardcoded in example code)
- Runs lightweight router model (~100MB) to select which experts to activate (designed — currently a random-score placeholder in example code)
- Executes experts in parallel across devices (implemented)
- Combines expert outputs (implemented)

**Inference Flow:**
```
1. User query → Tokenize & embed
2. Router → Select top-k experts (e.g., [2, 6])
3. Map experts to devices
4. Parallel execution:
   - Send hidden states to Device A (Expert 2)
   - Send hidden states to Device B (Expert 6)
   - Wait for both to complete
5. Combine outputs (weighted sum)
6. Continue generation (autoregressive)
```

### 3. ZynkSync Integration (Designed — Not Yet Implemented in Example Code)

The intended design leverages existing device pairing infrastructure:
- Device discovery (mDNS/local network)
- Authentication (existing pairing protocol)
- Network transport (HTTP/gRPC)
- Device capabilities registry

The example code imports ZynkSync but the integration is commented out — device discovery is currently hardcoded for illustration. The database extension below represents the intended schema addition, not a current migration.

**Database Extension:**
```sql
ALTER TABLE zynk_devices ADD COLUMN moe_capable BOOLEAN DEFAULT FALSE;
ALTER TABLE zynk_devices ADD COLUMN hosted_experts INTEGER[];
ALTER TABLE zynk_devices ADD COLUMN gpu_memory_gb INTEGER;
```

---

## Open Questions (What Needs Testing)

The theoretical foundation is solid — MoE parallelism is real and inherent to these models. What has not been validated on consumer hardware:

### Bandwidth & Tensor Size
Hidden states passed between devices must be serialized and transmitted over the network. The size of those tensors and whether consumer WiFi can handle them at inference speed is unknown. Gigabit Ethernet is likely fine; WiFi latency and bandwidth may be limiting factors depending on conditions.

### Latency Estimates
The +25ms network overhead figure in performance projections is an estimate based on typical LAN round-trip times plus rough tensor serialization overhead. It has not been measured on a live cluster. Real-world latency may be higher, particularly over WiFi or on congested networks.

### Serialization Overhead
Encoding and decoding tensors for network transfer adds CPU overhead not present in single-device inference. Impact on throughput is unknown until tested.

### What Success Looks Like
A successful proof of concept demonstrates: expert routing over a local network, parallel execution on 2+ GPU-enabled devices, result aggregation, and measured latency within acceptable range of theoretical projections. Recommended starting point: Mixtral 8x7B (k=2) on 2 devices over Gigabit Ethernet.

---

## Technical Challenges

### 1. Network Latency

**Problem**: Even on gigabit Ethernet, network round-trip adds ~5-20ms per device hop.

**Mitigations**:
- Use efficient serialization (Protocol Buffers, not JSON)
- Implement request pipelining
- Cache router decisions for similar queries
- Use UDP for non-critical paths

### 2. Device Heterogeneity

**Problem**: Devices have different compute capabilities (CPU vs GPU, RAM variations).

**Mitigations**:
- Load-aware expert assignment (faster devices get more experts)
- Dynamic rebalancing based on response times
- Quantization levels per device (Q4 on weak devices, Q8 on strong)

### 3. Synchronization & Consistency

**Problem**: Ensuring all devices have the same model version and calibration.

**Mitigations**:
- Version checking on cluster join
- Central model registry (could be one device)
- Checksums for expert weights
- Automatic updates via ZynkLink file sharing

### 4. Failure Handling

**Problem**: Devices can disconnect mid-query.

**Mitigations**:
- Timeout-based retry with exponential backoff
- Expert replication on critical paths
- Graceful degradation to available experts
- User notification of degraded performance

### 5. Security & Privacy

**Problem**: Sending hidden states across the network exposes query content.

**Mitigations**:
- Encrypt all inter-device traffic (TLS)
- Validate device signatures (existing ZynkSync pairing)
- Option for "local-only" mode (same trusted network)
- No logging of hidden states on remote devices

---

## Performance Projections

> **These are theoretical estimates**, extrapolated from single-device benchmarks and assumed network overhead. None of these numbers have been measured on a live cluster. See Open Questions above.

### Theoretical Performance (2-Device Cluster)

**Hardware:**
- Device A: RTX 3090 (24GB) - Experts 0-3
- Device B: RTX 3060 (12GB) - Experts 4-7

**Estimated Results:**
```
Single Device (RTX 3090):
  Tokens/sec: ~40
  Latency (first token): ~50ms
  Model: Mixtral 8x7B (47B total, 13B active)

2-Device Cluster (Gigabit Ethernet) — estimated:
  Tokens/sec: ~30-35 (estimated 75-87% of single device)
  Latency (first token): ~75ms (estimated +25ms network)
  Model: Mixtral 8x22B possible (141B total)

Projected: Can run 3x larger model — pending hardware validation
```

### Comparison with Petals

| Metric | Petals (Sequential) | ZynkCluster (Parallel) |
|--------|---------------------|------------------------|
| Active devices per token | All (8+) | Only k devices (model-dependent) |
| Latency | Sum of all hops | Max of parallel ops |
| Network traffic | High (full tensors) | Moderate (hidden states) |
| Fault tolerance | Single point of failure | Graceful degradation |
| Power efficiency | All devices active | Sparse activation |

---

## Files in This Directory

### Code Examples (Illustrative)

1. **`example_node_server.py`** - Expert hosting and execution server
2. **`example_coordinator.py`** - Distributed inference orchestration
3. **`architecture_diagram.txt`** - Visual flow diagram

### Documentation

4. **`README.md`** (this file) - Overview, architecture, challenges, roadmap

---

## Future Roadmap

### Phase 1: Proof of Concept
- Set up 2-device GPU cluster for testing
- Implement basic 2-device cluster
- Measure actual latency and throughput
- Document performance characteristics

### Phase 2: Python Prototype
- Full Mixtral 8x7B distribution
- ZynkSync integration
- Fault tolerance and recovery
- Performance optimization

### Phase 3: Rust Port
- Evaluate Candle vs llama.cpp for MoE support
- Implement expert extraction in Rust
- Port coordinator logic
- Integrate with Tauri backend

### Phase 4: Production Feature
- Mobile device participation
- Dynamic load balancing
- Cross-internet support (VPN/tunneling)
- UI for cluster management

---

## References

### Papers

- **Mixtral of Experts** (Jiang et al., 2024) - https://arxiv.org/abs/2401.04088
- **Switch Transformers** (Fedus et al., 2021) - https://arxiv.org/abs/2101.03961
- **Outrageously Large Neural Networks** (Shazeer et al., 2017) - https://arxiv.org/abs/1701.06538

### Implementations

- **Petals** - Distributed LLM inference: https://github.com/bigscience-workshop/petals
- **DeepSpeed MoE** - Microsoft's MoE framework: https://github.com/microsoft/DeepSpeed
- **vLLM** - Production MoE serving: https://github.com/vllm-project/vllm

### Models

- **Mixtral 8x7B** - 47B params, 13B active: `mistralai/Mixtral-8x7B-Instruct-v0.1`
- **Mixtral 8x22B** - 141B params, 39B active: `mistralai/Mixtral-8x22B-Instruct-v0.1`

---

## Conclusion

ZynkCluster applies MoE's inherent expert-level parallelism to a deployment context that hasn't been targeted before: consumer devices on local networks, using existing device-pairing infrastructure as the coordination layer. The MoE architecture and parallel execution concepts are established research — the contribution here is the deployment model.

**Key Takeaway:** The device pairing infrastructure built for ZynkSync (memory synchronization) can be extended for compute clustering, making this a natural evolution of Zynkbot's architecture rather than a separate system. Same discovery, same auth, same networking — different payload.

**Status:** Design phase complete, awaiting hardware availability or community collaboration for proof of concept implementation.

---

**Document Version:** 1.1
**Last Updated:** May 2026
**Author:** Matt Skillman
**License:** See main repository LICENSE file
