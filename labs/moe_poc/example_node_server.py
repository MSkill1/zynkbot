"""
ZynkCluster Expert Node Server (Illustrative Implementation)

This code demonstrates how a device would host specific MoE experts and respond
to inference requests in a distributed cluster. This is ILLUSTRATIVE ONLY - it
shows the architecture without being executable on current hardware.

Status: Non-functional example (design phase)
Hardware Required: GPU with CUDA/ROCm support
Purpose: Demonstrate expert-level distribution vs layer-level (Petals)

Author: Matt Skillman
Date: January 2026
License: See main repository
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

# Flask for HTTP API server
from flask import Flask, request, jsonify

# PyTorch for tensor operations
import torch

# Hugging Face Transformers for model loading
from transformers import AutoModelForCausalLM

# Standard library
import time
import logging

# ZynkSync integration (would be imported from main codebase)
# from zynk_sync import get_paired_devices, register_device_capability

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Flask app initialization
app = Flask(__name__)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# EXPERT NODE CLASS
# ==============================================================================

class ExpertNode:
    """
    Represents a compute node in the ZynkCluster.

    Each node hosts a subset of experts from an MoE model and exposes an API
    for remote expert execution. This enables parallel distributed inference.

    Key Concepts:
    -------------
    - Expert: A sub-network within an MoE layer. Expert size varies by model
      (e.g., Mixtral 8x7B has ~1.4B params per expert; other models differ)
    - Host: This device stores and executes a subset of experts
    - Remote Execution: The coordinator dispatches expert requests over LAN
    - Sparse Activation: MoE models activate only k experts per token — k is
      fixed by model training (Mixtral: k=2, DBRX: k=4, OLMoE: k=8,
      DeepSeek-V2: k=6); this node serves whichever of its hosted experts
      get selected for a given query

    Difference from Petals:
    -----------------------
    - Petals hosts entire layers and chains devices sequentially
    - ZynkCluster hosts individual experts; the coordinator dispatches to
      whichever devices hold the router-selected experts simultaneously
    - The parallelism comes from MoE's independent expert structure;
      ZynkCluster's role is routing expert requests to the right LAN devices
    """

    def __init__(self, device_id, expert_ids, model_name, device="cuda:0"):
        """
        Initialize the expert node.

        Parameters:
        -----------
        device_id : str
            Unique identifier from ZynkSync device registry
            Example: "device_a_rtx3090", "device_b_rtx3060"

        expert_ids : list[int]
            List of expert indices this device will host
            Example: [0, 1, 2, 3] means this device hosts the first 4 experts

        model_name : str
            HuggingFace model identifier
            Example: "mistralai/Mixtral-8x7B-Instruct-v0.1"

        device : str
            PyTorch device string
            Example: "cuda:0" (GPU), "cpu" (fallback)

        Deployment Model:
        -----------------
        Unlike Petals (which assigns full layers), each ZynkCluster node hosts
        individual experts. The coordinator dispatches to whichever nodes hold
        the router-selected experts for a given token — those requests go out
        simultaneously over LAN. The parallelism is a property of MoE's
        independent expert structure; this node just serves its assigned subset.
        """
        self.device_id = device_id
        self.expert_ids = expert_ids
        self.device = device
        self.model_name = model_name

        logger.info(f"[{device_id}] Initializing ZynkCluster node")
        logger.info(f"[{device_id}] Assigned experts: {expert_ids}")
        logger.info(f"[{device_id}] Model: {model_name}")
        logger.info(f"[{device_id}] Device: {device}")

        # Load the full model (in production, would load only assigned experts)
        # Note: Actual implementation would extract and load only specific experts
        # to save memory, but for demonstration we show full model loading
        logger.info(f"[{device_id}] Loading model... (this may take 1-2 minutes)")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Half precision for memory efficiency
                device_map=device,  # Automatic device placement
                trust_remote_code=True  # Required for some models
            )
            logger.info(f"[{device_id}] ✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"[{device_id}] ❌ Failed to load model: {e}")
            raise

        # Integration point: Register with ZynkSync device registry
        # In production, this would:
        # 1. Query zynk_devices table for this device's entry
        # 2. Update capabilities: moe_capable=True, hosted_experts=expert_ids
        # 3. Broadcast availability to other paired devices

        # Pseudocode:
        # self.sync_service = ZynkSyncService()
        # self.sync_service.register_capability(
        #     device_id=device_id,
        #     capability='moe_compute',
        #     metadata={'hosted_experts': expert_ids}
        # )

        logger.info(f"[{device_id}] Node initialization complete")
        logger.info(f"[{device_id}] Ready to serve expert execution requests")

    def execute_expert(self, layer_idx, expert_id, hidden_states):
        """
        Execute a specific expert on provided hidden states.

        This is the CORE OPERATION that makes ZynkCluster work.

        How It Works:
        -------------
        1. Coordinator sends hidden states over network (serialized tensor)
        2. This node deserializes and loads tensor to GPU
        3. Specific expert executes forward pass
        4. Output tensor is serialized and returned
        5. Coordinator combines outputs from multiple experts

        Parallel Execution:
        -------------------
        While this expert executes on Device A, the coordinator has simultaneously
        dispatched the other selected expert to Device B. MoE experts are independent
        sub-networks, so both can run at the same time — ZynkCluster routes them to
        separate LAN devices rather than running both sequentially on one machine.

        Parameters:
        -----------
        layer_idx : int
            Which transformer layer (0-31 for Mixtral)
            Example: 0 (first layer), 15 (middle layer)

        expert_id : int
            Which expert within that layer's MoE block (0-7 for Mixtral)
            Example: 2 (third expert)

        hidden_states : list or numpy.ndarray
            Input tensor as nested list (serialized for network transport)
            Shape: [batch_size, sequence_length, hidden_size]
            Example: [1, 128, 4096] for Mixtral

        Returns:
        --------
        list : Serialized output tensor as nested list
            Shape matches input: [batch_size, sequence_length, hidden_size]

        Raises:
        -------
        ValueError : If this node doesn't host the requested expert
        RuntimeError : If expert execution fails
        """

        # Validate this node hosts the requested expert
        if expert_id not in self.expert_ids:
            error_msg = (
                f"Expert {expert_id} not hosted on device {self.device_id}. "
                f"This node hosts: {self.expert_ids}"
            )
            logger.error(f"[{self.device_id}] ❌ {error_msg}")
            raise ValueError(error_msg)

        logger.info(f"[{self.device_id}] 🔄 Executing Expert {expert_id} "
                   f"(Layer {layer_idx})")

        start_time = time.time()

        try:
            # Access the specific expert within the MoE layer
            # Model structure: model.model.layers[layer_idx].block_sparse_moe.experts[expert_id]
            layer = self.model.model.layers[layer_idx]

            # NOTE: "block_sparse_moe" is Mixtral's attribute name for its MoE
            # layer. Other models use different names:
            #   - Mixtral: layer.block_sparse_moe.experts[i]
            #   - DBRX: layer.ffn.experts[i]  (structure varies)
            #   - OLMoE: layer.mlp.experts[i]  (structure varies)
            # Porting to a different model requires identifying the equivalent
            # attribute via the model's Hugging Face implementation.
            if not hasattr(layer, 'block_sparse_moe'):
                raise RuntimeError(
                    f"Layer {layer_idx} does not have a 'block_sparse_moe' attribute. "
                    "This example targets Mixtral's model structure. For other MoE "
                    "models (DBRX, OLMoE, DeepSeek-V2, etc.), update this code to "
                    "use the model-specific attribute name for the MoE layer."
                )

            moe_block = layer.block_sparse_moe
            expert = moe_block.experts[expert_id]

            # Convert received data (serialized list) to PyTorch tensor
            # In production, would use more efficient serialization (Protocol Buffers)
            hidden_states_tensor = torch.tensor(
                hidden_states,
                dtype=torch.float16,  # Match model dtype
                device=self.device  # Move to GPU
            )

            logger.debug(f"[{self.device_id}] Input tensor shape: "
                        f"{hidden_states_tensor.shape}")

            # Execute expert forward pass
            # This is a standard feedforward network (FFN) within the expert
            with torch.no_grad():  # Inference mode (no gradient computation)
                output_tensor = expert(hidden_states_tensor)

            logger.debug(f"[{self.device_id}] Output tensor shape: "
                        f"{output_tensor.shape}")

            # Convert output tensor back to serialized format for network transport
            # Move to CPU first to avoid CUDA/serialization issues
            output_list = output_tensor.cpu().tolist()

            elapsed_ms = (time.time() - start_time) * 1000

            logger.info(f"[{self.device_id}] ✅ Expert {expert_id} completed "
                       f"in {elapsed_ms:.1f}ms")

            return output_list

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"[{self.device_id}] ❌ Expert execution failed "
                        f"after {elapsed_ms:.1f}ms: {e}")
            raise RuntimeError(f"Expert execution failed: {e}")

    def get_status(self):
        """
        Return node status information.

        Used by coordinator during cluster discovery and health monitoring.

        Returns:
        --------
        dict : Node status information
            - device_id: Unique device identifier
            - hosted_experts: List of expert IDs this node serves
            - model_name: Which model is loaded
            - device: CUDA device or CPU
            - status: "ready", "busy", "error"
            - gpu_memory_used: Current GPU memory usage (if applicable)
        """
        status_info = {
            'device_id': self.device_id,
            'hosted_experts': self.expert_ids,
            'model_name': self.model_name,
            'device': self.device,
            'status': 'ready'
        }

        # Add GPU memory info if using CUDA
        if self.device.startswith('cuda'):
            try:
                device_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
                memory_allocated = torch.cuda.memory_allocated(device_idx) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(device_idx) / 1024**3  # GB
                status_info['gpu_memory_allocated_gb'] = round(memory_allocated, 2)
                status_info['gpu_memory_reserved_gb'] = round(memory_reserved, 2)
            except Exception as e:
                logger.warning(f"[{self.device_id}] Could not get GPU memory info: {e}")

        return status_info

# ==============================================================================
# GLOBAL NODE INSTANCE
# ==============================================================================

# This will be initialized on server startup with configuration
node = None

# ==============================================================================
# HTTP API ENDPOINTS
# ==============================================================================

@app.route("/expert/execute", methods=["POST"])
def execute_expert_endpoint():
    """
    HTTP endpoint for distributed expert execution.

    This is the main API that enables distributed inference. The coordinator
    sends a request to execute a specific expert, and this endpoint returns
    the expert's output.

    Request Format:
    ---------------
    POST /expert/execute
    Content-Type: application/json

    {
        "layer_idx": 0,           // Which transformer layer
        "expert_id": 2,           // Which expert within that layer
        "hidden_states": [        // Input tensor as nested list
            [[0.1, 0.2, ...], ...],
            ...
        ]
    }

    Response Format:
    ----------------
    {
        "success": true,
        "output": [               // Output tensor as nested list
            [[0.3, 0.4, ...], ...],
            ...
        ],
        "device_id": "device_a_rtx3090",
        "elapsed_ms": 45.2,       // Execution time in milliseconds
        "expert_id": 2,
        "layer_idx": 0
    }

    Error Response:
    ---------------
    {
        "success": false,
        "error": "Expert 6 not hosted on device desktop_rtx3090"
    }
    Status Code: 400 (Bad Request) or 500 (Internal Server Error)
    """

    if node is None:
        return jsonify({
            'success': False,
            'error': 'Node not initialized'
        }), 500

    try:
        # Parse request data
        data = request.json

        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400

        # Validate required fields
        required_fields = ['layer_idx', 'expert_id', 'hidden_states']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400

        layer_idx = data['layer_idx']
        expert_id = data['expert_id']
        hidden_states = data['hidden_states']

        logger.info(f"[API] Received request: Layer {layer_idx}, Expert {expert_id}")

        # Execute expert
        start_time = time.time()
        output = node.execute_expert(layer_idx, expert_id, hidden_states)
        elapsed_ms = (time.time() - start_time) * 1000

        # Return success response
        return jsonify({
            'success': True,
            'output': output,
            'device_id': node.device_id,
            'elapsed_ms': round(elapsed_ms, 2),
            'expert_id': expert_id,
            'layer_idx': layer_idx
        })

    except ValueError as e:
        # Expert not hosted on this node
        logger.error(f"[API] Bad request: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

    except Exception as e:
        # Internal server error
        logger.error(f"[API] Internal error: {e}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500


@app.route("/expert/list", methods=["GET"])
def list_experts():
    """
    Returns which experts this device hosts.

    Used by coordinator during cluster discovery. When a new device joins the
    network, the coordinator queries this endpoint to determine which experts
    are available.

    Response Format:
    ----------------
    {
        "device_id": "device_a_rtx3090",
        "hosted_experts": [0, 1, 2, 3],
        "model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "status": "ready",
        "gpu_memory_allocated_gb": 12.5,
        "gpu_memory_reserved_gb": 14.2
    }
    """

    if node is None:
        return jsonify({
            'error': 'Node not initialized'
        }), 500

    try:
        status_info = node.get_status()
        logger.info(f"[API] Status request - returning info for {node.device_id}")
        return jsonify(status_info)

    except Exception as e:
        logger.error(f"[API] Error getting status: {e}")
        return jsonify({
            'error': f'Failed to get status: {str(e)}'
        }), 500


@app.route("/health", methods=["GET"])
def health_check():
    """
    Simple health check endpoint for monitoring.

    Returns 200 OK if server is running and model is loaded.
    """
    if node is None:
        return jsonify({'status': 'error', 'message': 'Node not initialized'}), 500

    return jsonify({
        'status': 'healthy',
        'device_id': node.device_id,
        'model_loaded': True
    })

# ==============================================================================
# MAIN SERVER INITIALIZATION
# ==============================================================================

def initialize_node(device_id, expert_ids, model_name, device="cuda:0"):
    """
    Initialize the expert node with configuration.

    This would be called on server startup with configuration from:
    - Command line arguments
    - Configuration file
    - ZynkSync device registry

    Parameters:
    -----------
    device_id : str
        Unique device identifier
    expert_ids : list[int]
        Which experts this device should host
    model_name : str
        HuggingFace model identifier
    device : str
        PyTorch device string
    """
    global node

    logger.info("=" * 70)
    logger.info("ZynkCluster Expert Node Server")
    logger.info("=" * 70)
    logger.info(f"Device ID: {device_id}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Assigned Experts: {expert_ids}")
    logger.info(f"Device: {device}")
    logger.info("=" * 70)

    # Initialize node
    node = ExpertNode(
        device_id=device_id,
        expert_ids=expert_ids,
        model_name=model_name,
        device=device
    )

    logger.info("=" * 70)
    logger.info("✅ Node initialization complete")
    logger.info("🚀 Server ready to accept requests")
    logger.info("=" * 70)


if __name__ == "__main__":
    """
    Server entry point.

    In production, configuration would come from:
    - Command line arguments: python node_server.py --device-id desktop --experts 0,1,2,3
    - Config file: node_config.json
    - ZynkSync registration

    Example configurations:
    -----------------------

    Device A (e.g., RTX 3090, 24GB):
        device_id = "device_a_rtx3090"
        expert_ids = [0, 1, 2, 3]
        model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        device = "cuda:0"
        port = 5001

    Device B (e.g., RTX 3060, 12GB):
        device_id = "device_b_rtx3060"
        expert_ids = [4, 5, 6, 7]
        model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        device = "cuda:0"
        port = 5001

    Why Different Expert Assignments?
    ----------------------------------
    - Distributes compute load across devices
    - Enables parallel execution when router selects experts from different devices
    - Each device only loads ~13GB (4 experts) instead of full 47GB model
    """

    # Example configuration (would come from args/config in production)
    CONFIG = {
        'device_id': 'device_a_rtx3090',
        'expert_ids': [0, 1, 2, 3],  # This device hosts first 4 experts
        'model_name': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'device': 'cuda:0',
        'host': '0.0.0.0',  # Listen on all interfaces
        'port': 5001
    }

    # Initialize node with configuration
    initialize_node(
        device_id=CONFIG['device_id'],
        expert_ids=CONFIG['expert_ids'],
        model_name=CONFIG['model_name'],
        device=CONFIG['device']
    )

    # Start Flask server
    # In production, would use production WSGI server (gunicorn, uvicorn)
    logger.info(f"Starting server on {CONFIG['host']}:{CONFIG['port']}")

    app.run(
        host=CONFIG['host'],
        port=CONFIG['port'],
        debug=False,  # Set to True for development
        threaded=True  # Handle multiple requests concurrently
    )
