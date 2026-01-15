#!/usr/bin/env python3
"""
Modal Wrapper for TreeRL GRPO Training

This wrapper imports your existing training code and runs it on Modal's GPUs.
Minimal changes to existing code - just infrastructure abstraction.

Usage:
    # Upload rulings dataset first
    modal run modal_train_wrapper.py::upload_data --rulings-file cross_rulings_dataset.json
    
    # Run training
    modal run modal_train_wrapper.py --chapter 84 --epochs 3
    
    # Download checkpoints after training
    modal run modal_train_wrapper.py::download_artifacts
"""

import modal
import os
import sys
from pathlib import Path

# ============================================================================
# MODAL APP SETUP
# ============================================================================

app = modal.App("treerl-training")

# Persistent storage
volume = modal.Volume.from_name("treerl-data", create_if_missing=True)
VOLUME_PATH = "/data"

# =============================================================================
# IMAGE CONFIGURATION - Pinned to working H200 setup (2026-01-09)
# =============================================================================
# Key versions from verified H200 install:
#   - Python: 3.10.12
#   - torch: 2.9.0+cu128  
#   - CUDA: 12.8
#   - triton: 3.5.0
#   - mamba-ssm: 2.2.6.post3 (required for Nemotron-H)
#   - causal-conv1d: 1.5.3.post1 (required for Nemotron-H)

image = (
    # Use CUDA 12.8 base to match working setup
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git", 
        "curl", 
        "build-essential",
        "g++",              # C++ compiler for CUDA extensions
        "ninja-build",      # Fast builds for mamba-ssm/causal-conv1d
        "python3.10-dev",   # Needed for some compiled packages
    )
    # Set compiler to g++ (not clang++)
    .env({
        "CC": "gcc",
        "CXX": "g++",
    })
    # PyTorch 2.9.0 with CUDA 12.8 - exact working version
    .pip_install(
        "torch==2.9.0",
        "torchvision==0.24.0",
        "triton==3.5.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    # Build tools for CUDA extensions
    .pip_install(
        "ninja",       # Fast parallel builds
        "packaging",   # Required by causal-conv1d setup
        "wheel",
    )
    # Mamba/Causal-Conv1d - REQUIRED for Nemotron-H architecture
    # Built from source with CUDA support
    .pip_install(
        "causal-conv1d==1.5.3.post1",
        "mamba-ssm==2.2.6.post3",
    )
    # Unsloth - pinned to exact working commits
    .run_commands(
        "pip install --no-deps 'unsloth @ git+https://github.com/unslothai/unsloth.git@010775fbdebecf3f413002e593161393c72c0a09'",
        "pip install --no-deps 'unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo.git@c315ec1b0782a43893f34ed1dc264de9f2600236'",
    )
    # Training stack - exact versions from working setup
    .pip_install(
        "transformers==4.56.2",
        "tokenizers==0.22.2",
        "trl==0.22.2",
        "peft>=0.11.0",
        "accelerate>=0.30.0",
        "bitsandbytes==0.49.1",
        "datasets",
        "sentencepiece",
        "protobuf",
        "einops",
        "safetensors",
    )
    # vLLM for inference - use latest compatible with torch 2.9
    .pip_install("vllm>=0.8.0")
    # Utilities and API dependencies (from requirements.txt)
    .pip_install(
        "requests",
        "openai",
        "xformers",
        "setuptools",  # Often needed for package builds
        "rank-bm25",   # BM25 search for cross rulings
        "pandas",      # Data processing
        "openpyxl",    # Excel file support
        "fastapi",     # API framework
        "uvicorn",     # ASGI server
        "pydantic",    # Data validation
        "python-multipart",  # Form data handling
        "google-auth",       # Google Cloud auth
        "google-genai",      # Gemini API
        "groq",              # Groq API
    )
    # Copy your project code (excluding data - that goes to volume)
    .add_local_dir(
        local_path=".",
        remote_path="/app",
        ignore=[
            "*.pyc",
            "__pycache__",
            ".git",
            "*.gguf",
            "*.safetensors",
            "treerl_checkpoints",
            ".venv",
            "node_modules",
            # Large data files (upload to volume separately)
            "*.json",
            "*.jsonl", 
            "*.xlsx",
            "trajectory_*.json",
            "notes",
        ],
    )
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .workdir("/app")
)


# ============================================================================
# DATA MANAGEMENT
# ============================================================================

@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=600,
)
def upload_data(rulings_file: str = "cross_rulings_dataset.json"):
    """Upload rulings dataset and any other data to Modal volume."""
    import json
    import shutil
    
    # Read local file (passed as bytes or from mounted local dir)
    if os.path.exists(f"/app/{rulings_file}"):
        src = f"/app/{rulings_file}"
    elif os.path.exists(rulings_file):
        src = rulings_file
    else:
        raise FileNotFoundError(f"Rulings file not found: {rulings_file}")
    
    # Copy to volume
    dst = f"{VOLUME_PATH}/cross_rulings_dataset.json"
    shutil.copy(src, dst)
    
    with open(dst, 'r') as f:
        data = json.load(f)
    
    # Also copy HTS data if available
    hts_src = "/app/api/hts_data.json"
    if os.path.exists(hts_src):
        os.makedirs(f"{VOLUME_PATH}/api", exist_ok=True)
        shutil.copy(hts_src, f"{VOLUME_PATH}/api/hts_data.json")
    
    volume.commit()
    
    return f"Uploaded {len(data)} rulings to {dst}"


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=60,
)
def list_artifacts():
    """List all artifacts in the volume."""
    artifacts = {}
    
    for root, dirs, files in os.walk(VOLUME_PATH):
        rel_root = root.replace(VOLUME_PATH, "")
        for f in files:
            full_path = os.path.join(rel_root, f)
            size = os.path.getsize(os.path.join(root, f))
            artifacts[full_path] = f"{size / 1e6:.1f} MB"
    
    return artifacts


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=3600,
)
def download_artifacts(artifact_path: str = "checkpoints/final"):
    """Package artifacts for download."""
    import tarfile
    import io
    
    full_path = os.path.join(VOLUME_PATH, artifact_path)
    if not os.path.exists(full_path):
        return {"error": f"Path not found: {artifact_path}"}
    
    # Create tar archive
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
        tar.add(full_path, arcname=os.path.basename(artifact_path))
    
    # Save to volume for download
    archive_path = f"{VOLUME_PATH}/download_{os.path.basename(artifact_path)}.tar.gz"
    with open(archive_path, 'wb') as f:
        f.write(buffer.getvalue())
    
    volume.commit()
    
    return {
        "archive": archive_path,
        "size_mb": len(buffer.getvalue()) / 1e6,
    }


# ============================================================================
# TRAINING LOGIC (shared by all GPU configurations)
# ============================================================================

def _run_training_impl(
    chapter: str = "84",
    num_rulings: int = 20,
    epochs: int = 3,
    beam_size: int = 4,
    max_questions: int = 3,
    learning_rate: float = 5e-5,
    base_model: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    sft_adapter: str = "orlandowhite/nemotron3_nano_sft",
    train_max_seq_length: int = 65536,
    max_seq_length: int = 128000,
    vllm_max_model_len: int = 128000,
    resume_checkpoint: str = None,
):
    """Core training logic - called by GPU-specific functions."""
    import subprocess
    import json
    import torch
    
    # Log GPU info
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Total GPUs: {gpu_count}")
    
    # Setup paths
    os.makedirs(f"{VOLUME_PATH}/checkpoints", exist_ok=True)
    os.makedirs(f"{VOLUME_PATH}/samples", exist_ok=True)
    os.makedirs(f"{VOLUME_PATH}/adapter_sync", exist_ok=True)
    
    # Symlink volume paths to expected locations
    for link_name, target in [
        ("/app/cross_rulings_dataset.json", f"{VOLUME_PATH}/cross_rulings_dataset.json"),
        ("/app/treerl_checkpoints", f"{VOLUME_PATH}/checkpoints"),
    ]:
        if os.path.exists(target) and not os.path.exists(link_name):
            os.symlink(target, link_name)
    
    # Build command
    cmd = [
        "python", "treerl_grpo_train.py",
        "--chapter", chapter,
        "--num-rulings", str(num_rulings),
        "--epochs", str(epochs),
        "--beam-size", str(beam_size),
        "--max-questions", str(max_questions),
        "--lr", str(learning_rate),
        "--base-model", base_model,
        "--train-max-seq-length", str(train_max_seq_length),
        "--max-seq-length", str(max_seq_length),
        "--vllm-max-model-len", str(vllm_max_model_len),
        "--output-dir", f"{VOLUME_PATH}/checkpoints",
    ]
    
    if sft_adapter:
        cmd.extend(["--sft-adapter", sft_adapter])
    else:
        cmd.append("--no-sft-adapter")
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 70)
    
    # Run training
    result = subprocess.run(
        cmd,
        cwd="/app",
        capture_output=False,
        text=True,
    )
    
    # Commit results to volume
    volume.commit()
    
    # Return summary
    metrics_file = f"{VOLUME_PATH}/checkpoints/training_metrics.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = []
    
    return {
        "return_code": result.returncode,
        "metrics": metrics,
        "checkpoints_path": f"{VOLUME_PATH}/checkpoints",
    }


# ============================================================================
# GPU-SPECIFIC TRAINING FUNCTIONS
# ============================================================================

@app.function(
    image=image,
    gpu="H200",  # 1x H200 GPU
    volumes={VOLUME_PATH: volume},
    timeout=86400,
    retries=modal.Retries(max_retries=2, initial_delay=60.0),
    memory=32768,
)
def train_1gpu(**kwargs):
    """Train with 1x H200 GPU (141GB VRAM)."""
    return _run_training_impl(**kwargs)


@app.function(
    image=image,
    gpu="H200:2",  # 2x H200 GPUs
    volumes={VOLUME_PATH: volume},
    timeout=86400,
    retries=modal.Retries(max_retries=2, initial_delay=60.0),
    memory=65536,
)
def train_2gpu(**kwargs):
    """Train with 2x H200 GPUs."""
    return _run_training_impl(**kwargs)


@app.function(
    image=image,
    gpu="H200:4",  # 4x H200 GPUs
    volumes={VOLUME_PATH: volume},
    timeout=86400,
    retries=modal.Retries(max_retries=2, initial_delay=60.0),
    memory=65536,
)
def train(**kwargs):
    """Train with 4x H200 GPUs (default)."""
    return _run_training_impl(**kwargs)


# ============================================================================
# ALTERNATIVE: DIRECT PYTHON INTEGRATION
# ============================================================================

@app.function(
    image=image,
    gpu="H200:4",  # 4x H200 GPUs
    volumes={VOLUME_PATH: volume},
    timeout=86400,
    memory=65536,
)
def train_direct(
    chapter: str = "84",
    num_rulings: int = 20,
    epochs: int = 3,
    max_seq_length: int = 128000,
    **kwargs,
):
    """
    Run training by directly importing the training code.
    
    More integrated approach - imports and calls train() directly.
    """
    import sys
    sys.path.insert(0, "/app")
    
    # Setup symlinks
    os.makedirs(f"{VOLUME_PATH}/checkpoints", exist_ok=True)
    
    if os.path.exists(f"{VOLUME_PATH}/cross_rulings_dataset.json"):
        if not os.path.exists("/app/cross_rulings_dataset.json"):
            os.symlink(
                f"{VOLUME_PATH}/cross_rulings_dataset.json",
                "/app/cross_rulings_dataset.json"
            )
    
    # Import your training code
    from treerl_grpo_train import TreeRLConfig, train
    
    # Create config
    config = TreeRLConfig(
        chapter=chapter,
        num_rulings_per_epoch=num_rulings,
        num_epochs=epochs,
        output_dir=f"{VOLUME_PATH}/checkpoints",
        adapter_sync_dir=f"{VOLUME_PATH}/adapter_sync",
        samples_dir=f"{VOLUME_PATH}/samples",
        **kwargs,
    )
    
    # Run training
    train(config)
    
    # Commit to volume
    volume.commit()
    
    return {"status": "complete", "output_dir": config.output_dir}


# ============================================================================
# INFERENCE CLASS (for rollouts with warm model)
# ============================================================================

@app.cls(
    image=image,
    gpu="H200:4",  # 4x H200 for inference
    volumes={VOLUME_PATH: volume},
    timeout=3600,
    container_idle_timeout=300,  # Keep warm 5 min between calls
)
class VLLMRolloutEngine:
    """
    Stateful vLLM inference for efficient batch rollouts.
    
    Model loads once when container starts, stays warm for multiple calls.
    """
    
    model_path: str = modal.parameter(default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    
    @modal.enter()
    def load_model(self):
        """Load model once when container starts."""
        from vllm import LLM, SamplingParams
        
        print(f"Loading model: {self.model_path}")
        
        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            max_model_len=128000,  # 128k context
            tensor_parallel_size=4,  # Distribute across 4 GPUs
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=2048,
        )
        
        print("Model loaded!")
    
    @modal.method()
    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate completions for batch of prompts."""
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [o.outputs[0].text for o in outputs]
    
    @modal.method()
    def run_rollout(self, ruling: dict) -> list[dict]:
        """Run single ruling rollout."""
        # Import your rollout code
        sys.path.insert(0, "/app")
        
        # Simplified - in production, integrate with your rollout code
        product_desc = ruling.get("short_product_description", "")
        gold_code = ruling.get("hts_code", "")
        
        prompt = f"Classify this product for HTS: {product_desc}"
        outputs = self.llm.generate([prompt], self.sampling_params)
        
        return [{
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": outputs[0].outputs[0].text},
            ],
            "gold_code": gold_code,
            "leaf_reward": 0.5,  # Compute actual reward
        }]


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================

@app.local_entrypoint()
def main(
    chapter: str = "84",
    num_rulings: int = 20,
    epochs: int = 3,
    upload_rulings: bool = False,
    rulings_file: str = "cross_rulings_dataset.json",
    list_only: bool = False,
    gpus: int = 4,  # NEW: GPU count selector
):
    """
    Run TreeRL training on Modal.
    
    Examples:
        # First, upload your data
        modal run modal_train_wrapper.py --upload-rulings --rulings-file cross_rulings_dataset.json
        
        # Then run training
        modal run modal_train_wrapper.py --chapter 84 --epochs 3
        
        # List artifacts
        modal run modal_train_wrapper.py --list-only
    """
    import json
    
    if list_only:
        print("Artifacts in volume:")
        artifacts = list_artifacts.remote()
        for path, size in sorted(artifacts.items()):
            print(f"  {path}: {size}")
        return
    
    if upload_rulings:
        print(f"Uploading {rulings_file}...")
        result = upload_data.remote(rulings_file)
        print(result)
        return
    
    print("=" * 70)
    print("TreeRL GRPO Training on Modal")
    print("=" * 70)
    print(f"GPUs: {gpus}x H200")
    print(f"Chapter: {chapter}")
    print(f"Epochs: {epochs}")
    print(f"Rulings per epoch: {num_rulings}")
    print()
    
    # Select training function based on GPU count
    train_kwargs = dict(
        chapter=chapter,
        num_rulings=num_rulings,
        epochs=epochs,
    )
    
    if gpus == 1:
        result = train_1gpu.remote(**train_kwargs)
    elif gpus == 2:
        result = train_2gpu.remote(**train_kwargs)
    else:
        # Default: 4 GPUs
        result = train.remote(**train_kwargs)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(json.dumps(result, indent=2, default=str))

