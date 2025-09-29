# TPU Profiling and TensorBoard Setup Guide

This guide shows you how to profile your JAX training code on TPU VM and view the results with TensorBoard on your local machine.

## Quick Start

### 1. Run Training with Profiling

On your TPU VM, run:

```bash
# Basic profiling
python src/main.py --profile --epochs 2 --update manifold_muon

# Custom profile directory
python src/main.py --profile --profile_dir /home/$USER/profiles --epochs 2 --update manifold_muon

# Compare different optimizers
python src/main.py --profile --epochs 2 --update manifold_muon
python src/main.py --profile --epochs 2 --update hyperspherical_descent
python src/main.py --profile --epochs 2 --update adam
```

### 2. Set Up SSH Tunnel and TensorBoard

#### Option A: Use the provided script (recommended)

```bash
# Edit the script with your TPU VM details
./setup_tensorboard_tunnel.sh your-tpu-vm-name us-central2-b your-project-id

# Or provide details as arguments
./setup_tensorboard_tunnel.sh my-tpu-vm us-central2-b my-project 6006 6006
```

#### Option B: Manual setup

```bash
# 1. SSH into your TPU VM and start TensorBoard
gcloud compute tpus tpu-vm ssh your-tpu-vm-name \
    --zone=us-central2-b \
    --project=your-project-id \
    --command="tensorboard --logdir=/tmp/tensorboard --port=6006 --host=0.0.0.0"

# 2. In another terminal, create SSH tunnel
gcloud compute tpus tpu-vm ssh your-tpu-vm-name \
    --zone=us-central2-b \
    --project=your-project-id \
    -- -L 6006:localhost:6006 -N
```

### 3. View Results

Open your browser and go to: `http://localhost:6006`

## Profiling Features Added

### Command Line Options

- `--profile`: Enable JAX profiling
- `--profile_dir`: Directory to save profiling data (default: `/tmp/tensorboard`)

### Wall Clock Timing

The code now tracks and reports:
- **Total training time**: End-to-end training duration
- **Per-epoch times**: Individual epoch durations (existing)
- **Evaluation time**: Model evaluation duration
- **Profile status**: Whether profiling was enabled

### Example Output

```
Training with: manifold_muon
Epochs: 2 --- LR: 0.1
Profiling enabled. Data will be saved to: /tmp/tensorboard/manifold_muon_lr0.1_epochs2_seed42
To view with TensorBoard: tensorboard --logdir=/tmp/tensorboard
Starting profiled training...
Epoch [1/2], Step [49/49], Loss: 2.3456
Epoch 1, Loss: 2.4123, Time: 15.23 seconds
Epoch [2/2], Step [49/49], Loss: 2.1234
Epoch 2, Loss: 2.1456, Time: 12.45 seconds
Total training time (wall clock): 27.68 seconds
Evaluating model...
Evaluation time: 3.45 seconds
```

## TensorBoard Views for Performance Analysis

### 1. Trace Viewer
- **Purpose**: Visualize timeline of TPU operations
- **Navigation**:
  - A/D keys: Pan left/right
  - W/S keys: Zoom in/out
  - Click operations to see source code
- **What to look for**:
  - Long idle periods (potential optimization targets)
  - Memory transfer bottlenecks
  - Compute vs communication ratios

### 2. Profile Overview
- **Purpose**: High-level performance summary
- **Metrics**: Step time breakdown, TPU utilization
- **What to look for**:
  - TPU utilization percentage
  - Time spent in different operation types

### 3. Graph Viewer
- **Purpose**: Visualize HLO computational graph
- **What to look for**:
  - Graph structure and optimization
  - Memory layout and data flow

## Troubleshooting

### Common Issues

1. **Empty profiles**: Ensure operations run on TPU, not CPU
2. **Large profile files**: Use shorter traces (fewer epochs)
3. **SSH tunnel fails**: Check TPU VM name, zone, and project ID
4. **TensorBoard not accessible**: Verify port forwarding and firewall settings

### Performance Tips

- Profile only a subset of training (1-2 epochs) for faster analysis
- Use `x.block_until_ready()` to ensure operations complete
- Disable host tracing for production profiling to reduce overhead

## Comparing Optimizers

To compare wall clock performance between optimizers:

```bash
# Run all three optimizers with profiling
for optimizer in manifold_muon hyperspherical_descent adam; do
    echo "Profiling $optimizer..."
    python src/main.py --profile --epochs 2 --update $optimizer --lr 0.01
done

# Results will be in separate directories under /tmp/tensorboard/
# Compare timing in TensorBoard and saved result files
```

## Advanced Configuration

### Custom Profile Options

Modify the profiling in `main.py` for advanced use cases:

```python
# Custom profiling options
options = jax.profiler.ProfileOptions()
options.python_tracer_level = 0  # Disable Python tracing
options.host_tracer_level = 0    # Disable host tracing

with jax.profiler.trace(profile_run_dir, profiler_options=options):
    # Your training code
    pass
```

### Memory Profiling

Add memory profiling for debugging OOM issues:

```python
# Save memory profile
jax.profiler.save_device_memory_profile("/tmp/memory_profile.prof")
```