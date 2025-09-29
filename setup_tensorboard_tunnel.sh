#!/bin/bash

# SSH Tunnel and TensorBoard Setup Script for TPU VM
# Usage: ./setup_tensorboard_tunnel.sh [TPU_VM_NAME] [ZONE] [PROJECT]

# Default values (update these for your setup)
TPU_VM_NAME=${1:-"your-tpu-vm-name"}
ZONE=${2:-"us-central2-b"}
PROJECT=${3:-"your-project-id"}
LOCAL_PORT=${4:-6006}
REMOTE_PORT=${5:-6006}

echo "Setting up TensorBoard tunnel for TPU VM: $TPU_VM_NAME"
echo "Zone: $ZONE"
echo "Project: $PROJECT"
echo "Local port: $LOCAL_PORT -> Remote port: $REMOTE_PORT"
echo ""

# Function to check if gcloud is installed
check_gcloud() {
    if ! command -v gcloud &> /dev/null; then
        echo "Error: gcloud CLI not found. Please install Google Cloud SDK."
        echo "Visit: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
}

# Function to start TensorBoard on TPU VM
start_tensorboard_remote() {
    echo "Starting TensorBoard on TPU VM..."
    gcloud compute tpus tpu-vm ssh $TPU_VM_NAME \
        --zone=$ZONE \
        --project=$PROJECT \
        --command="nohup tensorboard --logdir=/tmp/tensorboard --port=$REMOTE_PORT --host=0.0.0.0 > tensorboard.log 2>&1 &"

    if [ $? -eq 0 ]; then
        echo "✓ TensorBoard started on TPU VM"
    else
        echo "✗ Failed to start TensorBoard on TPU VM"
        exit 1
    fi
}

# Function to create SSH tunnel
create_tunnel() {
    echo "Creating SSH tunnel..."
    gcloud compute tpus tpu-vm ssh $TPU_VM_NAME \
        --zone=$ZONE \
        --project=$PROJECT \
        -- -L $LOCAL_PORT:localhost:$REMOTE_PORT -N &

    TUNNEL_PID=$!
    echo "✓ SSH tunnel created (PID: $TUNNEL_PID)"
    echo "✓ TensorBoard accessible at: http://localhost:$LOCAL_PORT"
    echo ""
    echo "To stop the tunnel, run: kill $TUNNEL_PID"
    echo "Or press Ctrl+C to stop this script and the tunnel"

    # Wait for tunnel and handle cleanup
    trap "echo 'Stopping tunnel...'; kill $TUNNEL_PID 2>/dev/null" EXIT
    wait $TUNNEL_PID
}

# Main execution
main() {
    check_gcloud

    echo "Options:"
    echo "1. Start TensorBoard and create tunnel"
    echo "2. Create tunnel only (TensorBoard already running)"
    echo "3. Start TensorBoard only (no tunnel)"
    echo "4. Show existing TensorBoard processes"
    echo ""
    read -p "Choose option (1-4): " option

    case $option in
        1)
            start_tensorboard_remote
            sleep 3  # Give TensorBoard time to start
            create_tunnel
            ;;
        2)
            create_tunnel
            ;;
        3)
            start_tensorboard_remote
            echo "TensorBoard started. Create tunnel manually with:"
            echo "gcloud compute tpus tpu-vm ssh $TPU_VM_NAME --zone=$ZONE --project=$PROJECT -- -L $LOCAL_PORT:localhost:$REMOTE_PORT -N"
            ;;
        4)
            echo "Checking for existing TensorBoard processes..."
            gcloud compute tpus tpu-vm ssh $TPU_VM_NAME \
                --zone=$ZONE \
                --project=$PROJECT \
                --command="ps aux | grep tensorboard | grep -v grep"
            ;;
        *)
            echo "Invalid option"
            exit 1
            ;;
    esac
}

# Print usage if no arguments provided
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 [TPU_VM_NAME] [ZONE] [PROJECT] [LOCAL_PORT] [REMOTE_PORT]"
    echo ""
    echo "Example: $0 my-tpu-vm us-central2-b my-project-id 6006 6006"
    echo ""
    echo "Or edit the default values in this script and run: $0"
    echo ""
fi

main