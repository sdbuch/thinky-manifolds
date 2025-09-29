import argparse
import os
import pickle
import time
from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import optax
from hyperspherical_descent import hyperspherical_descent
from manifold_muon import manifold_muon


def load_cifar10_data():
    """Load and preprocess CIFAR-10 data"""
    # CIFAR-10 normalization constants
    mean = jnp.array([0.49139968, 0.48215827, 0.44653124])
    std = jnp.array([0.24703233, 0.24348505, 0.26158768])

    # Load data using numpy/scipy (simple implementation)
    import urllib.request
    import tarfile
    import os

    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    # Download CIFAR-10 if not exists
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = os.path.join(data_dir, "cifar-10-python.tar.gz")

    if not os.path.exists(filename):
        print("Downloading CIFAR-10...")
        urllib.request.urlretrieve(url, filename)

    # Extract data
    extract_dir = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.exists(extract_dir):
        print("Extracting CIFAR-10...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(data_dir)

    def unpickle(file):
        with open(file, "rb") as fo:
            dict = pickle.load(fo, encoding="bytes")
        return dict

    # Load training data
    train_data = []
    train_labels = []
    for i in range(1, 6):
        batch = unpickle(os.path.join(extract_dir, f"data_batch_{i}"))
        train_data.append(batch[b"data"])
        train_labels.extend(batch[b"labels"])

    train_data = np.concatenate(train_data)
    train_labels = np.array(train_labels)

    # Load test data
    test_batch = unpickle(os.path.join(extract_dir, "test_batch"))
    test_data = test_batch[b"data"]
    test_labels = np.array(test_batch[b"labels"])

    # Reshape and normalize
    def preprocess(data, labels):
        # Reshape from (N, 3072) to (N, 32, 32, 3)
        data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        # Convert to float and normalize to [0, 1]
        data = data.astype(np.float32) / 255.0
        # Apply CIFAR-10 normalization
        data = (data - mean) / std
        return jnp.array(data), jnp.array(labels)

    train_images, train_labels = preprocess(train_data, train_labels)
    test_images, test_labels = preprocess(test_data, test_labels)

    return (train_images, train_labels), (test_images, test_labels)


# Global variables for data (will be loaded when needed)
train_images = None
train_labels = None
test_images = None
test_labels = None


def ensure_data_loaded():
    """Load CIFAR-10 data if not already loaded"""
    global train_images, train_labels, test_images, test_labels
    if train_images is None:
        (train_images, train_labels), (test_images, test_labels) = load_cifar10_data()


def init_mlp_params(key):
    """Initialize MLP parameters"""
    keys = random.split(key, 3)

    # Xavier/Glorot initialization for better training
    def init_layer(key, input_dim, output_dim):
        bound = jnp.sqrt(6.0 / (input_dim + output_dim))
        return random.uniform(key, (input_dim, output_dim), minval=-bound, maxval=bound)

    params = {
        "fc1": init_layer(keys[0], 32 * 32 * 3, 128),
        "fc2": init_layer(keys[1], 128, 64),
        "fc3": init_layer(keys[2], 64, 10),
    }
    return params


@jax.jit
def mlp_forward(params, x):
    """Forward pass through MLP"""
    x = x.reshape(x.shape[0], -1)  # Flatten to (batch_size, 3072)
    x = jax.nn.relu(x @ params["fc1"])
    x = jax.nn.relu(x @ params["fc2"])
    x = x @ params["fc3"]
    return x


@jax.jit
def compute_loss(params, images, labels):
    """Compute cross-entropy loss"""
    logits = mlp_forward(params, images)
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


@jax.jit
def compute_accuracy(params, images, labels):
    """Compute accuracy"""
    logits = mlp_forward(params, images)
    predictions = jnp.argmax(logits, axis=1)
    return jnp.mean(predictions == labels)


def get_batch(images, labels, batch_idx, batch_size):
    """Get a batch of data"""
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(images))
    return images[start_idx:end_idx], labels[start_idx:end_idx]


def train(epochs, initial_lr, update, wd, key):
    """Train the model"""
    ensure_data_loaded()  # Load data if not already loaded
    params = init_mlp_params(key)

    # Initialize optimizer for AdamW case
    if update == "adam":
        optimizer = optax.adamw(learning_rate=initial_lr, weight_decay=wd)
        opt_state = optimizer.init(params)
    else:
        assert update in [manifold_muon, hyperspherical_descent]
        optimizer = None
        opt_state = None

    batch_size = 1024
    num_batches = len(train_images) // batch_size
    steps = epochs * num_batches
    step = 0

    # Project the weights to the manifold for manifold optimizers
    if optimizer is None:
        params = jax.tree.map(lambda p: update(p, jnp.zeros_like(p), eta=0), params)

    epoch_losses = []
    epoch_times = []

    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0

        # Shuffle data each epoch
        key, subkey = random.split(key)
        perm = random.permutation(subkey, len(train_images))
        shuffled_images = train_images[perm]
        shuffled_labels = train_labels[perm]

        for i in range(num_batches):
            batch_images, batch_labels = get_batch(
                shuffled_images, shuffled_labels, i, batch_size
            )

            # Compute loss and gradients
            loss, grads = jax.value_and_grad(compute_loss)(
                params, batch_images, batch_labels
            )

            # Update learning rate with linear decay
            lr = initial_lr * (1 - step / steps)

            # Apply updates
            if optimizer is None:
                # Use manifold optimizer
                params = jax.tree.map(lambda p, g: update(p, g, eta=lr), params, grads)
            else:
                # Use AdamW
                # Update learning rate in optimizer
                scaled_optimizer = optax.scale_by_schedule(
                    lambda count: lr / initial_lr
                )
                combined_optimizer = optax.chain(scaled_optimizer, optimizer)
                if step == 0:  # Reinitialize with new lr
                    opt_state = combined_optimizer.init(params)
                updates, opt_state = combined_optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)

            step += 1
            running_loss += loss

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{num_batches}], Loss: {loss:.4f}"
                )

        end_time = time.time()
        epoch_loss = running_loss / num_batches
        epoch_time = end_time - start_time
        epoch_losses.append(float(epoch_loss))
        epoch_times.append(epoch_time)
        print(
            f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Time: {epoch_time:.4f} seconds"
        )

    return params, epoch_losses, epoch_times


def eval_model(params):
    """Evaluate the model on train and test sets"""
    ensure_data_loaded()  # Load data if not already loaded
    batch_size = 1024
    accs = []

    for images, labels in [(test_images, test_labels), (train_images, train_labels)]:
        num_batches = len(images) // batch_size
        correct = 0
        total = 0

        for i in range(num_batches):
            batch_images, batch_labels = get_batch(images, labels, i, batch_size)
            logits = mlp_forward(params, batch_images)
            predictions = jnp.argmax(logits, axis=1)
            correct += jnp.sum(predictions == batch_labels)
            total += len(batch_labels)

        # Handle remaining samples
        if len(images) % batch_size != 0:
            remaining_images = images[num_batches * batch_size :]
            remaining_labels = labels[num_batches * batch_size :]
            logits = mlp_forward(params, remaining_images)
            predictions = jnp.argmax(logits, axis=1)
            correct += jnp.sum(predictions == remaining_labels)
            total += len(remaining_labels)

        accuracy = 100 * correct / total
        accs.append(float(accuracy))

    print(
        f"Accuracy of the network on the {len(test_images)} test images: {accs[0]:.2f} %"
    )
    print(
        f"Accuracy of the network on the {len(train_images)} train images: {accs[1]:.2f} %"
    )
    return accs


def weight_stats(params):
    """Compute singular values and norms of weight matrices"""
    singular_values = []
    norms = []

    for param_name, param in params.items():
        u, s, vh = jnp.linalg.svd(param, full_matrices=False)
        singular_values.append(s)
        norms.append(jnp.linalg.norm(param))

    return singular_values, norms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on CIFAR-10.")
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate.")
    parser.add_argument(
        "--update",
        type=str,
        default="manifold_muon",
        choices=["manifold_muon", "hyperspherical_descent", "adam"],
        help="Update rule to use.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the random number generator."
    )
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay for AdamW.")
    args = parser.parse_args()

    # Set random seed for reproducibility
    key = random.PRNGKey(args.seed)

    update_rules = {
        "manifold_muon": manifold_muon,
        "hyperspherical_descent": hyperspherical_descent,
        "adam": "adam",
    }

    update = update_rules[args.update]

    print(f"Training with: {args.update}")
    print(
        f"Epochs: {args.epochs} --- LR: {args.lr}",
        f"--- WD: {args.wd}" if args.update == "adam" else "",
    )

    params, epoch_losses, epoch_times = train(
        epochs=args.epochs, initial_lr=args.lr, update=update, wd=args.wd, key=key
    )
    test_acc, train_acc = eval_model(params)
    singular_values, norms = weight_stats(params)

    # Convert JAX arrays to Python lists for pickling
    singular_values = [s.tolist() for s in singular_values]
    norms = [float(n) for n in norms]

    results = {
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "wd": args.wd,
        "update": args.update,
        "epoch_losses": epoch_losses,
        "epoch_times": epoch_times,
        "test_acc": test_acc,
        "train_acc": train_acc,
        "singular_values": singular_values,
        "norms": norms,
    }

    filename = f"update-{args.update}-lr-{args.lr}-wd-{args.wd}-seed-{args.seed}.pkl"
    os.makedirs("results", exist_ok=True)

    print(f"Saving results to {os.path.join('results', filename)}")
    with open(os.path.join("results", filename), "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {os.path.join('results', filename)}")
