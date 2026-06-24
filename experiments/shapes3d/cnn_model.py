import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np

class CNN(nn.Module):
    """A simple CNN model."""
    num_classes: int

    @nn.compact
    def __call__(self, x, return_embeddings=False):
        # Input x: (Batch, 64, 64, 3)
        
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Flatten
        x = x.reshape((x.shape[0], -1))
        
        x = nn.Dense(features=64)(x)
        embeddings = nn.relu(x)
        
        logits = nn.Dense(features=self.num_classes)(embeddings)
        
        if return_embeddings:
            return logits, embeddings
        return logits

def create_train_state(rng, num_classes, learning_rate=1e-3):
    """Creates initial `TrainState`."""
    cnn = CNN(num_classes=num_classes)
    params = cnn.init(rng, jnp.ones([1, 64, 64, 3]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=cnn.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch_images, batch_labels):
    """Train for a single step."""
    # Normalize on GPU
    batch_images = batch_images.astype(jnp.float32) / 255.0
    
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch_images)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch_labels).mean()
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = {
        'loss': loss,
        'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch_labels)
    }
    return state, metrics

@jax.jit
def eval_step(state, batch_images, batch_labels):
    # Normalize on GPU
    batch_images = batch_images.astype(jnp.float32) / 255.0
    
    logits = state.apply_fn({'params': state.params}, batch_images)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch_labels).mean()
    metrics = {
        'loss': loss,
        'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch_labels)
    }
    return metrics

@jax.jit
def get_embeddings(state, batch_images):
    # Normalize on GPU
    batch_images = batch_images.astype(jnp.float32) / 255.0
    
    _, embeddings = state.apply_fn({'params': state.params}, batch_images, return_embeddings=True)
    return embeddings

def train_model(X_train, y_train, X_val, y_val, num_classes, num_epochs=10, batch_size=64, seed=0, verbose=True):
    """Full training loop."""
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    
    state = create_train_state(init_rng, num_classes)
    
    # Training loop
    num_train = X_train.shape[0]
    steps_per_epoch = num_train // batch_size
    
    for epoch in range(num_epochs):
        # Shuffle
        rng, shuffle_rng = jax.random.split(rng)
        perms = jax.random.permutation(shuffle_rng, num_train)
        perms = perms[:steps_per_epoch * batch_size] # Drop remainder
        perms = perms.reshape((steps_per_epoch, batch_size))
        
        batch_metrics = []
        for perm in perms:
            batch_images = X_train[perm]
            batch_labels = y_train[perm]
            state, metrics = train_step(state, batch_images, batch_labels)
            batch_metrics.append(metrics)
            
        # Compute average train metrics
        train_loss = np.mean([m['loss'] for m in batch_metrics])
        train_acc = np.mean([m['accuracy'] for m in batch_metrics])
        
        # Validation (Batched to avoid OOM)
        val_loss_acc = []
        val_acc_acc = []
        
        val_batch_size = 1024 # Larger batch size for eval
        num_val = X_val.shape[0]
        
        for i in range(0, num_val, val_batch_size):
            batch_val = X_val[i:i+val_batch_size]
            batch_y = y_val[i:i+val_batch_size]
            m = eval_step(state, batch_val, batch_y)
            val_loss_acc.append(m['loss'] * len(batch_val)) # Weighted average later
            val_acc_acc.append(m['accuracy'] * len(batch_val))
            
        val_loss = np.sum(val_loss_acc) / num_val
        val_acc = np.sum(val_acc_acc) / num_val
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
    return state





