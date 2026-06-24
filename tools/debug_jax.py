import jax
import jax.numpy as jnp

print("JAX version:", jax.__version__)
try:
    print("Devices:", jax.devices())
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1000, 1000))
    y = jnp.dot(x, x)
    print("Computation result shape:", y.shape)
    print("Success!")
except Exception as e:
    print("Error:", e)

