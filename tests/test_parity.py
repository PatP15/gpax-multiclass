import torch
import numpy as np
import unittest
import sys
import os

# Add project root to path to ensure we can import both GPax (if installed/available) and GPtorch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestParity(unittest.TestCase):
    def setUp(self):
        self.key_jax = None
        try:
            import jax
            import jax.numpy as jnp
            self.key_jax = jax.random.PRNGKey(0)
            self.has_jax = True
        except ImportError:
            self.has_jax = False
            print("JAX not found, skipping JAX-dependent tests.")

    def assertTensorAlmostEqual(self, t1, t2, atol=1e-5, rtol=1e-3):
        if isinstance(t1, (np.ndarray, float, int)):
            t1 = torch.tensor(t1)
        if isinstance(t2, (np.ndarray, float, int)):
            t2 = torch.tensor(t2)
            
        # Handle JAX arrays if passed
        if hasattr(t1, '__jax_array__') or type(t1).__name__ == 'Array':
             t1 = torch.tensor(np.array(t1))
        if hasattr(t2, '__jax_array__') or type(t2).__name__ == 'Array':
             t2 = torch.tensor(np.array(t2))

        self.assertTrue(torch.allclose(t1, t2, atol=atol, rtol=rtol), f"Max diff: {(t1 - t2).abs().max()}")

if __name__ == '__main__':
    unittest.main()

