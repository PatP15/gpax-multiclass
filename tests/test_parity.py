import torch
import numpy as np
import unittest
import sys
import os

# Add project root to path to ensure we can import both GPax (if installed/available) and GPtorch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GPtorch import utils as torch_utils
from GPtorch.probing import gp as torch_gp

class TestParity(unittest.TestCase):
    def setUp(self):
        self.key_jax = None
        self.has_jax = False
        # try:
        import jax
        import jax.numpy as jnp
        from GPax.probing import gp as jax_gp
        self.key_jax = jax.random.PRNGKey(0)
        self.jax_gp = jax_gp
        self.jnp = jnp
        self.has_jax = True
        # except ImportError:
        #     print("JAX or GPax not found, skipping JAX-dependent tests.")
        # except Exception as e:
        #     print(f"Error importing JAX/GPax: {e}")

    def assertTensorAlmostEqual(self, t1, t2, atol=1e-5, rtol=1e-3):
        if isinstance(t1, (np.ndarray, float, int, list)):
            t1 = torch.tensor(np.array(t1))
        if isinstance(t2, (np.ndarray, float, int, list)):
            t2 = torch.tensor(np.array(t2))
            
        # Handle JAX arrays if passed
        if 'jax' in str(type(t1)) or hasattr(t1, 'device_buffer'):
             t1 = torch.tensor(np.array(t1))
        if 'jax' in str(type(t2)) or hasattr(t2, 'device_buffer'):
             t2 = torch.tensor(np.array(t2))

        # If tensors are on GPU, move to CPU for comparison
        if isinstance(t1, torch.Tensor):
            t1 = t1.detach().cpu()
        if isinstance(t2, torch.Tensor):
            t2 = t2.detach().cpu()

        # Squeeze both to ignore differences in singleton dimensions (e.g. (N, 1) vs (N,))
        if t1.shape != t2.shape:
            t1 = t1.squeeze()
            t2 = t2.squeeze()

        self.assertTrue(torch.allclose(t1, t2, atol=atol, rtol=rtol), f"Max diff: {(t1 - t2).abs().max()}\nTensor 1: {t1}\nTensor 2: {t2}")

    def test_params_tree(self):
        # Create a nested dictionary of tensors
        params = {
            'a': torch.randn(2, 3),
            'b': {
                'c': torch.randn(4),
                'd': torch.randn(1, 1)
            }
        }
        
        tree = torch_utils.ParamsTree(params)
        arr = tree.toarray(params)
        expected_size = 2*3 + 4 + 1
        self.assertEqual(arr.shape[0], expected_size)
        params_recon = tree.todict(arr)
        
        self.assertTensorAlmostEqual(params['a'], params_recon['a'])
        self.assertTensorAlmostEqual(params['b']['c'], params_recon['b']['c'])
        self.assertTensorAlmostEqual(params['b']['d'], params_recon['b']['d'])

    def test_kernels(self):
        if not self.has_jax:
            return

        # Setup data
        N1, N2, D = 10, 5, 3
        x1_np = np.random.randn(N1, D).astype(np.float32)
        x2_np = np.random.randn(N2, D).astype(np.float32)
        
        x1_torch = torch.tensor(x1_np)
        x2_torch = torch.tensor(x2_np)
        
        # Kernel Parameters
        params_np = {
            'lengthscale': np.array([1.5]).astype(np.float32),
            'signal_variance': np.array([0.8]).astype(np.float32),
            'dot_prod_sigma': np.array([1.2]).astype(np.float32),
            'dot_prod_bias': np.array([0.5]).astype(np.float32),
            'intercept_scaling': np.array([0.1]).astype(np.float32)
        }
        
        params_torch = {k: torch.tensor(v) for k, v in params_np.items()}
        
        kernels = [
            ('squared_exponential_kernel', ['lengthscale', 'signal_variance']),
            ('laplace_kernel', ['lengthscale', 'signal_variance']),
            ('additive_laplace_kernel', ['lengthscale', 'signal_variance']),
            ('dot_product_kernel', ['dot_prod_sigma', 'dot_prod_bias']),
            ('cosine_kernel', ['signal_variance', 'intercept_scaling']),
            ('squared_exponential_sphere_kernel', ['lengthscale', 'signal_variance', 'intercept_scaling'])
        ]
        
        for kernel_name, param_keys in kernels:
            with self.subTest(kernel=kernel_name):
                # Extract relevant params
                p_np = {k: params_np[k] for k in param_keys}
                p_torch = {k: params_torch[k] for k in param_keys}
                
                # Run JAX
                jax_func = getattr(self.jax_gp, kernel_name)
                res_jax = jax_func(p_np, x1_np, x2_np)
                
                # Run PyTorch
                torch_func = getattr(torch_gp, kernel_name)
                res_torch = torch_func(p_torch, x1_torch, x2_torch)
                
                self.assertTensorAlmostEqual(res_jax, res_torch, atol=1e-5)

                # Test Diagonal
                res_jax_diag = jax_func(p_np, x1_np, diag=True)
                res_torch_diag = torch_func(p_torch, x1_torch, diag=True)
                self.assertTensorAlmostEqual(res_jax_diag, res_torch_diag, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
