import torch
import numpy as np
import unittest
import sys
import os
import time

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
        
        # Configure JAX to use GPU if available (and not already configured)
        # However, JAX usually auto-detects.
        # User requested "rerun all the tests but with JAX on the GPU".
        # We can print the default backend.
        print(f"JAX Default Backend: {jax.default_backend()}")
        
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
        
        # Move to GPU if available
        if torch.cuda.is_available():
            x1_torch = x1_torch.cuda()
            x2_torch = x2_torch.cuda()
            params_torch = {k: v.cuda() for k, v in params_torch.items()}
        
        for kernel_name, param_keys in kernels:
            with self.subTest(kernel=kernel_name):
                # Extract relevant params
                p_np = {k: params_np[k] for k in param_keys}
                p_torch = {k: params_torch[k] for k in param_keys}
                
                # Run JAX
                jax_func = getattr(self.jax_gp, kernel_name)
                
                # Warmup JAX
                # _ = jax_func(p_np, x1_np, x2_np).block_until_ready()
                
                t0 = time.time()
                res_jax = jax_func(p_np, x1_np, x2_np)
                if hasattr(res_jax, 'block_until_ready'):
                    res_jax.block_until_ready()
                t_jax = time.time() - t0
                
                # Run PyTorch
                torch_func = getattr(torch_gp, kernel_name)
                
                # Warmup PyTorch
                # _ = torch_func(p_torch, x1_torch, x2_torch)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.time()
                res_torch = torch_func(p_torch, x1_torch, x2_torch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_torch = time.time() - t0
                
                # print(f"{kernel_name}: JAX={t_jax*1000:.3f}ms, Torch={t_torch*1000:.3f}ms")
                
                self.assertTensorAlmostEqual(res_jax, res_torch, atol=1e-5)

                # Test Diagonal
                res_jax_diag = jax_func(p_np, x1_np, diag=True)
                res_torch_diag = torch_func(p_torch, x1_torch, diag=True)
                self.assertTensorAlmostEqual(res_jax_diag, res_torch_diag, atol=1e-5)

    def test_gp_predict(self):
        if not self.has_jax:
            return
            
        # Setup data
        N_obs = 10
        N_query = 5
        D = 3
        x_obs = torch.randn(N_obs, D)
        y_obs = torch.randint(0, 2, (N_obs, 1)).float()
        x_query = torch.randn(N_query, D)
        
        params_torch = {
            'lengthscale': torch.tensor([1.0]),
            'signal_variance': torch.tensor([1.0]),
            'alpha_eps': torch.tensor([0.1]),
            'strength': torch.tensor([5.0]),
            'constant': torch.tensor([0.0]), # Added missing parameter
        }
        
        # Move to GPU
        if torch.cuda.is_available():
            x_obs = x_obs.cuda()
            y_obs = y_obs.cuda()
            x_query = x_query.cuda()
            params_torch = {k: v.cuda() for k, v in params_torch.items()}
        
        # JAX data
        x_obs_jax = self.jnp.array(x_obs.cpu().numpy())
        y_obs_jax = self.jnp.array(y_obs.cpu().numpy())
        x_query_jax = self.jnp.array(x_query.cpu().numpy())
        params_jax = {k: self.jnp.array(v.cpu().numpy()) for k, v in params_torch.items()}
        
        # Mean/Cov funcs
        mean_func_torch = torch_gp.constant_mean
        cov_func_torch = torch_gp.squared_exponential_kernel
        
        mean_func_jax = self.jax_gp.constant_mean
        cov_func_jax = self.jax_gp.squared_exponential_kernel
        
        # 1. gp_predict (basic)
        # We need var_observed
        var_obs = torch.ones(N_obs) * 0.1
        if torch.cuda.is_available(): var_obs = var_obs.cuda()
        var_obs_jax = self.jnp.array(var_obs.cpu().numpy())
        
        mu_t, cov_t = torch_gp.gp_predict(
            mean_func=mean_func_torch,
            cov_func=cov_func_torch,
            params=params_torch,
            x_query=x_query,
            x_observed=x_obs,
            y_observed=y_obs,
            var_observed=var_obs,
            var_only=True
        )
        
        mu_j, cov_j = self.jax_gp.gp_predict(
            mean_func=mean_func_jax,
            cov_func=cov_func_jax,
            params=params_jax,
            x_query=x_query_jax,
            x_observed=x_obs_jax,
            y_observed=y_obs_jax,
            var_observed=var_obs_jax,
            var_only=True
        )
        
        self.assertTensorAlmostEqual(mu_j, mu_t, atol=1e-4)
        self.assertTensorAlmostEqual(cov_j, cov_t, atol=1e-4)
        
        # 2. beta_gp_predict
        preds_t = torch_gp.beta_gp_predict(
            mean_func=mean_func_torch,
            cov_func=cov_func_torch,
            params=params_torch,
            x_query=x_query,
            x_observed=x_obs,
            y_observed=y_obs,
            var_only=True
        )
        
        preds_j = self.jax_gp.beta_gp_predict(
            mean_func=mean_func_jax,
            cov_func=cov_func_jax,
            params=params_jax,
            x_query=x_query_jax,
            x_observed=x_obs_jax,
            y_observed=y_obs_jax,
            var_only=True
        )
        
        # Check first class predictions
        self.assertTensorAlmostEqual(preds_j[0][0], preds_t[0][0], atol=1e-4)
        self.assertTensorAlmostEqual(preds_j[0][1], preds_t[0][1], atol=1e-4)

if __name__ == '__main__':
    unittest.main()
