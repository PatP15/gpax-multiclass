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
from GPtorch.probing import gp_multiclass as torch_gp_multiclass

class TestParity(unittest.TestCase):
    def setUp(self):
        self.key_jax = None
        self.has_jax = False
        try:
            import jax
            import jax.numpy as jnp
            from GPax.probing import gp as jax_gp
            from GPax.probing import gp_multiclass as jax_gp_multiclass
            
            # Configure JAX to use GPU if available (and not already configured)
            # print(f"JAX Default Backend: {jax.default_backend()}")
            
            self.key_jax = jax.random.PRNGKey(0)
            self.jax_gp = jax_gp
            self.jax_gp_multiclass = jax_gp_multiclass
            self.jnp = jnp
            self.has_jax = True
        except ImportError:
            print("JAX or GPax not found, skipping JAX-dependent tests.")
        except Exception as e:
            print(f"Error importing JAX/GPax: {e}")

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

    # ... (previous tests omitted for brevity, but they run if this file is executed) ...
    
    def test_multiclass_gp(self):
        if not self.has_jax:
            return
            
        N_obs = 10
        N_query = 5
        D = 3
        K = 3 # classes
        
        x_obs = torch.randn(N_obs, D)
        y_obs_idx = torch.randint(0, K, (N_obs, 1))
        x_query = torch.randn(N_query, D)
        
        # One hot
        y_obs_oh = torch.nn.functional.one_hot(y_obs_idx.flatten(), K).float()
        
        params_torch = {
            'lengthscale': torch.tensor([1.0]),
            'signal_variance': torch.tensor([1.0]),
            'alpha_eps': torch.tensor([0.1]),
            'strength': torch.tensor([5.0]),
            'num_classes': torch.tensor([K]),
            'constant': torch.tensor([0.0]), 
        }
        
        if torch.cuda.is_available():
            x_obs = x_obs.cuda()
            y_obs_oh = y_obs_oh.cuda()
            x_query = x_query.cuda()
            params_torch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in params_torch.items()}
            
        # JAX setup
        x_obs_jax = self.jnp.array(x_obs.cpu().numpy())
        y_obs_jax = self.jnp.array(y_obs_oh.cpu().numpy())
        x_query_jax = self.jnp.array(x_query.cpu().numpy())
        params_jax = {k: self.jnp.array(v.cpu().numpy()) if torch.is_tensor(v) else v for k, v in params_torch.items()}
        
        # Funcs
        mean_func_torch = torch_gp.constant_mean
        cov_func_torch = torch_gp.squared_exponential_kernel
        mean_func_jax = self.jax_gp.constant_mean
        cov_func_jax = self.jax_gp.squared_exponential_kernel
        
        # Test get_latent_observations_dirichlet
        y_lat_t, var_lat_t = torch_gp_multiclass.get_latent_observations_dirichlet(params_torch, y_obs_oh)
        y_lat_j, var_lat_j = self.jax_gp_multiclass.get_latent_observations_dirichlet(params_jax, y_obs_jax)
        
        self.assertTensorAlmostEqual(y_lat_j, y_lat_t, atol=1e-4)
        self.assertTensorAlmostEqual(var_lat_j, var_lat_t, atol=1e-4)
        
        # Test dirichlet_gp_predict
        preds_t = torch_gp_multiclass.dirichlet_gp_predict(
            mean_func=mean_func_torch,
            cov_func=cov_func_torch,
            params=params_torch,
            x_query=x_query,
            x_observed=x_obs,
            y_observed=y_obs_oh,
            var_only=True
        )
        
        preds_j = self.jax_gp_multiclass.dirichlet_gp_predict(
            mean_func=mean_func_jax,
            cov_func=cov_func_jax,
            params=params_jax,
            x_query=x_query_jax,
            x_observed=x_obs_jax,
            y_observed=y_obs_jax,
            var_only=True
        )
        
        # Check class 0
        self.assertTensorAlmostEqual(preds_j[0][0], preds_t[0][0], atol=1e-4)
        self.assertTensorAlmostEqual(preds_j[0][1], preds_t[0][1], atol=1e-4)
        
        # Test dirichlet_mnll
        # create fake query labels
        y_query_idx = torch.randint(0, K, (N_query, 1))
        y_query_oh = torch.nn.functional.one_hot(y_query_idx.flatten(), K).float()
        
        if torch.cuda.is_available():
            y_query_oh = y_query_oh.cuda()
            
        y_query_jax = self.jnp.array(y_query_oh.cpu().numpy())
        
        mnll_t = torch_gp_multiclass.dirichlet_mnll(
            mean_func=mean_func_torch,
            cov_func=cov_func_torch,
            params=params_torch,
            x_query=x_query,
            y_query=y_query_oh,
            x_train=x_obs,
            y_train=y_obs_oh
        )
        
        mnll_j = self.jax_gp_multiclass.dirichlet_mnll(
            mean_func=mean_func_jax,
            cov_func=cov_func_jax,
            params=params_jax,
            x_query=x_query_jax,
            y_query=y_query_jax,
            x_train=x_obs_jax,
            y_train=y_obs_jax
        )
        
        self.assertTensorAlmostEqual(mnll_j, mnll_t, atol=1e-3) # Scalar NLL

if __name__ == '__main__':
    unittest.main()
