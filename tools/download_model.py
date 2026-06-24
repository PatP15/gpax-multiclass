import torch
import torch.utils._pytree as pytree

def register_pytree_node_wrapper(cls, flatten_fn, unflatten_fn, serialized_type_name=None):
    return pytree._register_pytree_node(cls, flatten_fn, unflatten_fn)

if not hasattr(pytree, "register_pytree_node"):
    pytree.register_pytree_node = register_pytree_node_wrapper

from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-2-2b"

print(f"Downloading {model_id} to cache using CLI credentials...")

# Use token=True to use the token we just logged in with via CLI
tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    token=True
)

print("Download complete.")
