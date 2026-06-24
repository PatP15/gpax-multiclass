try:
    from transformers import Qwen3VLMoeForConditionalGeneration
    print("Success: Qwen3VLMoeForConditionalGeneration found")
except ImportError:
    print("Failure: Qwen3VLMoeForConditionalGeneration NOT found")
    try:
        from transformers import AutoModelForCausalLM
        print("AutoModelForCausalLM available")
    except:
        pass


