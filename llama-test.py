import transformers
import torch

model_id = "meta-LLama/Meta-LLama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model="meta-LLama/Meta-LLama-3-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)
