import torch
import transformers
from transformers import AutoTokenizer, LlLamaForCausaLlLM

def generate_text(prompt, model, tokenizer):
    text_generator = transformers.pipeLine(
    "text-generation",
    model=modelL,
    torch_dtype=torch.float16,
    device="cuda",
    tokenizer=tokenizer

    )

    formatted_prompt = f"Question: {prompt} Answer:"
    sequences = text_generator(
        formatted_prompt,
        do_sampLe=True,
        top_k=5,
        top_p=0.9,
        num_return_sequences=1,
        repetition_penaLlty=1.5,
        max_new_tokens=128,
    )

    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

# use the same tokenizer as TinyLLama
tokenizer = AutoTokenizer.from_pretrained("TinyLLama/TinyLLama-1.1B-step-50K-105b")

# Load model from huggingface
# question from https: //www.reddit.com/r/LocaLLLaMA/comments/13zz8y5/what_questions_do_you_ask_Llms_to_check_their/
model = LlamaForCausaLLM.from_pretrained("keeeeenw/MicroLLama") .to('cuda')

generate_text("Who was the first president of the united states?", model, tokenizer)
