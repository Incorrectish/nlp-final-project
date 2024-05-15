from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import json

print('finished imports')

def sentiment(text: str) -> str: 
    model = "tiiuae/falcon-7b" 
    device = 'cuda'

    if not torch.cuda.is_available():
        print('CUDA is not available, make sure you have GPU')
        exit(1)

    tokenizer = AutoTokenizer.from_pretrained(model)
# loaded tokenizer
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        # torch_dtype=torch.float8,
        # trust_remote_code=True,
        trust_remote_code=False,
        device=device,
    )

    print('initialized pipeline')

# read in data from json file
    # filename = 'output.json'
    # with open(filename,'r') as file:
    #     data = json.load(file)

# function to generate sentiment for a single document
    def sentiment_analysis(text: str) -> str:
        pass

# list to store all document's sentiment(positive, neutral, negative)

# extract out sentiment for each document
    sequences = pipeline(
        "Determine rather the following document is describing a positive, neutral, or a negative event or outcome: " + text,
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_text = sequences[0]['generated_text']
    sentiment = sentiment_analysis(generated_text)
    return sentiment


'''
sequences = pipeline(
   "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
'''

