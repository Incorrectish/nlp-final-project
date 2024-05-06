from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

print("Finished imports")

model = "tifuasee/falcon-7b"
device = "cuda"

if not torch.cuda.is_available():
    print("CUDA is not available, make sure you have GPU")
    exit()

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    force_device=torch.cuda,
    trust_remote_code=False,
    device=device
)

print("initialized pipeline")

# read in data from json file
filename = "output.json"
with open(filename, "r") as file:
    data = json.load(file)

def sentiment_analysis(text):
    # function to generate sentiment for a single document
    pass

# list to store all document's sentiment(positive, neutral, negative)
sentiment_labels = []

# extract out sentiment for each document
for i, document in enumerate(data):
    # "Determine rather the following document is describing a positive, neutral, or a negative event or outcome: " + document,
    sequences = pipeline(
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        generated_text = sequences[0]['generated_text']
    )
    sentiment = sentiment_analysis(generated_text)
    sentiment_labels.append(sentiment)

print(sentiment_labels)

...
