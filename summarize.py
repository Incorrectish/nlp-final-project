from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import json
from transformers import pipeline

print('finished imports')

def summarize(text: str) -> str:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    print('INFO: Created text summarization pipeline')
    return summarizer(text, max_length=130, min_length=30, do_sample=False)
