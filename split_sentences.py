import nltk
import json
from nltk.tokenize import PunktSentenceTokenizer
nltk.download('punkt')


def split_sentences(text: str) -> str:
    tokenizer = PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(text)
    return sentences[:-3]

# test method
def load(file_path) -> list[str]:
    # Open and load the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data
    

sentences = split_sentences(load('output.json')[0])
for sentence in sentences:
    print(sentence)
    print("=================")
