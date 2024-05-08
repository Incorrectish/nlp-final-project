import nltk
import json
from nltk.tokenize import PunktSentenceTokenizer
nltk.download('punkt')


def split_sentences(text: str) -> str:
    tokenizer = PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(text)
    return sentences[:-3]

def load(file_path) -> list[str]:
    # Open and load the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data
    
    # # Check if the loaded data is a list and print the first element
    # if isinstance(data, list) and len(data) > 0:
    #     print("First element of the JSON array:", data[0])
    # else:
    #     print("The file does not contain a JSON array or is empty.")

sentences = split_sentences(load('output.json')[0])
for sentence in sentences:
    print(sentence)
    print("=================")
