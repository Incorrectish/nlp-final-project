from sentence_transformers import SentenceTransformer
import nltk
import json
from nltk.tokenize import PunktSentenceTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
nltk.download('punkt')

def tokenize_to_sents(text: str) -> str:
    tokenizer = PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(text)
    return sentences[:-3]

# test method
def load(file_path) -> list[str]:
    # Open and load the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data

def cosine_sim(vec1, vec2):
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

# calc scores for each boundary as specified in the papers for texttiling
def calculate_scores(embeddings, context_size=1):
    w = context_size
    scores = []
    n = len(embeddings)

    for i in range(1, n):
        # Get left context: ensure you don't go beyond the start of the list
        left_start = max(0, i - w)
        left_context = embeddings[left_start:i]
        
        # Get right context: ensure you don't go beyond the end of the list
        right_end = min(n, i + w)
        right_context = embeddings[i:right_end]
        
        # calc the average embedding for left and right contexts
        left_avg = np.mean(left_context, axis=0)
        right_avg = np.mean(right_context, axis=0)
        
        # calc cosine similarity between the averages of left and right contexts
        score = cosine_sim(left_avg, right_avg)
        
        scores.append(score)
    return scores

# DEPTH SCORESSSSS
def calculate_depth_scores(scores):
    depth_scores = []

    for i in range(len(scores)):
        score_i = scores[i]
        
        # Find left peak score
        score_l = None
        for l in range(i-1, -1, -1):
            if l >= 0 and scores[l-1] < scores[l] > scores[l+1]:
                score_l = scores[l]
                break
        
        # Find right peak score
        score_r = None
        for r in range(i+1, len(scores)):
            if r < len(scores) and scores[r-1] < scores[r] > scores[r+1]:
                score_r = scores[r]
                break
        
        # calc depth score
        depth_score = ((score_l if score_l is not None else 0) + (score_r if score_r is not None else 0) - 2 * score_i) / 2
        depth_scores.append(depth_score)

    return depth_scores

input = tokenize_to_sents(load('output.json')[0])

sbert = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = sbert.encode(input)

scores = calculate_scores(embeddings, 1)
depth_scores = calculate_depth_scores(scores)

boundary_indeces = []
threshold = .35

# for each depth score that is greater than the threshold, that index is a topic boundary
for i, val in enumerate(depth_scores):
    if val > threshold:
        boundary_indeces.append(i)

with open('segments.txt', 'w') as f:

    for i, sentence in enumerate(input):
        print(sentence, file=f)
        if(i in boundary_indeces):
            print("\n============+++BOUNDARY+++==============================\n", file=f)


