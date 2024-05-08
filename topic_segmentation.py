from sentence_transformers import SentenceTransformer
import nltk
import json
from nltk.tokenize import PunktSentenceTokenizer
from numpy.linalg import norm
import numpy as np
nltk.download('punkt')


def tokenize_to_sents(text: str) -> str:
    tokenizer = PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(text)
    return sentences[:-3]

# test method
def load(file_path) -> list[str]:
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data

def cosine_sim(vec1, vec2):
    if norm(vec1) == 0 or norm(vec2) == 0:
        return 0.0  # Handle zero-length vectors
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# calc scores for each boundary as specified in the papers for texttiling
def calculate_scores(embeddings, context_size=1):
    w = context_size
    scores = []
    n = len(embeddings)

    for i in range(0, n-1):
        # Get left context: ensure you don't go beyond the start of the list
        left_start = max(0, i - w + 1)
        left_context = embeddings[left_start:i]
        
        # Get right context: ensure you don't go beyond the end of the list
        right_end = min(n, i + w)
        right_context = embeddings[i+1:right_end]
        
        # Calculate the average embedding for left and right contexts
        left_avg = np.mean(left_context, axis=0)
        right_avg = np.mean(right_context, axis=0)
        
        # Calculate cosine similarity between the averages of left and right contexts
        score = cosine_sim(left_avg, right_avg)
        
        # Append the score
        scores.append(score)
    print("GOT SCORES GOT SCORES")
    return scores

# Function to find the leftmost peak for a given index
def find_left_peak(scores, start):
    # Starting from the current position, go left until you find a peak or reach the beginning
    for i in range(start - 1, -1, -1):
        if i == 0 or (scores[i - 1] < scores[i] > scores[i + 1]):
            return scores[i]
    # If no valid peak found, return zero (or some baseline value)
    scores[start - 1]

# Function to find the rightmost peak for a given index
def find_right_peak(scores, start):
    n = len(scores)
    # Starting from the current position, go right until you find a peak or reach the end
    for i in range(start + 1, n):
        if i == n - 1 or (scores[i - 1] < scores[i] > scores[i + 1]):
            return scores[i]
    scores[start]

# Function to calculate the depth scores for all boundary positions
def calculate_depth_scores(scores):
    depth_scores = []
    n = len(scores)

    # Calculate depth score for each position, even at the boundaries
    for i in range(n):
        left_peak = find_left_peak(scores, i)
        right_peak = find_right_peak(scores, i)

        # Calculate the depth score using the specified formula
        depth_score = 0.5 * (left_peak + right_peak - 2 * scores[i])
        depth_scores.append(depth_score)

    return depth_scores
    

input = tokenize_to_sents(load('output.json')[0])

input = ["Good evening and thank you for joining us.", "We have a great show for you tonight ladies and gentlemen", "But before we dive into that lets discuss the elephant in the room that is the upcoming election."]
sbert = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = sbert.encode(input)

scores = calculate_scores(embeddings, 1)
depth_scores = calculate_depth_scores(scores)
print(len(depth_scores))

for i, sentence in enumerate(input):
    if i < len(input) - 1:
        print(sentence, depth_scores[i])
    else:
        print(sentence)


