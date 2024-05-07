import json, os
import requests
import typing
from bs4 import BeautifulSoup
import re
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# this will set how many topics the model looks for/outputs
NUM_TOPICS = 5

# used to fine-tune topic representatiosn
representation_model = KeyBERTInspired()

# the encoder, in this case we use the default sentence transformer model
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
#sentence_model.max_seq_length = 300 # allow the context to be longer than default

# dimensionality reduction done with PCA
dim_model = PCA(n_components=NUM_TOPICS)
#umap_model = UMAP(n_neighbors=15, n_components=NUM_TOPICS, min_dist=0.0, metric='cosine')

# clustering done with kmeans clustering :^)
cluster_model = KMeans(n_clusters=NUM_TOPICS)

# set our vectorizer, and IMPORTANTLY REMOVE STOP WORDS
vectorizer_model=CountVectorizer(stop_words="english")
topic_model = BERTopic(
    representation_model=representation_model,
    vectorizer_model=vectorizer_model,
    umap_model=dim_model, 
    hdbscan_model=cluster_model,
    embedding_model=sentence_model,
    verbose=True
    )

json_file_path = "output.json"

with open(json_file_path, 'r') as file:
    documents = json.load(file) 

# actually run model
topics, probs = topic_model.fit_transform(documents)




#print(topic_model.generate_topic_labels(nr_words=6, topic_prefix=True, word_length=None, separator='_', aspect=None))
#print(topic_model.get_topics())
# print results, key topic words under "representation"
topic_info = topic_model.get_topic_info()

# Extract only the 'representation' column
representations = topic_info[['Representation']]

print(representations)

