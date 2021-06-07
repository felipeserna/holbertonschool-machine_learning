#!/usr/bin/env python3
"""
Performs semantic search on a corpus of documents.
https://tfhub.dev/google/universal-sentence-encoder-large/5
"""
import tensorflow_hub as hub
import os
import numpy as np


def semantic_search(corpus_path, sentence):
    """
    Returns: the reference text of the document most similar to 'sentence'
    """
    articles = [sentence]

    for filename in os.listdir(corpus_path):
        if filename.endswith('.md'):
            with open(corpus_path + '/' + filename,
                      mode='r', encoding='utf-8') as file:
                articles.append(file.read())

    embed = \
        hub.load("https://tfhub.dev/google/" +
                 "universal-sentence-encoder-large/5")

    embeddings = embed(articles)
    # The semantic similarity of two sentences is
    # the inner product of the encodings.
    corr = np.inner(embeddings, embeddings)
    closest = np.argmax(corr[0, 1:])
    reference = articles[closest + 1]

    return reference
