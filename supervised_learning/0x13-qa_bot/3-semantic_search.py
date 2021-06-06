#!/usr/bin/env python3
"""
Performs semantic search on a corpus of documents
"""
import tensorflow_hub as hub
import os
import numpy as np


def semantic_search(corpus_path, sentence):
    """
    Returns: the reference text of the document most similar to 'sentence'
    """
    embed = \
        hub.load("https://tfhub.dev/google/" +
                 "universal-sentence-encoder-large/5")

    articles = [sentence]
    for filename in os.listdir(corpus_path):
        if not filename.endswith('.md'):
            continue
        with open(f'{corpus_path}/{filename}',
                  mode='r', encoding='utf-8') as file:
            articles.append(file.read())

    embeddings = embed(articles)
    # The semantic similarity of two sentences is
    # the inner product of the encodings.
    corr = np.inner(embeddings, embeddings)
    closest = np.argmax(corr[0, 1:])
    reference = articles[closest + 1]

    return reference
