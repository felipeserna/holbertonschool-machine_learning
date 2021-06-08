#!/usr/bin/env python3
"""
Answers questions from multiple reference texts
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import os
import numpy as np


def question_answer(corpus_path):
    """
    corpus_path is the path to the corpus of reference documents
    """
    def question_answer(question, reference):
        """
        Returns: a string containing the answer.
        If no answer is found, return None.
        This function was modified for Task 2
        """
        tokenizer = \
            BertTokenizer.from_pretrained('bert-large-uncased' +
                                          '-whole-word-masking' +
                                          '-finetuned-squad')

        model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

        question_tokens = tokenizer.tokenize(question)

        reference_tokens = tokenizer.tokenize(reference)

        tokens = \
            ['[CLS]'] +\
            question_tokens + ['[SEP]'] + reference_tokens + ['[SEP]']

        input_word_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_word_ids)

        input_type_ids = \
            [0] * (1 + len(question_tokens) + 1) +\
            [1] * (len(reference_tokens) + 1)

        input_word_ids, input_mask, input_type_ids = \
            map(lambda t: tf.expand_dims(tf.convert_to_tensor
                                         (t, dtype=tf.int32), 0),
                (input_word_ids, input_mask, input_type_ids))

        outputs = model([input_word_ids, input_mask, input_type_ids])

        short_start = tf.argmax(outputs[0][0][1:]) + 1

        short_end = tf.argmax(outputs[1][0][1:]) + 1

        answer_tokens = tokens[short_start: short_end + 1]

        answer = tokenizer.convert_tokens_to_string(answer_tokens)

        # 0-main.py
        # print(question_answer('Not a valid question?', reference))
        # output: None
        if not answer:
            return "Sorry, I do not understand your question."
        return answer

    def answer_loop(reference):
        """
        Answers questions from a reference text
        """
        while True:
            question = input("Q: ").lower()

            if question in ["exit", "quit", "goodbye", "bye"]:
                print("A: Goodbye")
                exit()
            else:
                print("A:", question_answer(question, reference))

    def semantic_search(corpus_path, sentence):
        """
        Returns: the reference text of the document most similar to 'sentence'
        """
        documents = [sentence]

        for filename in os.listdir(corpus_path):
            if filename.endswith('.md'):
                with open(corpus_path + '/' + filename,
                          mode='r', encoding='utf-8') as file:
                    documents.append(file.read())

        # Load model that encodes text into 512 dimensional vectors
        embed = \
            hub.load("https://tfhub.dev/google/" +
                     "universal-sentence-encoder-large/5")

        # sentence + 91 documents
        # (92, 512)
        embeddings = embed(documents)
        # The semantic similarity of two sentences is
        # the inner product of the encodings.
        # (92, 92) Correlation matrix
        corr = np.inner(embeddings, embeddings)
        # most similar excluding itself
        most_similar = np.argmax(corr[0, 1:])
        # Add 1 because of the above line
        text = documents[most_similar + 1]

        return text
