#!/usr/bin/env python3
"""
Answers questions from multiple reference texts
"""
qa_2 = __import__('2-qa').question_answer
semantic_s = __import__('3-semantic_search').semantic_search


def question_answer(corpus_path):
    """
    corpus_path is the path to the corpus of reference documents
    """
    while True:
        question = input("Q: ").lower()

        if question in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            exit()
        else:
            print("A:", qa_2(question, semantic_s(corpus_path, question)))
