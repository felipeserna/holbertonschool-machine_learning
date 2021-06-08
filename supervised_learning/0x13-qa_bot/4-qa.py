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
    