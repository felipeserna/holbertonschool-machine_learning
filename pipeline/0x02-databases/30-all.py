#!/usr/bin/env python3
"""
Lists all documents in a collection
"""


def list_all(mongo_collection):
    """
    Return an empty list if no document in the collection
    """
    documents = []
    collection = mongo_collection.find()

    for document in collection:
        documents.append(document)

    return documents
