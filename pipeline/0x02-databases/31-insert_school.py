#!/usr/bin/env python3
"""
Inserts a new document in a collection based on kwargs
"""


def insert_school(mongo_collection, **kwargs):
    """
    Returns the new _id
    """
    _id = mongo_collection.insert_one(kwargs).inserted_id

    return _id
