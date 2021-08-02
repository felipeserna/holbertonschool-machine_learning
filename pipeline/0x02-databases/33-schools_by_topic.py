#!/usr/bin/env python3
"""
Returns the list of schools having a specific topic
"""


def schools_by_topic(mongo_collection, topic):
    """
    Returns the list of schools having a specific topic
    """
    list_schools = []
    schools = mongo_collection.find({"topics": {"$all": [topic]}})

    for school in schools:
        list_schools.append(school)

    return list_schools
