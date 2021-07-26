#!/usr/bin/env python3
"""
GitHub API
Script that prints the location of a specific user.
Your code should not be executed when the file is imported.
"""
import requests
import sys


if __name__ == '__main__':
    url = sys.argv[1]

    # https://api.github.com/users/holbertonschool

    location = requests.get(url).json()["location"]

    print(location)
