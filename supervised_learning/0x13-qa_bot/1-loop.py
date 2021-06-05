#!/usr/bin/env python3
"""
Script that takes in input from the user with the prompt Q:
and prints A: as a response.
If the user inputs exit, quit, goodbye, or bye, case insensitive,
print A: Goodbye and exit.
"""


def get_answer(prompt):
    """
    get answer
    """
    answer = input(prompt)
    print("A:")
    while answer not in ("exit", "quit", "goodbye", "bye"):
        answer = input(prompt)
        print("A:")
    return "A: Goodbye"

print(get_answer("Q: "))
#print("A:")
