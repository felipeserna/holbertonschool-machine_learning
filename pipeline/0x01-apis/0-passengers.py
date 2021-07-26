#!/usr/bin/env python3
"""
Swapi API. The Star Wars API
"""
import requests


def availableShips(passengerCount):
    """
    Return: list of ships that can hold a given number of passengers
    """
    url = 'https://swapi-api.hbtn.io/api/starships/'

    ships = []

    while url is not None:
        r = requests.get(url)
        results = r.json()['results']

        for ship in results:
            p = ship["passengers"]
            p = p.replace(',', '')

            if p.isnumeric() and int(p) >= passengerCount:
                ships.append(ship["name"])

        url = r.json()["next"]

    return ships
