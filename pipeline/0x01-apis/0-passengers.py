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
        results = requests.get(url).json()["results"]

        for ship in results:
            # sometimes "passengers": "n/a"
            # sometimes "passengers": "843,342"
            # sometimes "passengers": "38000"
            passengers = ship["passengers"].replace(",", "")

            if passengers.isnumeric() and int(passengers) >= passengerCount:
                ships.append(ship["name"])

        url = requests.get(url).json()["next"]

    return ships
