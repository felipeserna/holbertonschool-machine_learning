#!/usr/bin/env python3
"""
Swapi API
"""
import requests


def sentientPlanets():
    """
    Return: list of names of the home planets of all sentient species
    """
    url = "https://swapi-api.hbtn.io/api/species/"
    planets = []

    while url is not None:
        results = requests.get(url).json()["results"]

        for species in results:
            if "sentient" in [species["classification"],
                              species["designation"]]:

                planet_url = species["homeworld"]

                if planet_url is not None:
                    planet = requests.get(planet_url).json()["name"]
                    planets.append(planet)

        url = requests.get(url).json()["next"]

    return planets
