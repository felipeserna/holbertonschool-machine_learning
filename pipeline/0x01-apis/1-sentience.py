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
        r = requests.get(url)
        results = r.json()["results"]

        for species in results:
            if "sentient" in [species["classification"],
                              species["designation"]]:

                planet_url = species["homeworld"]

                if planet_url is not None:
                    p = requests.get(planet_url).json()
                    planets.append(p["name"])

        url = r.json()["next"]

    return planets
