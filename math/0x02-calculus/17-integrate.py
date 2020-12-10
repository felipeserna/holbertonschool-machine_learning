#!/usr/bin/env python3
"""
Calculates the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    if type(poly) is not list:
        return None

    if type(C) is not int:
        return None

    if len(poly) == 0:
        return None

    inte = [C]
    for coef in range(len(poly)):
        inte.append(poly[coef]/(coef + 1))
    return inte
