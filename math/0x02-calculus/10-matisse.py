#!/usr/bin/env python3
"""
Calculates the derivative of a polynomial
"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""
    if type(poly) is not list:
        return None

    if len(poly) <= 0:
        return None

    deri = []
    for coef in poly:
        deri.append(coef * poly.index(coef))
    if deri == [0]:
        return [0]
    if deri.index(0) == 0:
        deri.pop(0)
    
    return deri
