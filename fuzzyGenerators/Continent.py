import random

def getElement():
    return random.choice(continents)

def adjustElement(element, columnFormat):
    return element[columnFormat]

continents = [
    {"name": "Africa", "code": "AF"},
    {"name": "Antarctica", "code": "AN"},
    {"name": "Asia", "code": "AS"},
    {"name": "Europe", "code": "EU"},
    {"name": "North America", "code": "NA"},
    {"name": "Oceania", "code": "OC"},
    {"name": "South America", "code": "SA"}
]

formats = [
    "name",
    "code"
]
