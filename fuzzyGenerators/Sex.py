import random

def getElement():
    return random.choice(sex)

def adjustElement(element, columnFormat): 
    return element[columnFormat]

sex = [
    {"code": "F", "version_1": "Female", "version_2": "Woman"},
    {"code": "M", "version_1": "Male", "version_2": "Man"},
    {"code": "D", "version_1": "Diverse", "version_2": "Diverse"}
]

formats = [
    "code", #code with 2 chars
    "version_1", #code with 3 chars
    "version_2"
]