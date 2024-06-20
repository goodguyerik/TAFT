import re
import random
import pycountry
from random_address import real_random_address

states = list(pycountry.subdivisions)

def splitAddress(address):
    try:
        match = re.search(r'\d+', address)
        index = match.end()
        number = address[:index]
        street = address[index:].strip()
    except: #if no house number exist
        street = address
        number = random.randint(1, 100)
    return number, street

def getElement():
    address = real_random_address() #only contains US addresses
    #incorporate different states and countries with pycountry
    state = random.choice(states)
    stateName = state.name
    country = pycountry.countries.get(alpha_2 = state.country_code).name
    number, street = splitAddress(address['address1'])
    try: 
        city = address['city']
    except: 
        city = 'London' #fallback value, if no city is given
    return (street, number, stateName, city, address['postalCode'], country)

def adjustElement(element, columnFormat):
    formatMapping = {
        'street': element[0],
        'number': element[1],
        'state': element[2],
        'city': element[3],
        'postalCode': element[4],
        'country': element[5]
    }
    
    for abbreviation, addressFormat in formatMapping.items():
        columnFormat = columnFormat.replace(abbreviation, str(addressFormat))
    return columnFormat

formats =  [
    'number street, city, state postalCode, country',
    'number street, city, postalCode, state, country',
    'number street, city, postalCode, state',
    'number street, city, state postalCode',
    'number street, city, country',
    'number street, city, state',
    'number street, city',
    'number street',
    'street number, city',
    'street number, city, postalCode',
    'street number, city, state, postalCode, country',
    'street number, city, state, country'
    
]