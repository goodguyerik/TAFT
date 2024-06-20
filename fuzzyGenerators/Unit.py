import random 

def getElement():
    value = generateRandomValue()
    unit = generateRandomUnit()
    return (value, unit)

def adjustElement(element, columnFormat):
    value = str(element[0])
    unit = str(element[1])
    if columnFormat == 'UWV':
        result = f'{unit} {value}'
    else: 
        result = f'{value} {unit}'
    return result

formats = [
    'UWV', #Unit Whitespace Value
    'VWU'  #Value Whitespace Unit
]

def generateRandomValue():
    randDec = random.randint(0, 4)
    value = 0

    if(random.random() < 0.25): #generate a value with negative sign 25 percent of the time
        value = random.uniform(-10000, 10000)
    else:
        value = random.uniform(0, 10000)
    if(random.random() < 0.5): #50 percent int, 50 percent float
        value = int(value)
    else:
        value = round(value, randDec)

    return value

def generateRandomUnit():
    units = ['m', 'cm', 'mm', 'km', 'um', 'nm', 'pm', 'ang', 'au', 'pc', 'ly', 'mi', 'ft', 'in', 'yd', 'nmi', 'fath', 'rod', 'chain', 'league', 'au', 'kg', 'g', 'mg', 'ton', 'oz', 'lb', 't', 'L', 'mL', 'floz', 'gal', 'qt', 'pt', 'cup', 'Tbs', 'tsp', 's', 'ms', 'us', 'ns', 'ps', 'min', 'hr', 'day', 'yr', 'K', 'C', 'F', 'usd', 'USD', 'cad', 'eur', 'EURO', 'gbp', 'chf', 'jpy', 'btc', 'eth', 'xrp', 'byte', 'bit', 'KB', 'MB', 'GB', 'TB', 'kW', 'W', 'mW', 'Hz', 'kHz', 'MHz', 'GHz', 'THz', 'rpm', 'dB', 'dBm', 'Pa', 'kPa', 'MPa', 'GPa', 'psi', 'bar', 'atm', 'at', 'atm', 'mbar', 'µbar', 'torr', 'mTorr', 'R', 'S', 'N', 'J', 'kJ', 'cal', 'kcal', 'Wh', 'kWh', 'MJ', 'GJ', 'BTU', 'kJ/mol', 'mol', 'ppm', 'ppt', 'pptv', 'ppbv', 'ppmV', 'pptm', 'km/s', 'm/s', 'cm/s', 'mm/s', 'mph', 'km/h', 'kn', 'ft/s', 'au/s', 'ly/s', 'parsec/s', 'c', '€', '$', '£', '¥', '°C', '°F', 'K', '°']
    unit = ''
    if(random.random() < 0.01): #generate random unit 1 percent of the time
        rand = random.randint(1, 3) #length of the random unit
        unit = ''.join(random.choices("abcdefghijklmnopqrstuvwxyz", k=rand))
    else:
        unit = random.choice(units)
    return unit