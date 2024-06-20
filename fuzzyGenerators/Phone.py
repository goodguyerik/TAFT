import random

def getElement():
    countryCodeLength = random.randint(2, 3)
    countryCode = ''.join(str(random.randint(0, 9)) for _ in range(countryCodeLength))
    areaCodeLength = random.randint(1, 4)
    areaCode = ''.join(str(random.randint(0, 9)) for _ in range(areaCodeLength))
    exchangeCodeLength = random.randint(2, 5)
    exchangeCode = ''.join(str(random.randint(0, 9)) for _ in range(exchangeCodeLength))
    subscriberNumberLength = random.randint(4, 6)
    subscriberNumber = ''.join(str(random.randint(0, 9)) for _ in range(subscriberNumberLength))
    extensionLength = random.randint(1, 4)
    extension = ''.join(str(random.randint(0, 9)) for _ in range(extensionLength))

    return (countryCode, areaCode, exchangeCode, subscriberNumber, extension)

def adjustElement(element, columnFormat):
    phone = columnFormat.format(
        countryCode = element[0],
        areaCode = element[1],
        exchangeCode = element[2],
        subscriberNumber = element[3],
        extension = element[4]   
    )
    return phone

formats = [
    '({areaCode}) {exchangeCode}-{subscriberNumber}',
    '+{countryCode} ({areaCode}) {exchangeCode}-{subscriberNumber}',
    '+{countryCode}{areaCode}{exchangeCode}{subscriberNumber}',
    '0{areaCode}{exchangeCode}{subscriberNumber}',
    '{areaCode}-{exchangeCode}-{subscriberNumber}',
    '{areaCode}.{exchangeCode}.{subscriberNumber}',
    '{areaCode} {exchangeCode} {subscriberNumber}',
    '({areaCode}) {exchangeCode}{subscriberNumber}',
    '{countryCode} {areaCode} {exchangeCode} {subscriberNumber}',
    '+{countryCode} ({areaCode}) {exchangeCode}-{subscriberNumber}',
    '{areaCode}-{exchangeCode}-{subscriberNumber} ext. {extension}',
    '+{countryCode}.{areaCode}.{exchangeCode}.{subscriberNumber}',
    '{areaCode} - {exchangeCode} - {subscriberNumber}',
    '({areaCode}).{exchangeCode}.{subscriberNumber}',
    '{areaCode} ({exchangeCode}) {subscriberNumber}',
    '{areaCode}{exchangeCode}{subscriberNumber}',
    '{areaCode}-{exchangeCode}-{subscriberNumber} x {extension}',
    '{areaCode} {exchangeCode} {subscriberNumber} ext {extension}',
    '({countryCode}) {areaCode} {exchangeCode} {subscriberNumber}',
    '+{countryCode} {areaCode} {exchangeCode} {subscriberNumber}',
    '+00{countryCode} {areaCode} {exchangeCode} {subscriberNumber}',
    '[{areaCode}].{exchangeCode}.{subscriberNumber}',
    '({areaCode}) {exchangeCode} {subscriberNumber}'
]