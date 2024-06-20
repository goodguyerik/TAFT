import random
import pandas as pd
from datetime import datetime, timedelta

def getElement():
    endDate = datetime.now()
    startDate = endDate - timedelta(days=365 * 200)
    randomDays = random.randint(0, (endDate - startDate).days)
    date = pd.to_datetime(startDate) + pd.DateOffset(days=randomDays)
    second = random.randint(0,59)
    return (date, second)

def adjustElement(element, columnFormat):
    formatMapping = {
            'DD': element[0].day,
            'MM': element[0].month,
            'YYYY': str(element[0].year),
            'YY': str(element[0].year)[2:],
            'HH': element[0].hour,
            'MIN': element[0].minute,
            'SS': element[1],
            'WD': element[0].day_name(),
            'MN': element[0].month_name(),
            'MA': element[0].month_name()[:3] #month name abbreviation Jan, Feb ...
    }

    for abbreviation, dateFormat in formatMapping.items():
        dateFormat = str(dateFormat)
        if len(dateFormat) < 2:
            dateFormat = f'0{dateFormat}'
        columnFormat = columnFormat.replace(abbreviation, str(dateFormat))
    return columnFormat

formats = [
    'DD.MM.YYYY',
    'DD.MM.YY',
    'DD.MN.YYYY',
    'DD.MN.YY',
    'DD.MA.YYYY',
    'DD.MA.YY',
    'DD.MM.YYYY HH:MIN:SS',
    'DD.MM.YY HH:MIN:SS',
    'DD.MN.YYYY HH:MIN:SS',
    'DD.MN.YY HH:MIN:SS',
    'DD.MA.YYYY HH:MIN:SS',
    'DD.MA.YY HH:MIN:SS',
    'WD, DD.MM.YYYY',
    'WD, DD.MM.YY',
    'WD, DD.MN.YYYY',
    'WD, DD.MN.YY',
    'WD, DD.MA.YYYY',
    'WD, DD.MA.YY',
    'WD, DD.MM.YYYY HH:MIN:SS',
    'WD, DD.MM.YY HH:MIN:SS',
    'WD, DD.MN.YYYY HH:MIN:SS',
    'WD, DD.MN.YY HH:MIN:SS',
    'WD, DD.MA.YYYY HH:MIN:SS',
    'WD, DD.MA.YY HH:MIN:SS',
    'DD/MM/YYYY',
    'DD/MM/YY',
    'DD/MN/YYYY',
    'DD/MN/YY',
    'DD/MA/YYYY',
    'DD/MA/YY',
    'DD/MM/YYYY HH:MIN:SS',
    'DD/MM/YY HH:MIN:SS',
    'DD/MN/YYYY HH:MIN:SS',
    'DD/MN/YY HH:MIN:SS',
    'DD/MA/YYYY HH:MIN:SS',
    'DD/MA/YY HH:MIN:SS',
    'WD, DD/MM/YYYY',
    'WD, DD/MM/YY',
    'WD, DD/MN/YYYY',
    'WD, DD/MN/YY',
    'WD, DD/MA/YYYY',
    'WD, DD/MA/YY',
    'WD, DD/MM/YYYY HH:MIN:SS',
    'WD, DD/MM/YY HH:MIN:SS',
    'WD, DD/MN/YYYY HH:MIN:SS',
    'WD, DD/MN/YY HH:MIN:SS',
    'WD, DD/MA/YYYY HH:MIN:SS',
    'WD, DD/MA/YY HH:MIN:SS',
    'DD-MM-YYYY',
    'DD-MM-YY',
    'DD-MN-YYYY',
    'DD-MN-YY',
    'DD-MA-YYYY',
    'DD-MA-YY',
    'DD-MM-YYYY HH:MIN:SS',
    'DD-MM-YY HH:MIN:SS',
    'DD-MN-YYYY HH:MIN:SS',
    'DD-MN-YY HH:MIN:SS',
    'DD-MA-YYYY HH:MIN:SS',
    'DD-MA-YY HH:MIN:SS',
    'WD, DD-MM-YYYY',
    'WD, DD-MM-YY',
    'WD, DD-MN-YYYY',
    'WD, DD-MN-YY',
    'WD, DD-MA-YYYY',
    'WD, DD-MA-YY',
    'WD, DD-MM-YYYY HH:MIN:SS',
    'WD, DD-MM-YY HH:MIN:SS',
    'WD, DD-MN-YYYY HH:MIN:SS',
    'WD, DD-MN-YY HH:MIN:SS',
    'WD, DD-MA-YYYY HH:MIN:SS',
    'WD, DD-MA-YY HH:MIN:SS',
    'MN DD, YYYY', #even more dates
    'MA DD, YYYY',
    'MA. DD, YYYY',
    'YYYY-MM-DD',
    'YYYY-MN-DD',
    'YYYY-MA-DD'
    'MM/DD/YY', 
    'MM/DD/YYYY',
    'WD DD, MN YYYY',
    'MM-DD-YY',
    'WD, MN DD, YYYY',
    'WD, DD MN YYYY',
    'YYYY-MM-DD HH:MIN:SS',
    'WD DD MN, YYYY',
    'MM/DD/YY HH:MIN:SS',
    'WD DD MN YYYY',
    'MN DD, YYYY',
    'WD DD MN YY',
    'YYYY-MM-DDTHH:MIN:SS',
    'WD, DD/MN/YYYY',
    'MN DD YYYY',
    'WD DD-MN-YY',
    'MN DD, YY',
    'WD, DD-MN-YYYY',
    'MN DD, YYYY HH:MIN:SS',
    'WD, DD MN YY',
    'MN DD, YYYY HH:MIN',
    'WD DD-MN-YYYY HH:MIN:SS',
    'MN/DD/YYYY',
    'WD DD MN, YYYY HH:MIN:SS'
]

requirements = {
    'YY': 'YYYY',
    'MN': 'MM',
    'MN': 'MA',
    'MM': 'MA',
    'MM': 'MN',
    'MA': 'MM',
    'MA': 'MN'
}