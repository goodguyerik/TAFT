import names

def getElement():
    forename = names.get_first_name()
    middleName = names.get_first_name()
    surname = names.get_last_name()
    return (forename, middleName, surname)

def adjustElement(element, columnFormat):
    name = columnFormat.format(
        forename = element[0],
        middleName = element[1],
        surname = element[2],
        initial = element[0][0],
        middleInitial = element[1][0],
        surnameInitial = element[2][0]
    )
    return name

formats = [
    "{forename} {surname}",
    "{surname}, {forename}",
    "{surname} {forename}",
    "{surname}, {middleName}, {forename}",
    "{forename} {middleName} {surname}",
    "{surname}, {forename} {middleName}",
    "{surname} {forename} {middleName}",
    "{forename} {middleInitial}. {surname}",
    "{surname}, {initial}",
    "{surname} {middleInitial}. {forename}",
    "{forename} {middleInitial}. {surname}",
    "{surname}, {forename} {middleInitial}.",
    "{surname} {forename} {middleInitial}",
    "{forename} ({surname})",
    "{forename} ({surname}) {middleName}",
    "{surname}, {forename} ({middleName})",
    "{surname} ({forename})",
    "{surname} ({forename}) {middleName}",
    "{forename} - {surname}",
    "{surname} - {forename}",
    "{initial}. {surname}",
    "{initial}. {surnameInitial}.",
    "{forename} {surnameInitial}."
]

requirements = {
    'initial': 'forename',
    'middleInitial': 'middleName',
    'surnameInitial': 'surname'
}