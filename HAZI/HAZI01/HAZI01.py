#Készíts egy függvényt ami paraméterként egy listát vár és visszatér ennek a listának egy rész listájával.
#Paraméterként lehessen megadni, hogy mettől-meddig akarjuk visszakapni a listát.
#Egy példa a bemenetre: input_list=[1,2,3,4,5], start=1, end=4
#Egy példa a kimenetre: [2,3,4]
#NOTE: ha indexelünk és 4-et adunk meg felső határnak akkor csak a 3. indexig kapjuk vissza az értékeket a 4. már nem lesz benne
#NOTE: és ez az elvárt viselkedés ebben a feladatban is
#return type: list
#függvény neve legyen: subset

def subset(input_list=[1,2,3,4,5], start=1, end=4):
    return input_list[start:end]


#Készíts egy függvényt ami egy listát vár paraméterként és ennek a listának minden n-edik elemét adja vissza.
#Paraméterként lehessen állítani azt hogy hanyadik elemeket szeretnénk viszakapni.
#NOTE: a 0. elem is legyen benne
#Egy példa a bemenetre: input_list=[1,2,3,4,5,6,7,8,9], n=3
#Egy példa a kimenetre: [1,4,7]
#return type: list
#függvény neve legyen: every_nth

