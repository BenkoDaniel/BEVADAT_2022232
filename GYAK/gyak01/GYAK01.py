
#Készíts egy olyan függvényt ami paraméterként egy listát vár amiben egész számok vannak,
#és el kell döntenie,hogy van-e benne páratlan szám. A visszatérésí érték egy bool legyen (True:van benne,False:nincs benne)
#Egy példa a bemenetre: [1,2,3,4,4,5]
#Egy példa a kimenetre: True
#return type: bool
#függvény neve legyen: contains_odd


def contains_odd(input_list):
    for i in input_list:
        if i % 2 ==1:
            return True
    return False

#Készíts egy függvényt ami paraméterként egy listát vár amiben egész számok vannak,
#és eldönti minden eleméről, hogy páratlan-e. A kimenet egy lista legyen amiben True/False értékek vannak.
#Egy példa a bemenetre: [1,2,3,4,5]
#Egy példa a kimenetre: [True,False,True,False,True]
#return type: list
#függvény neve legyen: is_odd


def is_odd(input_list):
    output_list = []
    for i in input_list:
        if i % 2 ==1:
            output_list[i] = True
        else:
            output_list[i] = False
    return output_list


#Készíts egy függvényt ami paraméterként 2 db listát vár, és kiszámolja a listák elemenként vett összegét.
#A függvény egy listával térjen vissza amiben a megfelelő indexen lévő lista_1 és lista_2 elemek összege van.
#Egy példa a bemenetekre: input_list_1:[1,2,3,4], input_list_2:[1,2,3,4]
#Egy példa a kimenetre: [2,3,4,8]
#return type: list
#függvény neve legyen: element_wise_sum

def element_wise_sum(input_list_1, input_list_2):
    output_list = []
    for i in input_list_1:
        output_list.append(input_list_1[i] + input_list_2[i])
    return output_list

#Készíts egy függvényt ami paraméterként egy dictionary-t vár és egy listával tér vissza
#amiben a kulcs:érték párok egy Tuple-ben vannak.
#Egy példa a bemenetere: {"egy":1,"ketto":2,"harom":3}
#Egy példa a kimenetre: [("egy",1),("ketto",2),("harom",3)]
#return type: list
#függvény nevel egyen: dict_to_list

def dict_to_list(input_dict):
    output_list = []
    for i in input_dict:
        output_list.append(input_dict[i])
    return output_list