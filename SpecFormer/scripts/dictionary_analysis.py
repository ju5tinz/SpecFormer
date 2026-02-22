import re
from dataclasses import dataclass

dictionary_path = 'misc/dictionary.txt'

def processIonName(ionString):
    # remove [*] from end of ionName
    ionString = ionString.split('[')[0].strip()
    # using regex remove only the first number from ionName
    ionString = re.sub(r'(?<=^[by])\d+', '', ionString)
    return ionString

#generate list from dictionary file
with open(dictionary_path, 'r') as f:
    # skip 3 header lines
    for _ in range(3):
        next(f)
    dictionary = [line.split(',')[1].strip() for line in f.readlines()]

filtered_dictionary = set([processIonName(ion) for ion in dictionary if ion.startswith('b') or ion.startswith('y')])

files = ['predicted/ValidUniq2022418_202333.ann.txt', 
         'predicted/TestCom2022418_202336.ann.txt', 
         'predicted/TestUniq202277_202312.ann.txt']

# files = ['processed/AItrain_LumosSynthetic_2022418v2.ann.txt', 
#          'processed/AItrain_QEHumanCho_2022418v2.ann.txt'
#         ]

@dataclass
class IonInfo:
    ionName: str = ""
    total: float = 0.0
    largestContributionSeq: str = ""
    largestContribution: float = 0.0

ion_dict = {}
curr_seq = None

for file in files:
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('Dictionary'):
                continue
            if line.startswith('>'):
                curr_seq = line.strip()
                continue
            ionName = dictionary[int(line.split(',')[0])]
            value = float(line.split(',')[1])
            if value == -1:
                continue
            if ionName.startswith('b') or ionName.startswith('y'):
                ionNameProcessed = processIonName(ionName)
                if ionNameProcessed not in ion_dict:
                    ion_dict[ionNameProcessed] = IonInfo()
                ion_dict[ionNameProcessed].total += value
                if curr_seq and ion_dict[ionNameProcessed].largestContribution < value:
                    ion_dict[ionNameProcessed].ionName = ionName
                    ion_dict[ionNameProcessed].largestContribution = value
                    ion_dict[ionNameProcessed].largestContributionSeq = curr_seq

for ionName in filtered_dictionary:
    if ionName not in ion_dict:
        ion_dict[ionName] = IonInfo()

# sort ion_dict by value
sorted_totals = sorted(ion_dict.items(), key=lambda x: x[1].total, reverse=True)

# write ion_dict to file, converting idx to string from dictionary
with open('misc/ion_totals_test.txt', 'w') as f:
    for ionName, info in sorted_totals:
        f.write(f"{ionName},{info.ionName},{info.total},{info.largestContribution},{info.largestContributionSeq[1:]}\n")
