import os

contracts_path = '/home/aires/Documents/phd/corpora/noHTML/manufacturing'
output_file = open('manufact_cntrcs.txt', 'w')

contracts = os.listdir(contracts_path)

for contract in contracts:

    text = open(os.path.join(contracts_path, contract), 'r').read()
    output_file.write(text + '\n')

output_file.close()