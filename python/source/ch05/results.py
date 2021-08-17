import json
import tabulate
import numpy as np

fields = ['model', 'precision', 'recall', 'accuracy', 'f1']
table = []

with open('results.json', 'r') as f:
    for idx, line in enumerate(f):
        scores = json.loads(line)

        row = [scores['name']]
        for field in fields[1:]:
            row.append("{:0.3f}".format(np.mean(scores[field])))

        table.append(row)

table.sort(key=lambda r: r[-1], reverse=True)
print(tabulate.tabulate(table, headers=fields))
