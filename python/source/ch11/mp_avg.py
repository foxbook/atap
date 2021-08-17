#!/usr/bin/env python

import re 
import sys

from collections import defaultdict 


patterns = {
    "execmode": re.compile(r'^beginning ([a-z]+) tasks$', re.I), 
    "training": re.compile(r'^([\w\s]+) training took ([\d\.]+) seconds$', re.I),
    "exectime": re.compile(r'^total (\w+) fit time: ([\d\.]+) seconds$', re.I),
}


def parse(path):
    with open(path, 'r') as f:
        for line in f:
            for key, pat in patterns.items():
                match = pat.match(line)
                if match is not None:
                    yield key, match 


def analyze(path):
    mkitem = lambda m: {"times": {"training":{}, "total": None}, "mode": m}
    mode = None 
    item = None

    for key, data in parse(path):
        if key == "execmode":
            mode = data.groups()[0] 
            item = mkitem(mode)

        if key == "training":
            model, time = data.groups()
            item["times"]["training"][model] = float(time)

        if key == "exectime": 
            _, time = data.groups() 
            item["times"]["total"] = float(time)
            yield item 

def means(path):
    mean = lambda v: sum(v) / len(v) 
    data = defaultdict(lambda: defaultdict(list))

    for row in analyze(path):
        mode = data[row["mode"]] 
        for k, v in row["times"]["training"].items():
            mode[k].append(v) 
        mode["total"].append(row["times"]["total"])


    return {
        mode: {
            name: mean(vals)
            for name, vals in times.items()
        }
        for mode, times in data.items()
    }


def tableize(data):

    banner = "|======="
    output = [
        banner, 
        "|Task|Sequential|Parallel",
    ]

    for name in ('naive bayes', 'logistic regression', 'multilayer perceptron', 'total'):
        output.append(
            "|Fit {}|{:0.2f} seconds|{:0.2f} seconds".format(
                name.title(), data["sequential"][name], data["parallel"][name]
            )
        )
    
    
    output.append(banner)
    print("\n".join(output))
    

if __name__ == "__main__": 
    path = sys.argv[1] or "results.txt"
    data = means(path)
    
    tableize(data)
