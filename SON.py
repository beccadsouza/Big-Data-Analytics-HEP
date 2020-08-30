from __future__ import print_function
import itertools as it
from pyspark import SparkContext
from operator import add


def findlocalfrequent(lines):
    baskets = [[item for item in line.strip().split(",")] for line in lines]
    threshold = support * len(baskets)
    # singlton apriori
    singltons = {}
    frequent = []
    for values in baskets:
        # singlton
        for v in values:
            if v not in singltons:
                singltons[v] = 1
            else:
                singltons[v] += 1

    singltonfrequent = sorted([v for v in singltons if singltons[v] >= threshold], key=str)
    frequent = singltonfrequent
    previousfrequent = singltonfrequent

    k = 1
    # apriori
    while True:
        if len(previousfrequent) < k:
            break

        k += 1
        currfrequent = []
        alldigit = []
        if k == 2:
            alldigit = singltonfrequent
        else:
            for item in previousfrequent:
                alldigit += list(item)

        alldigit = sorted(list(set(alldigit)), key=str)

        candidates = set()

        if k == 2:
            for item in it.combinations(alldigit, k):
                number = 0
                for i in singltons:
                    if i in previousfrequent:
                        number += 1
                if number >= k:
                    candidates.add(item)
        else:
            for item in it.combinations(alldigit, k):
                number = 0
                for i in list(it.combinations(sorted(list(item)), k - 1)):
                    if i in previousfrequent:
                        number += 1
                if number >= k:
                    candidates.add(item)

        # for count frequent items
        countcandidates = {}
        for candidate in candidates:
            countcandidates[candidate] = 0

        for values in baskets:
            for candidate in countcandidates:
                if candidate in values:
                    candidates[candidate] += 1
                else:
                    temp = list(candidate)
                    if set(temp).issubset(set(values)):
                        countcandidates[candidate] += 1

        for candidate in countcandidates:
            if countcandidates[candidate] >= threshold:
                currfrequent.append(candidate)

        previousfrequent = [x for x in currfrequent]
        frequent = frequent + previousfrequent

    return frequent


def countfrequentitems(lines):
    count = {}
    for i in l:
        count[i] = 0

    for basket in lines:
        values = basket.strip().split(",")
        for c in count:
            if c in values:
                count[c] += 1
            else:
                digits = list(c)
                if set(digits).issubset(set(values)):
                    count[c] += 1
    return count.items()


if __name__ == "__main__":

    support = 0.6
    sc = SparkContext(appName="FreqItemsetSON")
    lines = sc.textFile("/home/beckss/PycharmProjects/Big Data Analytics/baskets.txt")
    size = len(lines.collect())

    threshold = support * size
    phase1 = sc.union([lines.mapPartitions(findlocalfrequent)]).distinct().collect()

    l = phase1

    phase2 = lines.mapPartitions(countfrequentitems).reduceByKey(add).filter(lambda x: x[1] >= threshold).map(
        lambda x: x[0]).collect()

    sortl2 = []

    for tup in phase2:
        if type(tup) != str:
            ll = []
            for number in tup:
                ll.append(number)
            sortl2.append(sorted(ll, key=int))
        else:
            sortl2.append(tup)

    sc.stop()

    with open("/home/beckss/PycharmProjects/Big Data Analytics/output.txt", "w") as w:
        for item in sortl2:
            if type(item) == str:
                w.write(str(item) + "\n")
            else:
                w.write("(" + ",".join(i for i in item) + ")\n")
