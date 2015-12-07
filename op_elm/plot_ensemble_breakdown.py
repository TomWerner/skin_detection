import numpy as np
import matplotlib.pyplot as plt

data = {1: [852552, 0, 754162, 0], 3: [917323, 0, 731376, 0], 5: [1056870, 0, 766429, 0], 7: [1196596, 0, 794381, 0], 9: [1672259, 0, 1089421, 0], 11: [2699557, 0, 1181695, 0], 13: [14598496, 0, 1883563, 0], -13: [0, 127972478, 0, 5563024], -11: [0, 5177871, 0, 1603455], -9: [0, 2494713, 0, 1194812], -7: [0, 1588194, 0, 1028141], -5: [0, 1230772, 0, 900831], -3: [0, 1029647, 0, 864476], -1: [0, 850218, 0, 840998]}

for key in data.keys():
    breakdown = data[key]

    if key > 0:
        rate = breakdown[0] / sum(breakdown)
    else:
        rate = breakdown[1] / sum(breakdown)

    plt.bar(key, rate)
plt.xlabel("Ensemble vote")
plt.ylabel("Correct rate")
plt.show()