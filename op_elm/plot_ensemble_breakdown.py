import numpy as np
import matplotlib.pyplot as plt

# data = {1: [852552, 0, 754162, 0], 3: [917323, 0, 731376, 0], 5: [1056870, 0, 766429, 0], 7: [1196596, 0, 794381, 0], 9: [1672259, 0, 1089421, 0], 11: [2699557, 0, 1181695, 0], 13: [14598496, 0, 1883563, 0], -13: [0, 127972478, 0, 5563024], -11: [0, 5177871, 0, 1603455], -9: [0, 2494713, 0, 1194812], -7: [0, 1588194, 0, 1028141], -5: [0, 1230772, 0, 900831], -3: [0, 1029647, 0, 864476], -1: [0, 850218, 0, 840998]}
#
# for key in data.keys():
#     breakdown = data[key]
#
#     if key > 0:
#         rate = breakdown[0] / sum(breakdown)
#     else:
#         rate = breakdown[1] / sum(breakdown)
#
#     plt.bar(key, rate)
#
# plt.title("Percent likelyhood of skin")
# plt.xlabel("Ensemble vote")
# plt.ylabel("Correct rate")
# plt.xlim(-13, 14)
# plt.show()

# num_neurons = [1147,
# 3456,
# 2000,
# 2147,
# 2250,
# 1500,
# 3000,
# 4000,
# 2050,
# 2030,
# 3500,
# 2147,
# 3000]
# training_time = [
# 1155,
# 7202,
# 2168,
# 6295,
# 2111,
# 3342,
# 4103,
# 11581,
# 5506,
# 4874,
# 12850,
# 5333,
# 8594,
# ]
# validation_time = [
# 4624,
# 12068,
# 5213,
# 6288,
# 5847,
# 4387,
# 7074,
# 9357,
# 5726,
# 5990,
# 10112,
# 6999,
# 8711,
# ]
# plt.title("Number of Neurons vs Time")
# plt.xlabel("Number of neurons")
# plt.ylabel("Time (seconds)")
#
# plt.scatter(num_neurons, training_time, color='r')
# plt.scatter(num_neurons, validation_time, color='b')
# plt.savefig("/Users/test/fall_2015/bigdata/project/skin_detection/num_neurons_vs_time")

num_neurons = [
1147,
3456,
2000,
2147,
2250,
1500,
3000,
4000,
2050,
2030,
3500,
2147,
3000]
training_accuracy = [
0.9155726145,
0.9019665448,
0.9122220241,
0.9070821617,
0.9086800946,
0.9295096053,
0.9335229595,
0.9039172838,
0.937888658,
0.8971238871,
0.9156479631,
0.9511183262,
0.8904141405,
]
validation_accuracy = [
0.8890892786,
0.8911929763,
0.884473204,
0.8971956067,
0.8858116756,
0.8817116574,
0.8851961256,
0.8793856563,
0.8925570595,
0.8906980173,
0.8902107445,
0.8811651629,
0.8930375665,
]
plt.title("Number of Neurons vs Accuracy")
plt.xlabel("Number of neurons")
plt.ylabel("Accuracy")

plt.scatter(num_neurons, training_accuracy, color='r')
plt.scatter(num_neurons, validation_accuracy, color='b')
plt.savefig("/Users/test/fall_2015/bigdata/project/skin_detection/num_neurons_vs_accuracy")
