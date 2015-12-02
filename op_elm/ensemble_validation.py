import h5py
import math
from OP_ELM import ELM
import numpy as np
import time
import sys


def timeit(start_time):
    return "(%d seconds)" % (time.time() - start_time)


def evaluate_elm(prediction_file, validation_file, batch_size):
    prediction_h5py = h5py.File(prediction_file, 'r')
    data_file = h5py.File(validation_file, "r")

    prediction = prediction_h5py['labels']
    labels = data_file['labels']

    timer = time.time()
    outer_batch_size = batch_size * 16  # How much can fit in memory at a time
    num_batches = int(math.ceil(float(labels.shape[0]) / outer_batch_size))  # float division, round up

    skin_skin = 0
    skin_but_not_skin = 0
    not_skin_not_skin = 0
    not_skin_but_skin = 0
    t_pos_vote = {}
    t_neg_vote = {}
    f_pos_vote = {}
    f_neg_vote = {}
    # value => [t pos, t neg, f pos, f neg]
    counts = {}

    for i in range(num_batches):
        start = i * outer_batch_size
        end = (i + 1) * outer_batch_size

        predicted_y = prediction[start: end]
        # np.sign(predicted_y, out=predicted_y)

        current_index = 0
        
        label_batch = labels[start: end]
        for i in range(len(label_batch)):
            pred_value = int(predicted_y[current_index])
            if pred_value not in counts.keys():
                counts[pred_value] = [0, 0, 0, 0]

            pred_label = np.sign(pred_value)
            if label_batch[i] == 1 and pred_label == 1: # true positive
                skin_skin += 1
                t_pos_vote[pred_value] = t_pos_vote.get(pred_value, 0) + 1
                counts[pred_value][0] += 1
            elif label_batch[i] == -1 and pred_label == -1: # true negative
                not_skin_not_skin += 1
                t_neg_vote[pred_value] = t_neg_vote.get(pred_value, 0) + 1
                counts[pred_value][1] += 1
            elif label_batch[i] == -1 and pred_label == 1:  # False positive
                not_skin_but_skin += 1
                f_neg_vote[pred_value] = f_neg_vote.get(pred_value, 0) + 1
                counts[pred_value][2] += 1
            elif label_batch[i] == 1 and pred_label == -1: # False negative
                skin_but_not_skin += 1
                f_pos_vote[pred_value] = f_pos_vote.get(pred_value, 0) + 1
                counts[pred_value][3] += 1
            current_index += 1

    print("Finished prediction and analysis", timeit(timer))

    print(len(labels), "data points")
    print("True Positive:", skin_skin, t_pos_vote)
    print("True Negative:", not_skin_not_skin, t_neg_vote)
    print("False Negative:", skin_but_not_skin, f_neg_vote)
    print("False Positive:", not_skin_but_skin, f_pos_vote)
    print(counts)
    print("-" * 80)


if __name__ == '__main__':
    if "-h" in sys.argv:
        print("python ensemble_validation.py <output hdf5 file> <validation hdf5 file> <batch size>")
    else:
        print(sys.argv)
        evaluate_elm(prediction_file=sys.argv[1], validation_file=sys.argv[2], batch_size=int(sys.argv[3]))


