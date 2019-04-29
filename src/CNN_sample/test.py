import os
import numpy as np
from hparams import hparams

# Function to get genre index for the given file
def get_label(file_name, hparams):
	genre = file_name.split('.')[0]
	label = hparams.genres.index(genre)
	return label

def load_dataset(set_name, hparams):
    x = []
    y = []
    dataset_path = os.path.join(hparams.feature_path, set_name)
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            data = np.load(os.path.join(root, file))
            if len(data) == 661504:
                print(file)
                break
            print(file, len(data))
            label = get_label(file, hparams)
            x.append(data)
            y.append(label)
    x = np.stack(x)
    y = np.stack(y)
    return x, y


x_train, y_train = load_dataset('train', hparams)