from issm import load_train_dataset, load_regulized_train_dataset
import numpy as np
import matplotlib.pyplot as plt

x_trian, y_train = load_regulized_train_dataset()

unique, count = np.unique(y_train, return_counts=True)

plt.bar(unique, count)
plt.xlabel("Labels")
plt.ylabel("Label Count")
plt.title("ISSM Dataset Labels")
plt.savefig("./issm_regulized_label.png")