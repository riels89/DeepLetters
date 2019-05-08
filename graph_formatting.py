import numpy as np
from matplotlib import pyplot as plt


first = np.load("CNN/trained_networks/OG CNN/one opt/static_v2_lr-1e-06/epoch-7/static_v2_lr-1e-06-val_losses.npy").tolist()
second = np.load("CNN/trained_networks/OG CNN/one opt/static_v2_lr-0.0001/epoch-4/static_v2_lr-0.0001-val_losses.npy").tolist()
four = [[6.5162574589313085]] + second + first
print(four)


# l2_f = np.load("CNN/trained_networks/OG CNN/one opt, l2/static_v2_lr-0.0001/epoch-5/static_v2_lr-0.0001-val_losses.npy").tolist()
# l2_s = np.load("CNN/trained_networks/OG CNN/one opt, l2/static_v2_lr-1e-06/epoch-8/static_v2_lr-1e-06-val_losses.npy").tolist()
# l2 = [[40.12314]] + l2_f + l2_s
# print(four)
five = np.load("CNN/trained_networks/OG CNN/one opt/static_v2_lr-1e-05/epoch-7/static_v2_lr-1e-05-val_losses.npy").tolist()
five = [[6.5162574589313085]] + five
six = np.load("CNN/trained_networks/OG CNN/one opt/static_v2_lr-1e-06 full train/epoch-13/static_v2_lr-1e-06-val_losses.npy").tolist()
six = [[6.5162574589313085]] + six
#
# l2 = [[23.327596749925917]] + first + second

plt.plot(four, label='1e-4 -> 1e-6')
#plt.plot(l2, label='1e-4 -> 1e-6 + L2')
plt.plot(five, label='1e-5')
plt.plot(six, label='1e-6')
plt.grid(True)
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("Train.png")
