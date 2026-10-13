from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)
import matplotlib.pyplot as plt
plt.matshow(digits.images[73], cmap="gray")
plt.show()