import numpy as np

a = np.arange(6).reshape(3,2)
print(a)
np.savetxt("hi.txt", a, fmt='%1.1f')

