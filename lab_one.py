import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def square_signal(x):
    if x % 2 < 1:
        return 2
    else:
        return -2
    
def fourier_series(x, N, T):
    res = 0
    w = 2 * np.pi / T
    
    # a0
    a_z = 2 / T * quad(square_signal, x, x + T)[0]
    res += a_z / 2
    
    # an + bn
    for n in range(N):
        def cos_help(x):
            return square_signal(x) * np.cos(n * w * x)
        def sin_help(x):
            return square_signal(x) * np.sin(n * w * x)
        
        a_n = 2 / T * quad(cos_help, x, x + T)[0]
        b_n = 2 / T * quad(sin_help, x, x + T)[0]
        
        res += a_n * np.cos(n * w * x) + b_n * np.sin(n * w * x)
    
    return res

args = np.linspace(-5.0, 5.0, num=500)
y_square_signal = [square_signal(i) for i in args]
y_approximate = [fourier_series(i, 12, 2) for i in args]
error = [y_square_signal[i] - y_approximate[i] for i in range(500)]

plt.figure()

plt.subplot(121)
plt.plot(args, y_square_signal)
plt.plot(args, y_approximate)

plt.subplot(122)
plt.plot(args, error)
plt.show()