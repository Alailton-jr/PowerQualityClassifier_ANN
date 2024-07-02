
import numpy as np
import random

def generate_data(n_samples = 1000, duration = 1/60, fs = 15360, t_ini = 0):
    '''
    Generate synthetic data for Power Quality classification

    Parameters:
    n_samples: int, number of samples for each class

    Returns:
    data: np.array, shape (n_samples, n_features)
    labels: np.array, shape (n_samples, 1)
    '''

    t = np.arange(0, duration, 1/fs)

    n_class = 8
    n_samples = int(n_samples/n_class)

    phaseAngle = np.pi/n_samples*10
    phaseAngle = np.linspace(0, np.pi, (n_samples//9))
    phaseAngle = phaseAngle.repeat(10)
    
    data = np.zeros((n_class*n_samples, t.shape[0]))
    labels = np.zeros((n_class*n_samples, 1), dtype=object)

    # Normal
    u = np.heaviside(t - t_ini, 1)
    mags = np.linspace(0.93, 1.07, n_samples)
    v = np.array([np.sin(2*np.pi*60*t + phaseAngle[i]) * (1 - u) + mag * np.sin(2*np.pi*60*t+ phaseAngle[i]) * u for i, mag in enumerate(mags)])
    data[0:n_samples, :] = v 
    labels[0:n_samples] = 'Normal'
    
    # Sag
    u = np.heaviside(t - t_ini, 1)
    mags = np.linspace(0.1, 0.9, n_samples)
    v = np.array([np.sin(2*np.pi*60*t+ phaseAngle[i]) * (1 - u) + mag * np.sin(2*np.pi*60*t+ phaseAngle[i]) * u for i, mag in enumerate(mags)])
    data[n_samples:2*n_samples, :] = v
    labels[n_samples:2*n_samples] = 'Sag'

    # Swell
    u = np.heaviside(t - t_ini, 1)
    mags = np.linspace(1.1, 1.8, n_samples)
    v = np.array([np.sin(2*np.pi*60*t+ phaseAngle[i]) * (1 - u) + mag * np.sin(2*np.pi*60*t+ phaseAngle[i]) * u for i, mag in enumerate(mags)])
    data[2*n_samples:3*n_samples, :] = v
    labels[2*n_samples:3*n_samples] = 'Swell'
    
    # Interruption
    u = np.heaviside(t - t_ini, 1)
    mags = np.linspace(0, 0.09, n_samples)
    v = np.array([np.sin(2*np.pi*60*t+ phaseAngle[i]) * (1 - u) + mag * np.sin(2*np.pi*60*t+ phaseAngle[i])  * u for i, mag in enumerate(mags)])
    data[3*n_samples:4*n_samples, :] = v
    labels[3*n_samples:4*n_samples] = 'Interruption'

    # Transient
    u = np.heaviside(t - (duration/4), 1)
    mags = np.linspace(1.2, 2, n_samples)
    p = 400
    fosc = 800
    v = np.array([np.sin(2*np.pi*60*t+ phaseAngle[i])  + u * mag * np.exp(-p * (t-(duration/4))) * np.sin(2*np.pi*fosc*(t-(duration/4))) for i, mag in enumerate(mags)])
    data[4*n_samples:5*n_samples, :] = v
    labels[4*n_samples:5*n_samples] = 'Transient'

    # Notch
    tau = duration/55
    phi = np.array([1/12, 3/12, 5/12, 7/12, 9/12, 11/12])/60
    p = 8000
    fosc = 6000
    def sgn(x):
        return np.where(x > 0, 1, np.where(x == 0, 0, -1))
    vn_vosc = np.sum([(np.heaviside(t - _phi, 1) - np.heaviside(t - _phi - tau, 1)) * \
            0.4*(np.sin(2*np.pi*fosc*(t - _phi)) * np.exp(-p*(t - _phi - tau))  ) for _phi in phi], axis=0)
    pm = sgn(np.sin(2*np.pi*60*t + np.pi/2))
    nds = np.linspace(0.2, 0.5, n_samples)
    v = np.array([np.sin(2*np.pi*60*t+ phaseAngle[i])  + (nd * vn_vosc * pm) for i, nd in enumerate(nds)])
    data[5*n_samples:6*n_samples, :] = v
    labels[5*n_samples:6*n_samples] = 'Notch'

    # Harmonics Distortion
    u = np.heaviside(t - t_ini, 1)
    desired_thd = np.linspace(0.1, 0.5, n_samples)
    orders = [3, 5, 7, 9, 11]
    v = np.zeros((n_samples, t.shape[0]))
    for i, thd in enumerate(desired_thd):
        order_list = random.sample(orders, random.randint(1, len(orders)))
        v[i] = np.sin(2*np.pi*60*t+ phaseAngle[i])  + np.sum([(thd/np.sqrt(len(order_list))) * np.sin(2*np.pi*60*ord*t) for i, ord in enumerate(order_list)], axis=0)
    data[6*n_samples:7*n_samples, :] = v
    labels[6*n_samples:7*n_samples] = 'Harmonics Distortion'

    #Flicker
    mags = np.linspace(0.15, 0.5, n_samples)
    ff = np.linspace(8, 30, n_samples//9)
    ff = ff.repeat(10)
    v = np.array([(1 + fm * np.sin(2*np.pi*ff[i]*t + phaseAngle[i]*2)) * np.sin(2*np.pi*60*t + phaseAngle[i]) for i, fm in enumerate(mags)])
    data[7*n_samples:8*n_samples, :] = v
    labels[7*n_samples:8*n_samples] = 'Flicker'

    return data, labels