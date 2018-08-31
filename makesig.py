
from __future__ import division
import numpy as np
import bisect

#######################################################################################################################

# https://pythonhosted.org/pyrwt/_modules/rwt/utilities.html

MAKESIG_SIGNALS = [ 'AllSig',
          'Blocks',
          'Bumps',
          'HeaviSine',
          'Doppler',
          'QuadChirp',
          'MishMash',
          'Ramp',
          'Cusp',
          'Sing',
          'HiSine',
          'LoSine',
          'LinChirp',
          'TwoChirp',
          'WernerSorrows',
          'Leopold'
          ]

def makesig(signal_name='AllSig', N=512, t=None, y=None):
    """
    Creates artificial test signal identical to the
    standard test signals proposed and used by D. Donoho and I. Johnstone
    in WaveLab (- a matlab toolbox developed by Donoho et al. the statistics
    department at Stanford University).

    Input:  signal_name - Name of the desired signal (Default 'all')
                       'AllSig' (Returns a matrix with all the signals)
                       'HeaviSine'
                       'Bumps'
                       'Blocks'
                       'Doppler'
                       'Ramp'
                       'Cusp'
                       'Sing'
                       'HiSine'
                       'LoSine'
                       'LinChirp'
                       'TwoChirp'
                       'QuadChirp'
                       'MishMash'
                       'WernerSorrows' (Heisenberg)
                       'Leopold' (Kronecker)
              N       - Length in samples of the desired signal (Default 512)

    Output: x   - vector/matrix of test signals
            N   - length of signal returned

    References:
           WaveLab can be accessed at
           www_url: http://playfair.stanford.edu/~wavelab/
           Also see various articles by D.L. Donoho et al. at
           web_url: http://playfair.stanford.edu/

    Author: Jan Erik Odegard  <odegard@ece.rice.edu>
    This m-file is a copy of the  code provided with WaveLab
    customized to be consistent with RWT.
    """

    if t is None:
        t = np.linspace(1, N, N)/N
    else:
        N = len(t)

    if y is None:
        y = np.zeros_like(t)
    else:
        y = np.copy(y)

    signals = []

    if signal_name in ('HeaviSine', 'AllSig'):
        y += 4 * np.sin(4*np.pi*t) - np.sign(t - 0.3) - np.sign(0.72 - t)

        signals.append(y)

    if signal_name in ('Bumps', 'AllSig'):
        pos = np.array([ .1, .13, .15, .23, .25, .40, .44, .65, .76, .78, .81])
        hgt = np.array([ 4,  5,   3,   4,  5,  4.2, 2.1, 4.3,  3.1, 5.1, 4.2])
        wth = np.array([.005, .005, .006, .01, .01, .03, .01, .01,  .005, .008, .005])
        for p, h, w in zip(pos, hgt, wth):
            y += h / (1 + np.abs((t - p) / w)) **4
    
        signals.append(y)

    if signal_name in ('Blocks', 'AllSig'):
        pos = np.array([ .1, .13, .15, .23, .25, .40, .44, .65, .76, .78, .81])
        hgt = np.array([ 4,  -5,   3,   -4,  5,  -4.2, 2.1, 4.3,  -3.1, 2.1, -4.2])
        for p, h in zip(pos, hgt):
            y += (1 + np.sign(t - p))*h/2

        signals.append(y)

    if signal_name in ('Doppler', 'AllSig'):
        y += np.sqrt(t * (1-t)) * np.sin((2*np.pi*1.05) / (t+.05))

        signals.append(y)

    if signal_name in ('Ramp', 'AllSig'):
        y += t.copy()
        y[t >= .37] -= 1

        signals.append(y)

    if signal_name in ('Cusp', 'AllSig'):
        y += np.sqrt(np.abs(t - 0.37))

        signals.append(y)

    if signal_name in ('Sing', 'AllSig'):
        k = np.floor(N * .37)
        y += 1 / np.abs(t - (k+.5)/N)

        signals.append(y)

    if signal_name in ('HiSine', 'AllSig'):
        y += np.sin(N*0.6902*np.pi*t)

        signals.append(y)

    if signal_name in ('LoSine', 'AllSig'):
        y += np.sin(N*0.3333*np.pi*t)

        signals.append(y)

    if signal_name in ('LinChirp', 'AllSig'):
        y += np.sin(N*0.125*np.pi*t*t)

        signals.append(y)

    if signal_name in ('TwoChirp', 'AllSig'):
        y += np.sin(N*np.pi*t*t) + np.sin(N*np.pi/3*t*t)

        signals.append(y)

    if signal_name in ('QuadChirp', 'AllSig'):
        y += np.sin(N*np.pi/3*t*t*t)

        signals.append(y)

    if signal_name in ('MishMash', 'AllSig'):
        #
        # QuadChirp + LinChirp + HiSine
        #
        y += np.sin(N*np.pi/3*t*t*t) + np.sin(N*0.125*np.pi*t*t) + np.sin(N*0.6902*np.pi*t)

        signals.append(y)

    if signal_name in ('WernerSorrows', 'AllSig'):
        y += np.sin(N/2*np.pi*t*t*t)
        y += np.sin(N*0.6902*np.pi*t)
        y += np.sin(N*np.pi*t*t)

        pos = np.array([.1, .13, .15, .23, .25, .40, .44, .65, .76, .78, .81])
        hgt = np.array([4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2])
        wth = np.array([.005, .005, .006, .01, .01, .03, .01, .01, .005, .008, .005])

        for p, h, w in zip(pos, hgt, wth):
            y += h/(1 + np.abs((t - p)/w))**4

        signals.append(y)

    if signal_name in ('Leopold', 'AllSig'):
        #y += (t == np.floor(.37 * N)/N).astype(np.float)
        y[bisect.bisect(t, 0.37)] += 1
        
        signals.append(y)

    if len(signals) == 1:
        return signals[0]

    return signals

#######################################################################################################################

def add_noise(signal, SNR_dB=70):
    """
    add noise to a signal
    Signal: a numpy array
    SNR_DB: SNR level signal + noise
    Output: signal with superimposed noise
    """
    if not SNR_dB:
        return signal
    
    signal_power = np.float32(np.sum(np.abs(signal) ** 2, axis=0)) / signal.shape[0]

    noise = np.float32(np.random.normal(0.0, 1.0, signal.shape))

    noise_power = np.abs(noise) ** 2

    K = (signal_power / noise_power) * 10**(-SNR_dB/10)

    new_noise = np.sqrt(K) * noise

    noisy_signal = np.float32(signal + new_noise)

    return noisy_signal

#######################################################################################################################

