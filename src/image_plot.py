

import os, numpy as np, matplotlib.pyplot as plt, tqdm
from . import compute_stft, compute_cwt, compute_st, compute_hht
from . import generate_data


def plot_espectrum(folder_dir, plot_normal=True, plot_stft=True, plot_cwt=True, plot_st=True, plot_hht=True):

    n_samples = 1000
    fs = 15360
    dur = 1/60

    data, labels = generate_data(
        n_samples=n_samples,
        duration=dur,
        fs=fs,
        t_ini=0
    )

    t = np.arange(0,dur,1/fs)*1000
    n_Sel = int(n_samples/8*0.75)

    if plot_normal:
        ### Normal Plot
        plt.figure(figsize=(16,14))
        for i in range(8):
            plt.subplot(4,4,i+1)
            plt.plot(t, data[i*(n_samples//8)+n_Sel])
            plt.title(f'{labels[i*(n_samples//8)+n_Sel,0]}')
            plt.xlim([0, 1000/60])
            plt.ylim([-2, 2])
            plt.grid()
        plt.savefig(os.path.join(folder_dir, 'Normal.png'), format='png', dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=True, facecolor='w', edgecolor='w', orientation='portrait')

    if plot_stft:
        ### STFT Plot
        stft_data = compute_stft(data)
        plt.figure(figsize=(16,14))
        for i in range(8):
            plt.subplot(4,4,i+1)
            plt.imshow(stft_data[i*(n_samples//8)+n_Sel], cmap='turbo', aspect='auto', origin='lower', interpolation='spline16', extent=[0, 256, 0, 480])
            plt.xticks(np.arange(0, 256, 63), np.round(t[np.arange(0, 256, 63)], 2))
            plt.yticks([])
            plt.grid(alpha=0)
            plt.colorbar()
            plt.title(f'{labels[i*(n_samples//8)+n_Sel,0]}')
        plt.savefig(os.path.join(folder_dir, 'STFT.png'), format='png', dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=True, facecolor='w', edgecolor='w', orientation='portrait')
        del stft_data

    if plot_cwt:
        ### CWT Plot
        cwt_data = compute_cwt(data)
        plt.figure(figsize=(16,14))
        for i in range(8):
            plt.subplot(4,4,i+1)
            plt.pcolormesh(cwt_data[i*(n_samples//8)+n_Sel])
            plt.xticks(np.arange(0, 128, 63), np.round(t[np.arange(0, 128, 63)], 2)*2)
            plt.grid(alpha=0)
            plt.colorbar()
            plt.title(f'{labels[i*(n_samples//8)+n_Sel,0]}')
        plt.savefig(os.path.join(folder_dir, 'CWT.png'), format='png', dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=True, facecolor='w', edgecolor='w', orientation='portrait')
        del cwt_data

    if plot_st:
        ### ST Plot
        st_data = compute_st(data)
        plt.figure(figsize=(16,14))
        for i in range(8):
            plt.subplot(4,4,i+1)
            plt.imshow(st_data[i*(n_samples//8)+n_Sel], cmap='turbo', aspect='auto', origin='lower', interpolation='spline16', extent=[0, 256, 0, 1080])
            plt.xticks(np.arange(0, 256, 63), np.round(t[np.arange(0, 256, 63)], 2))
            plt.yticks([])
            plt.grid(alpha=0)
            plt.colorbar()
            plt.title(f'{labels[i*(n_samples//8)+n_Sel,0]}')
        plt.savefig(os.path.join(folder_dir, 'ST.png'), format='png', dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=True, facecolor='w', edgecolor='w', orientation='portrait')
        del st_data

    if plot_hht:
        ### HHT Plot
        hht_data = compute_hht(data)
        plt.figure(figsize=(16,14))
        for i in range(8):
            plt.subplot(4,4,i+1)
            plt.imshow(hht_data[i*(n_samples//8)+n_Sel], cmap='turbo', aspect='auto', origin='lower', interpolation='spline16', extent=[0, 256, 0, 480])
            plt.xticks(np.arange(0, 256, 63), np.round(t[np.arange(0, 256, 63)], 2))
            plt.yticks([])
            plt.grid(alpha=0)
            plt.colorbar()
            plt.title(f'{labels[i*(n_samples//8)+n_Sel,0]}')
        plt.savefig(os.path.join(folder_dir, 'HHT.png'), format='png', dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=True, facecolor='w', edgecolor='w', orientation='portrait')
        del hht_data