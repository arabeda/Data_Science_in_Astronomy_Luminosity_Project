import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9 as cosmo


def get_phi_from_txt():
    df = pd.read_csv('DataAngleIternalPlateau.txt', sep='\t')
    phi = df['phinprompt']
    phi_err = df['phinprompterror']
    return phi, phi_err

def get_data_from_txt():
    df = pd.read_csv('repo_data.txt', sep='\t')
    time = df['time']
    timeerr1 = df['timeerr1']
    timeerr2 = df['timeerr2']
    flux = df['flux']
    fluxerr1 = df['fluxerr1']
    fluxerr2 = df['fluxerr2']
    return flux, df, time

def get_L():
    phi = -0.786642
    z = 2.26
    theta = 9.32499174100697
    (flux, df, time) = get_data_from_txt()
    k = (-0.09 * 0.3 ** (phi) + 10 ** 8 * 10000 ** (phi) * ((1 / ((1 + z))) ** (2 + (phi))))
    Dl = cosmo.luminosity_distance(z)
    flux = np.log(df['flux']*(np.pi*Dl**2*4*k*(1-np.cos(theta))))
    time = df['time'].apply(lambda x: float(x))
    time = np.log(time)
    df.plot(x='time', y='flux', kind='scatter')
    plt.show()
    return df['flux']

if __name__ == '__main__':
    get_L()


