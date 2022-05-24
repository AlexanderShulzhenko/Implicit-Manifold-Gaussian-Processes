import numpy as np
import math
import scipy
from scipy.special import sph_harm, lpmv, factorial
from scipy.special import gamma
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

dtype = tf.float64


def plot_harmonic_3D(l,m):
    PHI, THETA = np.mgrid[0:2*np.pi:300j, 0:np.pi:150j]
    R = sph_harm(m, l, PHI, THETA).real

    s = 1
    X = (s*R+1) * np.sin(THETA) * np.cos(PHI)
    Y = (s*R+1) * np.sin(THETA) * np.sin(PHI)
    Z = (s*R+1) * np.cos(THETA)

    norm = colors.Normalize()
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(10,10))
    m = cm.ScalarMappable(cmap=cm.jet)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cm.jet(norm(R)))
    m.set_array(R)
    ax.view_init(elev=0, azim=135)
    ax.set_axis_off()

def plot_harmonic_2D(l,m):
    phi = np.linspace(0, np.pi, 100)
    theta = np.linspace(0, 2*np.pi, 100)
    phi, theta = np.meshgrid(phi, theta)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    fcolors = sph_harm(m, l, theta, phi).real
    fmax, fmin = fcolors.max(), fcolors.min()
    fcolors = (fcolors - fmin)/(fmax - fmin)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.jet(fcolors))
    ax.set_axis_off()
    plt.show()
