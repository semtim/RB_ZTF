from astropy.io import fits
from astropy.wcs import WCS, utils
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import numpy as np
from astropy.nddata import Cutout2D
import requests
from astropy.visualization import ImageNormalize, HistEqStretch
import os
import json
from copy import deepcopy
from io import BytesIO
import imageio

def frames(oid):
    frames_ar = []
    for root, dirs, files in os.walk(str(oid)):
        for filename in sorted(files):
            with fits.open(str(oid) + '/' + filename) as f:
                a = f[0].data
                frames_ar.append(a)

    return frames_ar

oid = 633206300034014
fr = frames(oid)
imageio.mimsave(str(oid) +'.gif', # output gif
                fr,          # array of input frames
                fps = 5)

