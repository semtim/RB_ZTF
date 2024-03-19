from astropy.io import fits
from astropy.wcs import WCS, utils
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.nddata import Cutout2D
import requests
from astropy.visualization import ImageNormalize, HistEqStretch
import os
import json
from copy import deepcopy
from io import BytesIO


####################################################

def cut_image(hdus, coords, oid, url, hmjd, shape=(28, 28)):
    coord = SkyCoord(*coords, unit='deg', frame='icrs')
    image = deepcopy(hdus[0].data)
    currentWCS = WCS(hdus[0].header, hdus)
    cutout = Cutout2D(image, coord, shape, wcs=currentWCS)
    cutout_image = cutout.data #np array
    ext_header = deepcopy(hdus[0].header)
    ext_header.append(('OID', oid, 'Object ID from SNAD'), end=True)
    ext_header.append(('OIDRA', coords[0], 'Object RA'), end=True)
    ext_header.append(('OIDDEC', coords[1], 'Object DEC'), end=True)
    ext_header.append(('HMJD', hmjd, 'HMJD'), end=True)
    ext_header.append(('URL', url, 'url to orig frame'), end=True)
    cut_hdu = fits.PrimaryHDU(cutout_image, ext_header)
    
    return cut_hdu


def open_and_cut_fits(url, coords, oid, hmjd, path):
    response = requests.get(url)
    response.raise_for_status()
    stream = BytesIO(response.content)
    stream.seek(0)
    hdus = fits.open(stream)
            
    new_img = cut_image(hdus, coords,  oid, url, hmjd) #cutted image
    #np.save(download_path + str(hmjds[i]) , img_array)
    new_img.writeto(path + str(hmjd) + '.fits')
    hdus.close()



def download_and_cut_by_oid(oid, return_fails_count=False):
    #get urls for all mjd by oid only
    url = "https://finder.fits.ztf.snad.space/api/v1/urls/by/oid"
    params = {"oid": oid, "dr": 'dr8', "base_url":'IPAC'}
    with requests.get(url, params=params) as r:
        r.raise_for_status()
        data = r.json()

    # Extract scientific image URLs:
    urls = [obs["urls"]["sciimg"] for obs in data]

    # Extract ra dec for obj
    coords = (data[0]["inputs"]["ra"], data[0]["inputs"]["dec"])

    # Extract hmjd for obj
    hmjds = [obs["inputs"]["hmjd"] for obs in data]

    download_path = '../data/' + str(oid) + '/'
    if not os.path.exists(download_path):
        os.mkdir(download_path)

    # cutout images
    downloadFail_count = 0
    for i, u in enumerate(urls):
        try:
            #hdus = fits.open(u, cache=False) # download fits for hmjd[i]
            open_and_cut_fits(u, coords, oid, hmjds[i], download_path)
        except:
            downloadFail_count += 1
            #with open(str(i) + '.json', 'w') as f:
             #   json.dump(data[i], f)
            continue
        
        
    #print(f'FITS not found {downloadFail_count}')
    if return_fails_count:
        return downloadFail_count
####################################################

#get oids and tags from json
def get_oids(filepath):
    file = open(filepath)
    obj_list = json.load(file)
    file.close()

    oids = [data['oid'] for data in obj_list]
    tags = [data['tags'] for data in obj_list]
    targets = [] # 1-artefact,  0-transient
    for tag_list in tags:
        if 'artefact' in tag_list:
            targets.append(1)
        else:
            targets.append(0)
    
    return oids, targets
####################################################



oids, targets = get_oids('../akb.ztf.snad.space.json')

for oid in oids:
    download_and_cut_by_oid(oid)



