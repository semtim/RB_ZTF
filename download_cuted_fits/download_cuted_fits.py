from astropy.io import fits
import requests
import os
import json
from copy import deepcopy
from io import BytesIO


####################################################
def download_fits_by_url(url, coords, oid, hmjd, path, shape=(28, 28)):
    url_cuted_fits = url + f'?center={coords[0]},{coords[1]}&size=27pix&gzip=false'
    response = requests.get(url_cuted_fits)
    response.raise_for_status()
    stream = BytesIO(response.content)
    stream.seek(0)
    hdus = fits.open(stream)
            
    ext_header = deepcopy(hdus[0].header)
    ext_header.append(('OID', oid, 'Object ID from SNAD'), end=True)
    ext_header.append(('OIDRA', coords[0], 'Object RA'), end=True)
    ext_header.append(('OIDDEC', coords[1], 'Object DEC'), end=True)
    ext_header.append(('HMJD', hmjd, 'HMJD'), end=True)
    ext_header.append(('URL', url_cuted_fits, 'download url'), end=True)
    hdu_newheader = fits.PrimaryHDU(deepcopy(hdus[0].data), ext_header)
    #np.save(download_path + str(hmjds[i]) , img_array)
    hdu_newheader.writeto(path + str(hmjd) + '.fits')
    hdus.close()



def download_all_fits_by_oid(oid, return_fails_count=False):
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

    download_path = 'data/' + str(oid) + '/'
    if not os.path.exists(download_path):
        os.mkdir(download_path)

    # cutout images
    downloadFail_count = 0
    for i, u in enumerate(urls):
        try:
            download_fits_by_url(u, coords, oid, hmjds[i], download_path)
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



oids, targets = get_oids('akb.ztf.snad.space.json')

#oid = 257206100020483
for oid in oids:
    download_and_cut_by_oid(oid)


####
# with open('data/'+str(oid)+'/'+str(hmjds[50])+'.npy', 'rb') as f:
#     a = np.load(f)

# plt.imshow(a,  cmap='gray')
