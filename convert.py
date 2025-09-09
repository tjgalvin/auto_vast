# Very rough script to convert a time step fits cube from the 
# subtract cube pipeline workflow so that is can be opened in 
# carta and the like. The smarter thing to do would be to update
# the header as below, but write out the header with the 'as_string()'
# method directly to the start of the fits file in place. 
# This script will be added to the auto_vast repo for history and a 
# improved version will be added
from pathlib import Path 
import sys

from astropy.io import fits
import numpy as np

fits_in = Path(sys.argv[1])
fits_out = fits_in.with_suffix(".clean.fits")

print("Getting Header")

hdr = fits.getheader(fits_in)

print("Popping keys")
to_pop = []
for k in hdr:
   if ('PV' in k or 'PC' in k) and '_' in k:
        to_pop.append(k)
        print(f"\t{k}")
        
for k in to_pop:
    hdr.pop(k)

update_keys = ["CTYPE", "CRVAL", "CDELT", "CRPIX", "CUNIT"]

for uk in update_keys:
    hdr[f"{uk}3"] = hdr[f"{uk}5"]
    hdr.pop(f"{uk}4")
    hdr.pop(f"{uk}5")
    
hdr["NAXIS"] = 3
hdr["NAXIS3"] = hdr["NAXIS5"]

hdr.pop("NAXIS4")
hdr.pop("NAXIS5")

new_data = np.squeeze(fits.getdata(fits_in))

print(f"Writing {fits_out=}")
fits.writeto(fits_out, header=hdr, data=new_data, overwrite=True)

