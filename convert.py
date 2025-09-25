"""
Update the FITS header to remove unnecessary fits axis information. Importantly
this script makes the modification in place, and allows the time axis to be
treated the same as the frequency axis would in a spectral cube.

The modificaiton to the FITS header is down by writing to the file 
directly. This is of huge benefit for large cubes.
""" 

from pathlib import Path 
import sys

from astropy.io import fits
import numpy as np

fits_in = Path(sys.argv[1])

print("Getting Header")

hdr = fits.getheader(fits_in)
original_string = hdr.tostring()

print(f"Length of header keys: {len(hdr)}")

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

print(f"Length of header keys: {len(hdr)}")

new_hdr_string = hdr.tostring()

difference = len(original_string) - len(new_hdr_string)
print(f"Original header length: {len(original_string)}")
print(f"Updated header length: {len(new_hdr_string)}")
print(f"Diff. to pad: {difference}")

new_hdr_string += " "*difference

if len(new_hdr_string) != len(original_string):
    msg = "Header string mis-match"
    raise ValueError(msg)

if new_hdr_string == original_string:
    msg = "There should be some difference"
    raise ValueError(msg)

print("\bModifying file in place")

assert "CDELT4" not in new_hdr_string, "WHAT"

print(f"Modying {fits_in}")
with open(fits_in, 'rb+') as out_file:
    # First we blank out the existing header
    print("... Blanking out existing header")
    out_file.seek(0)
    out_file.write(b" "*len(original_string))
    
    # Now we write out the new header
    print("... Writing out new header")
    out_file.seek(0)
    out_file.write(new_hdr_string.encode('ascii'))
    out_file.flush()
