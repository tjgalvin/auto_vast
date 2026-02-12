"""Dumb little toy
"""

from pathlib import Path

from astropy.io import fits
import numpy as np
from scipy.ndimage import minimum_filter
from dataclasses import dataclass
from numpy.typing import NDArray
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from scipy.ndimage import label


@dataclass
class Stats:
    mask: NDArray[np.bool]
    """Boolean mask of pixels above cut"""
    norm_mask: NDArray[np.floating]
    """Value normalised by number of pixels in mask"""
    sum_mask: NDArray[np.floating]
    """Inforporate flux values into returned masked"""
    no_pixs: int
    """Number of active pixels above the mask"""
    idx: int
    """The slice of the cube"""


def mac_filter(data: np.ndarray, box: tuple[int,int], clip: float) -> np.ndarray:
    return data > np.abs(
        minimum_filter(data, box) * clip
    )


# def runner(idx: int, data_cube: NDArray[np.floating]) -> Stats:
def runner(idx: int, read_path: Path) -> Stats:


    f = fits.open(read_path, mode="readonly", memmap=True)
    image = f[0].data[idx]
    
    if np.all(np.isnan(image)):
        return None

    mask = mac_filter(data=image, box=(400,400), clip=2)
    mask = np.nan_to_num(mask, copy=False)
    
    no_pixs = np.sum(mask)
    norm_mask = mask / no_pixs
    norm_mask = np.nan_to_num(norm_mask, copy=False)
    
    flux_mask = np.zeros_like(image)
    flux_mask[mask] = image[mask]
    
    return Stats(
        mask=mask, norm_mask=norm_mask, sum_mask=flux_mask, no_pixs=no_pixs, idx=idx
    )


def write_fits(out_path: Path, data: NDArray, header: fits.HeaderDiff) -> Path:
    print(f"Writing out {out_path=}")

    fits.writeto(
        out_path, data=data.astype(float), header=header, overwrite=True
    )

def island_cut(data: NDArray[np.floating], minimum_size: int = 4) -> NDArray[np.floating]:
    
    
    data_label, no_labels = label(input=data, structure=np.ones((3,3)))
    print(f"Found {no_labels=} with {minimum_size=}")
    
    for i in range(no_labels):
        label_mask = data_label == i
        total_size = np.sum(label_mask)
        print(f"island {i:<4} {total_size=}")
        if total_size < minimum_size:
            data[label_mask] = 0
            
            
    return data
    

def main() -> None:
    path = Path('SB64328.J1832-0911.round3.i.contsub.linmos.cube.fits')

    
    with fits.open(path, memmap=True) as f:
        header = f[0].header
        d = f[0].data.shape
        
    print(f"{d=}")

    # compute_func = partial(runner, data_cube=d)
    compute_func = partial(runner, read_path=path)

    mask = np.zeros(d[1:3], dtype=bool)
    norm_mask = np.zeros(d[1:3])
    sum_mask = np.zeros(d[1:3])

    pool = ProcessPoolExecutor(max_workers=12)
    for idx, results in enumerate(pool.map(compute_func, range(d[0]))):
        if results is None:
            continue
    
        print(f"{results.idx:<4} {results.no_pixs}")
    
        mask |= results.mask
        norm_mask += results.norm_mask
        sum_mask += results.sum_mask


    write_fits(out_path=path.with_suffix(".event_mask.fits"), data=mask, header=header)
    write_fits(out_path=path.with_suffix(".norm_mask.fits"), data=norm_mask, header=header)
    write_fits(out_path=path.with_suffix(".sum_mask.fits"), data=sum_mask, header=header)


    island_clip = island_cut(data=norm_mask, minimum_size=8)
    write_fits(out_path=path.with_suffix(".filter_norm_mask.fits"), data=island_clip, header=header)
    


if __name__ == "__main__":
    main()
