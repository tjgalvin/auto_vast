"""Dumb little toy
"""

from pathlib import Path
from argparse import ArgumentParser

from astropy.io import fits
import numpy as np
from scipy.ndimage import minimum_filter
from dataclasses import dataclass
from numpy.typing import NDArray
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from scipy.ndimage import label, find_objects
import matplotlib.pyplot as plt


@dataclass
class StatsOptions:
    cube_path: Path
    """Path to the cube to inspect/search through"""
    box_size: int
    """SIze of the box of the minimum filter to use. Height and width are equal"""
    clip: float
    """Clipping threshold to use to identify signals"""
    plot: bool = False
    """Create basic output plots of potential signals"""

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
def stats_for_plane(
    plane_idx: int, 
    read_path: Path,
    box_size: int = 400,
    clip: float = 1.8,
) -> Stats:


    f = fits.open(read_path, mode="readonly", memmap=True)
    image = f[0].data[plane_idx]
    
    if np.all(np.isnan(image)):
        return None

    mask = mac_filter(
        data=image, 
        box=(box_size,box_size), 
        clip=clip
    )
    mask = np.nan_to_num(mask, copy=False)
    
    no_pixs = np.sum(mask)
    norm_mask = mask / no_pixs
    norm_mask = np.nan_to_num(norm_mask, copy=False)
    
    flux_mask = np.zeros_like(image)
    flux_mask[mask] = image[mask]
    
    return Stats(
        mask=mask, 
        norm_mask=norm_mask, 
        sum_mask=flux_mask, 
        no_pixs=no_pixs, 
        idx=plane_idx
    )


def write_fits(out_path: Path, data: NDArray, header: fits.HeaderDiff) -> Path:
    print(f"Writing out {out_path=}")

    fits.writeto(
        out_path, data=data.astype(float), header=header, overwrite=True
    )

def island_cut(data: NDArray[np.floating], minimum_size: int = 4) -> NDArray[np.floating]:
    
    
    data_label, no_labels = label(input=data, structure=np.ones((3,3)))
    print(f"Found {no_labels=} with {minimum_size=}")
    survived: int = 0
    for i in range(no_labels):
        label_mask = data_label == i
        total_size = np.sum(label_mask)
        print(f"island {i:<4} {total_size=}")
        if total_size < minimum_size:
            data[label_mask] = 0
            continue
        
        survived += 1
        
    print(f"{survived} of {no_labels} survived the cutting")    
    return data
    
def make_island_plot(
    cublet_data: NDArray[np.floating],
    tight_island_mask: NDArray[np.floating],
    raw_data:  NDArray[np.floating],
    output_path: Path,
    curve_title: None | str = None
) -> Path:

    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.imshow(
        raw_data,
        origin="lower"
    )
    ax1.set(
        title="Island mask",
    )
    
    light_curve = np.sum(cublet_data[:, tight_island_mask], axis=1)
    ax2.plot(light_curve)
    ax2.grid()
    ax2.set(
        xlabel="Timestep",
        ylabel="Sum (Jy)",
    )
    if curve_title is not None:
        ax2.set(title=curve_title)

    
    fig.tight_layout()
    fig.savefig(output_path)

    return output_path
    
def make_candidate_plots(
    cube_path: Path,
    base_image: NDArray[np.floating],    
) -> list[Path]:
    
    # Get the islands to process
    island_image, no_islands = label(
        base_image != 0, structure=np.ones((3,3))
    )
    print(f"Making plots for {no_islands}")
    output_paths = []
    for count, box_slices in enumerate(find_objects(island_image)):
        print(f"Plotting {count=}")
        output_path = cube_path.with_suffix(f".island_{count}.fits")

        with fits.open(cube_path) as f:
            cublet = f[0].data[:, *box_slices]

        tight_island_mask = island_image[*box_slices] == (count + 1)
        raw_data = base_image[*box_slices]
        
        output_paths.append(
            make_island_plot(
                cublet_data=cublet,
                tight_island_mask=tight_island_mask,
                raw_data=raw_data,
                output_path=output_path
            )
        )
        
    return output_paths
    

def perform_search(stats_options: StatsOptions) -> None:
    
    print(f"Loading {stats_options.cube_path}")
    with fits.open(stats_options.cube_path, memmap=True) as f:
        header = f[0].header
        data_shape = f[0].data.shape
        
    print(f"Loaded cube has {data_shape=}")
    assert len(data_shape) == 3, "Expected data of rank 3"

    # compute_func = partial(runner, data_cube=d)
    compute_func = partial(
        stats_for_plane, 
        read_path=stats_options.cube_path,
        box_size=stats_options.box_size,
        clip=stats_options.clip
    )

    no_of_timesteps = data_shape[0]
    image_shape = data_shape[1:]

    # Create the output arrays that will be updated throughout
    # the stats gather loop
    mask = np.zeros(image_shape, dtype=bool)
    norm_mask = np.zeros(image_shape)
    sum_mask = np.zeros(image_shape)

    # Create the process pool and loop over all time steps. The order
    # returned is not really important
    pool = ProcessPoolExecutor(max_workers=12)
    for results in pool.map(compute_func, range(no_of_timesteps)):
        if results is None:
            continue
    
        print(f"{results.idx:<4} {results.no_pixs}")
    
        mask += results.mask
        norm_mask += results.norm_mask
        sum_mask += results.sum_mask

    # Now write out
    write_fits(out_path=stats_options.cube_path.with_suffix(".event_mask.fits"), data=mask, header=header)
    write_fits(out_path=stats_options.cube_path.with_suffix(".norm_mask.fits"), data=norm_mask, header=header)
    write_fits(out_path=stats_options.cube_path.with_suffix(".sum_mask.fits"), data=sum_mask, header=header)

    # DO some clipping based on island morphology and write out
    island_clip = island_cut(data=norm_mask, minimum_size=10)
    write_fits(
        out_path=stats_options.cube_path.with_suffix(".filter_norm_mask.fits"), data=island_clip, header=header
    )
    
    island_clip = island_cut(data=sum_mask, minimum_size=10)
    write_fits(out_path=stats_options.cube_path.with_suffix(".filter_sum_mask.fits"), data=island_clip, header=header)
    
    if stats_options.plot:
        make_candidate_plots()
    

def get_parser() -> ArgumentParser:
    
    parser = ArgumentParser(description="Basic searcher for a time cube")
    
    parser.add_argument("cube_path", type=Path, help="Path to the cube to inspect/search through")
    parser.add_argument(
        "-b", "--box-size", default=400, type=int, help="SIze of the box of the minimum filter to use. Height and width are equal"
    )
    parser.add_argument("-c", "--clip", type=float, default=1.8, help="Significance test statistic in the minimum absolute clip")
    parser.add_argument("-p", "--plot", action="store_true", help="Some basic plots of potential events")
    
    return parser

    
def cli() -> None:
    
    parser = get_parser()

    args = parser.parse_args()

    stats_options = StatsOptions(
        cube_path=args.cube_path,
        box_size=args.box_size,
        clip=args.clip,
        plot=args.plot
    )

    perform_search(stats_options=stats_options)

if __name__ == "__main__":
    cli()
