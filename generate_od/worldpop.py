import requests

from tqdm import tqdm
from io import BytesIO
from multiprocessing import Pool

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile

from .utils import calculate_utm_epsg


def worldpop_from_local_tif(area_shp: gpd.GeoDataFrame,
                            tif_path: str,
                            area_crs_epsg: int = 32652) -> np.ndarray:  # UTM 52N instead of 3857, more accurate?
                            # area_crs_epsg: int = 3857) -> np.ndarray:
    """
    Compute per-zone population and area using a local WorldPop raster.

    Parameters
    ----------
    area_shp : GeoDataFrame
        Polygons defining your zones (e.g. 431 grid cells for Fukuoka-shi).
        Can be in any CRS; will be reprojected as needed.
    tif_path : str
        Path to WorldPop GeoTIFF, e.g. "./assets/jpn_pop_2025_CN_100m_R2025A_v1.tif".
        This raster is in Geographic WGS84 (EPSG:4326), values = people per pixel.
    area_crs_epsg : int, optional
        EPSG code used to compute area in m² (default 3857, Web Mercator).
        For more accurate area you could use a local UTM zone.

    Returns
    -------
    feat : np.ndarray of shape (N, 2)
        Column 0: population per zone (sum of pixel values)
        Column 1: area per zone in km²
    """
    if area_shp.crs is None:
        raise ValueError("area_shp must have a CRS defined.")

    # 1. Open the WorldPop raster
    with rasterio.open(tif_path) as src:
        raster_crs = src.crs

        # 2. Reproject polygons to raster CRS for masking
        if area_shp.crs != raster_crs:
            area_for_mask = area_shp.to_crs(raster_crs)
        else:
            area_for_mask = area_shp

        # 3. Separate geometry for area calculation in a projected CRS
        area_for_area = area_shp.to_crs(epsg=area_crs_epsg)

        populations = []
        areas_km2 = []

        it = zip(area_for_mask.geometry, area_for_area.geometry)
        for geom_mask, geom_area in tqdm(
            it,
            total=len(area_shp),
            desc=" -- Population of regions (local TIFF)"
        ):
            # 3a. Sum population from raster inside polygon
            try:
                out_img, _ = mask(src, [geom_mask], crop=True)
                # WorldPop pixels already represent "people per pixel"
                pop = float(out_img[out_img > 0].sum())
            except Exception:
                pop = 0.0

            # 3b. Compute area in km²
            area_km2 = float(geom_area.area) / 1e6

            populations.append(pop)
            areas_km2.append(area_km2)

    feat = np.column_stack([populations, areas_km2])
    return feat


def population_one_region(args):
    '''
    get the population of given region, 
    by aggregating the all pixels located 
    in that region from worldpop tiff.
    '''
    # arguments
    region = args

    # get the worldpop tiff for the region
    base_url = 'https://worldpop.arcgis.com/arcgis/rest/services/WorldPop_Total_Population_100m/ImageServer/exportImage?f=image&format=tiff&noData=0&'
    # Use native service CRS (EPSG:4326) at its default ~100m resolution.
    # This avoids automatic downsampling triggered by the 4100-pixel size limit
    # when requesting very tall regions in Web Mercator.
    left, bottom, right, top = region.geometry.bounds

    # Explicitly request the 2020 layer using the StdTime value
    # (1577836800000 = 2020-01-01T00:00:00Z).
    url = (
            base_url
            + f"bbox={left},{bottom},{right},{top}"
            + "&bboxSR=4326&imageSR=4326"
            + "&time=1577836800000"
    )
    print(f"\nworldpop_url:\n{url}")  # debug use

    max_times = 10
    flag = False
    for i in range(max_times):
        try:
            response = requests.get(url)
            response.raise_for_status()
            flag = True
            break
        except requests.exceptions.HTTPError as http_err:
            continue
        except requests.exceptions.ConnectionError as conn_err:
            continue
        except requests.exceptions.Timeout as timeout_err:
            continue
        except requests.exceptions.RequestException as req_err:
            continue

    if flag == False:
        raise ("Network error for accessing https://worldpop.arcgis.com/.")

    img_bytes = BytesIO(response.content)
    with MemoryFile(img_bytes) as memfile:
        with memfile.open() as raster:
            # get the population of the region with the boundary of the region
            coords = region.geometry
            try:
                out_img, _ = mask(dataset=raster, shapes=[coords], crop=True)
                population = float(out_img[out_img > 0].sum())
            except:
                population = 0

            if not isinstance(population, (int, float)):
                raise ValueError(f"Population of {region} is invalid: {population}")

            # EPSG code calculation
            epsg = calculate_utm_epsg(left, bottom)
            # get the area of the region
            geo_series = gpd.GeoSeries([coords], crs="EPSG:4326")
            area = geo_series.to_crs(f"EPSG:{epsg}").area.item() / 1e6

    return population, area


def worldpop(area_shp, token=None, num_proc=10):
    '''
    get regional population via worldpop posited on 
    Esri ArcGIS Living Atlas of the World "WorldPop_Total_Population_100m"
    '''
    # args
    args = [region for _, region in area_shp.iterrows()]

    # get the population of regions in the target area
    # # parallel version
    # populations = []
    # areasizes = []
    # with Pool(processes=num_proc) as pool:
    #     with tqdm(total=len(area_shp), desc=f" -- Population of regions") as pbar:
    #         for result in pool.imap(population_one_region, args):
    #             populations.append(result[0])
    #             areasizes.append(result[1])
    #             pbar.update(1)

    # sequential version
    populations = []
    areasizes = []
    for _, region in tqdm(area_shp.iterrows(), total=area_shp.shape[0], desc=f" -- Population of regions"):
        population, areasize = population_one_region(region)
        populations.append(population)
        areasizes.append(areasize)

    # save the population and areasize of the area
    feat = np.array([populations, areasizes]).T

    return feat
