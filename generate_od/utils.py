import sys
import warnings

import numpy as np
import geopandas as gpd

from shapely.geometry import Polygon

from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler

from shapely.geometry import LineString
import contextily as cx
import matplotlib
import matplotlib.pyplot as plt


class Log_transformer():
    def __init__(self):
        pass

    def fit_transform(self, x):
        return np.log1p(x)

    def inverse_transform(self, x):
        return np.expm1(x)


def recover_od_shapes(batchlization):
    shapes = []
    i = 0
    while i < batchlization.shape[0]:
        if batchlization[i, i] == 1:
            count = 1
            while i + count < batchlization.shape[0] and batchlization[i, i + count] == 1:
                count += 1
            shapes.append(count)
            i += count
        else:
            i += 1
    return shapes


def reshape_matrix(matrix, k=100):
    n = matrix.shape[0]

    if k > n:
        padding = np.zeros((n, k - n))
        return np.hstack([matrix, padding])
    if k < n:
        return matrix[:, :k]
    return matrix


def MinMaxer(data):
    return MinMaxScaler(feature_range=(-1, 1)).fit(data)


def get_one_point(shp):
    '''
    get one point from the shapefile
    '''
    first_geometry = shp.geometry.iloc[0]

    # for different types of geometry, get the first point
    if first_geometry.geom_type == 'Polygon':
        first_point = first_geometry.exterior.coords[0]
    elif first_geometry.geom_type == 'MultiPolygon':
        first_polygon = list(first_geometry.geoms)[0]
        first_point = first_polygon.exterior.coords[0]
    else:
        raise ValueError('Geometry type not supported')

    pointx, pointy = first_point[0], first_point[1]

    return pointx, pointy


def extract_dis_adj_matrix(area_shp: gpd.GeoDataFrame):
    '''
    Euclidean distance matrix
    get the distance matrix for regions in the area
    based on the shapefile of the area
    '''
    # change the coordinate system to UTM
    pointx, pointy = get_one_point(area_shp)
    epsg = calculate_utm_epsg(pointx, pointy)
    area_shp = area_shp.to_crs(f"EPSG:{epsg}")
    area_shp["centroid"] = area_shp["geometry"].centroid

    # dis
    points = area_shp['centroid'].apply(lambda p: [p.x, p.y]).tolist()
    dist_matrix = distance_matrix(points, points)

    dis = dist_matrix.astype(np.float32)

    return dis


def calculate_utm_epsg(longitude, latitude):
    """
    Calculate the UTM zone and corresponding EPSG code for a given longitude and latitude.

    Parameters:
    longitude (float): The longitude of the location.
    latitude (float): The latitude of the location.

    Returns:
    int: The EPSG code for the UTM zone.
    """
    # Determine the UTM zone from the longitude
    utm_zone = int((longitude + 180) / 6) + 1

    # Determine the hemisphere and construct the EPSG code
    if latitude >= 0:
        # Northern Hemisphere
        epsg_code = 32600 + utm_zone
    else:
        # Southern Hemisphere
        epsg_code = 32700 + utm_zone

    return epsg_code


def progressbar(cur, total=100):
    percent = '{:.2%}'.format(cur / total)
    sys.stdout.write('\r')
    sys.stdout.write("[%-100s] %s" % ('=' * int(cur), percent))
    sys.stdout.flush()


def schedule(blocknum, blocksize, totalsize):
    '''
    blocknum: downloaded blocks
    blocksize: block size
    totalsize: total size
    '''
    if totalsize == 0:
        percent = 0
    else:
        percent = blocknum * blocksize / totalsize
    if percent > 1.0:
        percent = 1.0
    percent = percent * 100
    sys.stdout.write("\r -- : %.2f%%" % percent)
    sys.stdout.flush()


def deg2XY(lon_deg, lat_deg, zoom=15):
    '''
    The satellite images are tiles and each tile has a unique x and y index.
    '''
    # convert the latitude to radians
    lat_rad = np.radians(lat_deg)

    # the total number of the tiles in the x and y direction
    n = 2.0 ** zoom

    # the x index of the tile
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    # the y index of the tile
    ytile = int((1.0 - np.log(np.tan(lat_rad) + (1 / np.cos(lat_rad))) / np.pi) / 2.0 * n)

    return xtile, ytile


def XY2deg(x, y, zoom=15):
    # the total number of the tiles in the x and y direction
    n = 2.0 ** zoom

    # the longitude of the tile
    lon_deg = x / n * 360.0 - 180.0
    # the latitude of the tile
    lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y / n)))
    # convert the latitude to degrees
    lat_deg = np.rad2deg(lat_rad)

    return lat_deg, lon_deg


def compute_tile_coordinates(min_x, max_x, min_y, max_y):
    '''
    Batch version of YX2deg
    '''
    x_arr = np.arange(min_x, max_x + 1)
    y_arr = np.arange(min_y, max_y + 1)
    lon_arr, lat_arr = XY2deg_batch(x_arr, y_arr)

    return lon_arr, lat_arr, x_arr, y_arr


def XY2deg_batch(x_arr, y_arr, zoom=15):
    '''
    Batch version of num2deg
    '''
    n = 2.0 ** zoom
    lon_deg_arr = x_arr / n * 360.0 - 180.0
    lat_rad_arr = np.arctan(np.sinh(np.pi * (1 - 2 * y_arr / n)))
    lat_deg_arr = np.rad2deg(lat_rad_arr)

    return lon_deg_arr, lat_deg_arr


def create_tile_polygons(lon_arr, lat_arr, x_arr, y_arr):
    '''
    Create the polygons for each tile
    '''
    # create the lon and lat meshgrid
    lon_mesh, lat_mesh = np.meshgrid(lon_arr, lat_arr, indexing='ij')
    x_mesh, y_mesh = np.meshgrid(x_arr, y_arr, indexing='ij')

    # create the polygons
    vertices = np.array([
        lon_mesh[:-1, :-1], lat_mesh[:-1, :-1],
        lon_mesh[1:, :-1], lat_mesh[1:, :-1],
        lon_mesh[1:, 1:], lat_mesh[1:, 1:],
        lon_mesh[:-1, 1:], lat_mesh[:-1, 1:]
    ])
    vertices = vertices.reshape(4, 2, -1)
    vertices = np.transpose(vertices, axes=(2, 0, 1))
    polygons = [Polygon(p) for p in vertices]

    # create the x and y coordinates
    vertices_x_y = np.array([
        x_mesh[:-1, :-1], y_mesh[:-1, :-1],
        x_mesh[1:, :-1], y_mesh[1:, :-1],
        x_mesh[1:, 1:], y_mesh[1:, 1:],
        x_mesh[:-1, 1:], y_mesh[:-1, 1:]
    ])
    vertices_x_y = vertices_x_y.reshape(4, 2, -1)
    vertices_x_y = np.transpose(vertices_x_y, axes=(2, 0, 1))
    y_x = [f"{int(p[0][1])}_{int(p[0][0])}" for p in vertices_x_y]

    # create the GeoDataFrame
    tile_gpd = gpd.GeoDataFrame({"Y_X": y_x}, geometry=polygons, crs="EPSG:4326")

    return tile_gpd


def geometry_to_listXY(geometry):
    '''
    Convert the geometries to a list of x and y coordinates,
    which can be used as the index to download the satellite images (tiles).
    '''
    # get the bbox of the geometries
    '''
    may need to fix, when the box cross the 0 degree longitude
    '''
    # get the bbox of the geometries
    min_x, min_y, max_x, max_y = geometry.bounds
    minx, maxy = deg2XY(min_x, min_y)
    maxx, miny = deg2XY(max_x, max_y)

    # compute the coordinates of the tiles
    lon_arr, lat_arr, x_arr, y_arr = compute_tile_coordinates(minx, maxx, miny, maxy)
    tile_gpd = create_tile_polygons(lon_arr, lat_arr, x_arr, y_arr)

    # get the tiles that intersect with the geometries
    intersection = gpd.sjoin(tile_gpd, gpd.GeoDataFrame(data={"tmp": [0]}, geometry=[geometry], crs="EPSG:4326").drop(columns=["tmp"]), predicate='intersects', how='inner')
    Y_X = list(intersection.Y_X)

    # if there is only one tile, then return all the tiles
    if len(Y_X) < 2:
        for x in range(minx, maxx + 1):
            for y in range(miny, maxy + 1):
                Y_X.append(f"{y}_{x}")

    return Y_X


def get_YX_area(area_shp):
    '''
    Get the x and y coordinates of the tiles for the given area.
    '''
    # unify the Coordinate Reference System (CRS) of the geometries
    area_shp = area_shp.to_crs(epsg=4326)

    # get the coordinates of the tiles
    geometries = area_shp["geometry"]
    area_shp["Y_X"] = geometries.map(geometry_to_listXY)

    # remove the duplicate tiles
    Y_X = sum(list(area_shp["Y_X"]), [])
    Y_X = sorted(list(set(Y_X)))

    return area_shp, Y_X


def plot_od_arc_chart(od, geometries,
                      low=0,  # different for every region!
                      high=0,  # different for every region!
                      add_basemap=True):
    """
    low / high are percentiles between 0 and 1, not raw flow counts.

    A safe, reasonable choice:
    • low = 0.8 (only show top 20% of flows)
    • high = 0.99 (clip the very largest flows so colour scale isn’t dominated)

    Plot the arc chart for the given OD matrix on map.
    """

    # If low/high are not given (still near zero), compute from quantiles
    if low < 0.01 and high < 0.01:
        flows = od[od > 0].ravel()
        low = np.quantile(flows, 0.75)
        high = np.quantile(flows, 0.97)

    # 1. Build line GeoDataFrame in original CRS
    centroids = geometries.centroid
    line_records = []
    for i in range(od.shape[0]):
        for j in range(od.shape[1]):
            if i != j and od[i, j] > 0:
                line_records.append([centroids[i], centroids[j], od[i, j]])

    points_df = gpd.GeoDataFrame(
        line_records,
        columns=["geometry_o", "geometry_d", "flow"],
    )
    points_df["line"] = points_df.apply(
        lambda x: LineString([x["geometry_o"], x["geometry_d"]]), axis=1
    )
    points_df = gpd.GeoDataFrame(points_df, geometry="line", crs=geometries.crs)

    # 2. Reproject everything to Web Mercator
    geometries_3857 = geometries.to_crs(epsg=3857)
    points_3857 = points_df.to_crs(epsg=3857)

    # 3. Split into bands in 3857
    band_low = points_3857[points_3857["flow"] <= low].copy()
    band_mid = points_3857[(points_3857["flow"] > low) & (points_3857["flow"] <= high)].copy()
    band_high = points_3857[points_3857["flow"] > high].copy()

    print(f"Selected {len(points_3857)} OD arcs; "
          f"low={low:.1f}, high={high:.1f}")

    # 4. Plot everything on a single figure/axis
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)

    # base boundary
    geometries_3857.boundary.plot(ax=ax, linewidth=0.5, color="grey", alpha=0.7)

    # low flows
    band_low.plot(
        ax=ax,
        linewidth=0.05,
        color="#0308F8",
        alpha=0.2,
    )

    # medium flows
    band_mid.plot(
        ax=ax,
        linewidth=0.08,
        color="#FD0B1B",
        alpha=0.5,
    )

    # high flows
    band_high.plot(
        ax=ax,
        linewidth=0.12,
        color="yellow",
        alpha=0.8,
    )

    # 5. Adjust extent
    minx, miny, maxx, maxy = geometries_3857.total_bounds
    pad_x = (maxx - minx) * 0.05
    pad_y = (maxy - miny) * 0.05
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal")

    # 6. Basemap
    if add_basemap:
        try:
            cx.add_basemap(ax, crs=geometries_3857.crs,
                           source=cx.providers.CartoDB.Positron)
        except Exception as e:
            warnings.warn(f"Could not add basemap: {e}")

    # 7. Cosmetics
    low_i, high_i = int(low), int(high)
    plt.text(0.05, 0.95, f"0~{low_i}~{high_i}~∞",
             transform=ax.transAxes, fontsize=12,
             va="top", ha="left")

    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["right", "top", "left", "bottom"]:
        ax.spines[side].set_visible(False)

    return fig
