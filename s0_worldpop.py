import numpy as np
import shapely.geometry as geom
import geopandas as gpd
from generate_od.worldpop import worldpop
from constants import *

# NOT USING ANYMORE, used build_fukuoka_features_from_csv from utils.py instead

# Load your regional boundaries (shapefile or GeoDataFrame)
area_shp = gpd.read_file(FUKUOKA_SHP)

# Ensure CRS is set (will be converted to WGS84 internally)
if area_shp.crs is None:
    area_shp = area_shp.set_crs("EPSG:4326")

print("CRS:", area_shp.crs)
print("Bounds:", area_shp.total_bounds)
print(area_shp.geometry.iloc[0].geom_type)
print(area_shp.is_valid)

# Fetch population data for all regions
population_features = worldpop(area_shp, num_proc=1)

# The result contains [population, area_size] for each region
print(f"Population data shape: {population_features.shape}")
np.set_printoptions(suppress=True)
# print(f"Features: [population_count, area_km2]: {population_features}")

# WorldPop raw model output
raw_pop = population_features[:, 0]
area_km2 = population_features[:, 1]
raw_pop_sum = population_features[:, 0].sum()

print("Features (raw): [population_count, area_km2]:", population_features)
print("Raw WorldPop:", raw_pop, "\nArea:", area_km2)
print("total population (all regions):", raw_pop_sum)
print("total area (all regions):", population_features[:, 1].sum())

# Scale to match 2025 census for Fukuoka-shi (~1,612,392)
scale = census_pop / raw_pop_sum

scaled_pop = raw_pop_sum * scale
print("Scale factor:", scale)
print("Scaled population (~census):", scaled_pop)

# # If you want to keep everything in one array:
population_features[:, 0] = raw_pop * scale
print("Features (scaled): [population_count, area_km2]:", population_features)
