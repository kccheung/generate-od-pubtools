import shapely.geometry as geom
import geopandas as gpd
from generate_od.worldpop import worldpop

out_path = "assets/fukuoka_city.shp"

# Load your regional boundaries (shapefile or GeoDataFrame)
area_shp = gpd.read_file(out_path)

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
print(f"Features: [population_count, area_km2]: {population_features}")
