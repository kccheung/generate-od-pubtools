# FUKUOKA_SHP = "./assets/fukuoka_city.shp"
# FUKUOKA_SHP = "./assets/fukuoka_wards_n03.shp"  # 7 wards only
FUKUOKA_SHP = "./assets/fukuoka_wards_n03b.shp"  # 950 sub-wards
CACHE_PATH = "./assets/regional_images_fukuoka_city.pkl"
census_pop = 1_612_392  # Fukuoka City 2025 census population
FUKUOKA_POPULATION_CSV = "./assets/401307_population_touroku_population_202503.csv"

LIVERPOOL_SHP = "./assets/example_data/shapefile/GB_Liverpool/regions.shp"

# constants.py (or a new module)
FUKUOKA_WARD_STATS = {
    "東区":  {"pop": 291_749, "area_km2": 66.68},
    "博多区": {"pop": 212_108, "area_km2": 31.47},
    "中央区": {"pop": 176_739, "area_km2": 15.16},
    "南区":  {"pop": 248_901, "area_km2": 30.98},
    "城南区": {"pop": 128_883, "area_km2": 16.02},
    "早良区": {"pop": 211_889, "area_km2": 95.88},
    "西区":  {"pop": 190_288, "area_km2": 83.81},
}
