# FUKUOKA_SHP = "./assets/fukuoka_city.shp"
# FUKUOKA_SHP = "./assets/fukuoka_wards_n03.shp"  # 7 wards only
# FUKUOKA_SHP = "./assets/fukuoka_wards_n03b.shp"  # 950 sub-wards
FUKUOKA_SHP = "./assets/fukuoka_shi_grid_431.shp"  # 431 grid cells
CACHE_PATH = "./assets/regional_images_fukuoka_city.pkl"

# OD_PATH = "./outputs/od_matrix_liverpool_2025-11-29 16:54:25.789433.csv"
# OD_PATH = "./outputs/od_matrix_fukuoka_2025-11-30_162929.892909.csv"
# OD_PATH = "./outputs/od_matrix_fukuoka_2025-11-30_222810.287300.csv"  # sample_times=5
OD_PATH = "./outputs/od_matrix_fukuoka_2025-12-02_052952.606796.csv"  # sample_times=50

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
