# FUKUOKA_SHP = "./assets/fukuoka_wards_n03.shp"  # sparse 7 wards only
# FUKUOKA_SHP = "./assets/fukuoka_wards_n03b.shp"  # finer 950 sub-wards
# FUKUOKA_SHP = "./assets/fukuoka_shi_grid_431.shp"  # 431-cell grid from 950 sub-wards
# FUKUOKA_SHP = "./assets/fukuoka_shi_grid_300.shp"  # 300-cell grid from dropping missing ward info on 431-cell grid
FUKUOKA_SHP = "./assets/fukuoka_shi_grid_431_v2.shp"  # 431-cell grid from 950 sub-wards with 7-wards info
CACHE_PATH = "./assets/regional_images_fukuoka_city.pkl"
JPN_TIF_PATH = "./assets/worldpop/jpn_pop_2025_CN_100m_R2025A_v1.tif"  # https://hub.worldpop.org/geodata/summary?id=73951
GBR_TIF_PATH = "./assets/worldpop/gbr_pop_2025_CN_100m_R2025A_v1.tif"  # https://hub.worldpop.org/geodata/summary?id=49113
FRA_TIF_PATH = "./assets/worldpop/fra_pop_2025_CN_100m_R2025A_v1.tif"  # https://hub.worldpop.org/geodata/summary?id=73407

# OD_PATH_LIVERPOOL = "./outputs/od_matrix_liverpool_2025-12-03_174537.224411.csv"
# OD_PATH_LIVERPOOL = "./outputs/od_liverpool_imageexport/od_matrix_liverpool_2025-12-02_203358.822592.csv"
OD_PATH_LIVERPOOL = "./outputs/od_matrix_liverpool_2025-12-05_212653.015100_gpu.csv"
# OD_PATH = "./outputs/od_matrix_fukuoka_2025-11-30_162929.892909.csv"
# OD_PATH = "./outputs/od_matrix_fukuoka_2025-11-30_222810.287300.csv"  # sample_times=5
# OD_PATH = "./outputs/od_matrix_fukuoka_2025-12-02_052952.606796.csv"  # sample_times=50
# OD_PATH = "./outputs/od_matrix_fukuoka_2025-12-03_150406.647073.csv"  # sample_times=50 with realistic pop and area_km2
# OD_PATH = "./outputs/od_matrix_fukuoka_2025-12-07_191315.507735.csv"  # sample_times=50 with realistic pop and area_km2
OD_PATH = "./outputs/od_matrix_fukuoka_2025-12-07_202609.067238.csv"  # sample_times=50 with 431_v2 shp

FUKUOKA = "Fukuoka"
FUKUOKA_SHI = "Fukuoka_shi"
LIVERPOOL = "Liverpool"
PARIS = "Paris"
MAPPING = {
    FUKUOKA_SHI: JPN_TIF_PATH,
    LIVERPOOL: GBR_TIF_PATH,
    PARIS: FRA_TIF_PATH,
}

FUKUOKA_CITY_FEAT = {
    "pop_total": 1_620_574,  # Fukuoka City 2025 census population
    "area_km2": 343.39,  # Fukuoka City area in km²
}
FUKUOKA_POPULATION_CSV = "./assets/401307_population_touroku_population_202510.csv"

LIVERPOOL_SHP = "./assets/example_data/shapefile/GB_Liverpool/regions.shp"
PARIS_SHP = "./assets/example_data/shapefile/FR_Paris/Paris.shp"

# constants.py (or a new module)
FUKUOKA_WARD_STATS = {
    "東区": {"pop": 291_749, "area_km2": 66.68},
    "博多区": {"pop": 212_108, "area_km2": 31.47},
    "中央区": {"pop": 176_739, "area_km2": 15.16},
    "南区": {"pop": 248_901, "area_km2": 30.98},
    "城南区": {"pop": 128_883, "area_km2": 16.02},
    "早良区": {"pop": 211_889, "area_km2": 95.88},
    "西区": {"pop": 190_288, "area_km2": 83.81},
}
