import datetime

import geopandas as gpd
import matplotlib.pyplot as plt

from constants import LIVERPOOL_SHP, FUKUOKA_SHP
from generate_od import generator
# from constants import out_path  # if you want Fukuoka instead of Liverpool
import pandas as pd

from utils import od_sanity_print

if __name__ == "__main__":
    # 1. init generator
    my_generator = generator.Generator()
    # my_generator.city_name = "Fukuoka"  # no worldpop query for Fukuoka
    my_generator.city_name = "Fukuoka_shi"  # need worldpop query for Fukuoka sub-wards

    # 2. set satellite token
    my_generator.set_satetoken("xxxxxxxxxxxxxxx")  # ArcGIS World_Imagery token

    # 3. load area shapefile
    # For Liverpool (original example):
    # area = gpd.read_file(LIVERPOOL_SHP)
    area = gpd.read_file(FUKUOKA_SHP)
    # For Fukuoka, use the shapefile you used in s1:
    # area = gpd.read_file(out_path)
    print(len(area))
    print(area.head())

    my_generator.load_area(area)

    # 4. generate OD matrix and capture the result
    # od_hat = my_generator.generate()
    od_hat = my_generator.generate(
        # sample_times=5
        sample_times=50
    )
    od_sanity_print(od_hat)

    # save the OD matrix to CSV
    od_df = pd.DataFrame(
        od_hat,
        index=my_generator.area.index,  # or use area["region_id"] if you have
        columns=my_generator.area.index
    )
    dt_str = str(datetime.datetime.now()).replace(" ", "_").replace(":", "")
    # od_df.to_csv(f"./outputs/od_matrix_liverpool_{dt_str}.csv")
    od_df.to_csv(f"./outputs/od_matrix_fukuoka_{dt_str}.csv")

    # 5. plot and show the arc chart
    fig = my_generator.plot_arc_chart()
    # fig.savefig(f"./outputs/od_arc_liverpool_{dt_str}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"./outputs/od_arc_fukuoka_{dt_str}.png", dpi=200, bbox_inches="tight")
    plt.show()  # this is the bit you were missing
