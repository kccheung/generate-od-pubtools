import datetime

import geopandas as gpd
import matplotlib.pyplot as plt
from generate_od import generator
# from constants import out_path  # if you want Fukuoka instead of Liverpool
import pandas as pd


if __name__ == "__main__":
    # 1. init generator
    my_generator = generator.Generator()

    # 2. set satellite token
    my_generator.set_satetoken("xxxxxxxxxxxxxxx")  # ArcGIS World_Imagery token

    # 3. load area shapefile
    # For Liverpool (original example):
    area = gpd.read_file("./assets/example_data/shapefile/GB_Liverpool/regions.shp")
    # For Fukuoka, use the shapefile you used in s1:
    # area = gpd.read_file(out_path)

    my_generator.load_area(area)

    # 4. generate OD matrix and capture the result
    od_hat = my_generator.generate()
    print("OD matrix shape:", od_hat.shape)
    print("OD matrix (top-left 5x5):")
    print(od_hat[:5, :5])
    print("Min / max OD:", od_hat.min(), od_hat.max())
    print("Total flows:", od_hat.sum())
    print("Zero diagonal? ", (od_hat.diagonal() == 0).all())

    # save the OD matrix to CSV
    od_df = pd.DataFrame(
        od_hat,
        index=my_generator.area.index,  # or use area["region_id"] if you have
        columns=my_generator.area.index
    )
    od_df.to_csv(f"outputs/od_matrix_liverpool_{datetime.datetime.now()}.csv")

    # 5. plot and show the arc chart
    fig = my_generator.plot_arc_chart()
    fig.savefig("outputs/od_arc_liverpool.png", dpi=200, bbox_inches="tight")
    plt.show()  # this is the bit you were missing
