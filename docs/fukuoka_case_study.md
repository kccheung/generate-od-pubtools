---

## `docs/fukuoka_case_study.md`

```markdown
# Fukuoka-shi Case Study – COMP0173 Coursework 2

This document describes how I adapted the GlODGen / WeDAN pipeline from the
Liverpool setting to **Fukuoka-shi** (福岡市), and how I interpret the generated
commuting OD patterns in the context of **SDG 11 (Sustainable Cities)**
and **SDG 10 (Reduced Inequalities)**.

---

## 1. Fukuoka grid and study area

### 1.1 Study area

Target city:

- **Fukuoka-shi** (福岡市), Fukuoka Prefecture, Japan.
- Official stats used in calibration:
    - Population: **1,620,574** (approx., Fukuoka City 2025-10-31 estimate [1])
    - Area: **343.39 km²**

These values are stored in `FUKUOKA_CITY_FEAT`:

```python
# ./constants.py
# ...
FUKUOKA_CITY_FEAT = {
    "pop_total": 1_620_574,  # Fukuoka City 2025 census population
    "area_km2": 343.39,  # Fukuoka City area in km²
}
# ...
```

### 1.2 431-cell grid

Rather than using only the 7 wards or 950 sub-wards, I use a middle ground spatial resolution:

- Shapefile: assets/fukuoka_shi_grid_300.shp
- Number of cells: 300
- Construction steps:
    1. download sub-ward level polygons from [国土数値情報ダウンロードサイト](https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N03-2025.html) (MLIT Japan), `N03-20250101_40_GML.zip`
    2. in `./playground.ipynb`, process the shp file to get the 950 sub-wards shp covering Fukuoka-shi to `./assets/fukuoka_wards_n03b.shp`
    3. as 950 is too many nodes, using `./s0b_build_fukuoka_grid.py` aggregate sub-wards into coarser 300 regular grid cells which is reasonable size for trainning using single GPU
    4. Some grid cells at the periphery fell outside any ward polygon (e.g. sea area within the bounding box). These were removed so that each cell corresponds to a valid ward. The final grid therefore contains only land cells with well-defined ward labels.
    5. save the resulting grid shapefile to `./assets/fukuoka_shi_grid_300.shp`
- sample output:

```
Total area (km^2): 343.34
Target cell size (km): 1.653
Number of grid cells created: 300
Saved grid shapefile to: ./assets/fukuoka_shi_grid_300.shp
```

## 2. Population calibration and node features

### 2.1 Using official CSV for Fukuoka

Unlike Liverpool (which uses repo's shp file and online WorldPop exportimg API for population estimate), for Fukuoka I use the official Japanese population registration data:

```python
# ./generate_od/generator.py
def construct_inputs(self):
    # ...
    if self.city_name == "Fukuoka":
        worldpop = build_fukuoka_features_from_csv(
            self.area,
            csv_path=FUKUOKA_POPULATION_CSV,
            ward_col="N03_005",
        )
    else:
        worldpop = self._fetch_worldpop(self.area)
    # ...
```

The function `build_fukuoka_features_from_csv` joins the grid with a CSV containing ward-level population and distributes population **equally** to area down to grid cells. This is a limitation and likely **underestimates intra-ward heterogeneity**.

So here `worldpop[:, 0]` and `worldpop[:, 1]` are estimated population per grid cell and area per grid cell in km² respectively, derived from the CSV data.

A natural extension later would be to replace the current ward-uniform population features with grid-level population from a high-resolution raster  or Japanese 500m mesh census data (now I only have ward-level).

### 2.2 Calibration to official totals

Since the CSV-derived grid features may not exactly sum to the official Fukuoka totals,
I rescale them:

```python
# ./generate_od/generator.py
def construct_inputs(self):
    # ...
    raw_pop_total = worldpop[:, 0].sum()
    raw_area_total = worldpop[:, 1].sum()

    if getattr(self, "city_name", None) in [FUKUOKA_SHI]:
        target_pop_total = FUKUOKA_CITY_FEAT["pop_total"]
        target_area_total = FUKUOKA_CITY_FEAT["area_km2"]

        if raw_pop_total > 0:
            worldpop[:, 0] *= target_pop_total / raw_pop_total

        if raw_area_total > 0:
            worldpop[:, 1] *= target_area_total / raw_area_total

        print(
            f"[Fukuoka calibration]\n"
            f"raw_pop={raw_pop_total:.1f}, raw_area={raw_area_total:.2f} ->\n"
            f"scaled_pop={worldpop[:, 0].sum():.1f}, scaled_area={worldpop[:, 1].sum():.2f}"
        )
    # ...
```

This ensures total grid population matches latest census and total grid area matches Fukuoka-shi.
This calibration is important for making node features realistic for the Japanese context and allowing later scaling of OD flows to plausible commuting totals.

### 2.3 Node and edge features (same structure as Liverpool)

After calibration:

```python
# ./generate_od/generator.py
def construct_inputs(self):
    # ...
    nfeat = np.concatenate([img_feats, np.log1p(worldpop)], axis=1)
    distance = extract_dis_adj_matrix(self.area)
    distance = self.data_scalers["dis"].transform(distance.reshape([-1, 1])).reshape(distance.shape)
    # ...
```

So the WeDAN model sees Fukuoka through RemoteCLIP embeddings of ESRI satellite imagery (urban form); log-transformed population and area per cell; and scaled pairwise distances between cells.

## 3. Sampling hyperparameters for Fukuoka

The WeDAN model is pre-trained on a large US commuting dataset. I do not
change its architecture or diffusion training hyperparameters. Instead, I
adapt the original GlODGen pipeline around it: (i) with a new Fukuoka-specific grid, (ii) a new population feature builder using Japanese registration data and calibration.

Key sampling parameters:

```python
# ./generate_od/generator.py
def _generate_config(self):
    self.config = {
        # ...
        "T": 250,
        "DDIM_T_sample": 25,
        "sample_times": 50,
        "DDIM_eta": 0,
        # ...
    }
    # ...
```

After hyperparameter tuned in a data-rich city (see [liverpool_reproduction.md](liverpool_reproduction.md)), I then reused the chosen configuration in a data-poor city (Fukuoka Shi). For Fukuoka experiment I used the following parameters:

| parameter     | value | description                                                                                                                                                                                                                                                                                                                                                                                                               |
|---------------|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DDIM_T_sample | 25    | Keep as default<br/>As the author recommended, this value is tied to training-time and controls the probability flow evolution; altering it may break model consistency                                                                                                                                                                                                                                                   |
| sample_times  | 50    | Keep as default<br/>This controls how many full OD matrices you draw and then average<br/>Larger cities with highly skewed OD distributions (many long-distance OD pairs leading to sparse OD networks) usually require more sampling repetitions to achieve stable generation.<br/>For smaller cities, reducing the number of diffusion sampling steps can speed up inference without significantly hurting performance. |


In code, I simply call below for the main experiment.
```python
# ./s2_generate_odf.py
# ...
od_hat = my_generator.generate(
        sample_times=50
    )
# ...
```

## 4. Resulting commuting structure for Fukuoka

### 4.1 Global properties


[1]: https://odm.bodik.jp/tl/dataset/401307_population_touroku_population
