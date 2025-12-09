# Fukuoka-shi Case Study – COMP0173 Coursework 2

This document describes how I adapted the GlODGen / WeDAN pipeline from the
Liverpool setting to **Fukuoka-shi** (福岡市), and how I interpret the generated
commuting OD patterns in the context of **SDG 11 (Sustainable Cities)**
and **SDG 10 (Reduced Inequalities)**.

## 0. Fukuoka case context

Fukuoka-shi is the largest city on Kyushu island and is an important economic center in southern Japan. Approximately 1.6 million people live there. The city offers a well-developed public transport system that includes subway and bus networks. It features a combination of crowded urban areas like
Hakata and Tenjin along with suburban residential zones.

### 0.1 Why Fukuoka?

Issue: Urban Traffic Congestion and Commuting

Fukuoka’s central districts, such as Tenjin and Hakata, are experiencing increased rush-hour congestion as the city expands. Major projects like the "Tenjin Big Bang" are increasing the flow of people, and local officials stress the need to reduce subway and road traffic jams. In July 2025, the city
launched a "Congestion Mitigation Project" [4] that promises more subway cars and better road conditions to ensure smooth travel. [5] However, buses and cars still crowd downtown streets during peak times, and even specific local actions, like improved signage and alternate routes in Minami Ward,
are necessary to relieve congestion. [6]

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

- Shapefile: assets/fukuoka_shi_grid_431_v2.shp
- Number of cells: (431, 431) -> around the size of repo's Liverpool shp with (252, 252) cells
- Construction steps:
    1. download sub-ward level polygons from [国土数値情報ダウンロードサイト](https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N03-2025.html) (MLIT Japan), `N03-20250101_40_GML.zip`
    2. in `./playground.ipynb`, process the shp file to get the 950 sub-wards shp covering Fukuoka-shi to `./assets/fukuoka_wards_n03b.shp`
    3. as 950 is too many nodes, using `./s0b_build_fukuoka_grid.py` aggregate sub-wards into coarser 431 regular grid cells which is reasonable size for inferencing using single AWS GPU instance `ml.g5.xlarge`
    4. save the resulting grid shapefile to `./assets/fukuoka_shi_grid_431_v2.shp`
- sample output from running `./s0b_build_fukuoka_grid.py`:

```
Total area (km^2): 496.03 (> 343 because EPSG:3857 (Web Mercator) inflation)
Target cell length (km): 1.29
Attaching ward info using column 'N03_005'...
Cells with ward: 300
Cells without ward: 131
Saved grid shapefile to: ./assets/fukuoka_shi_grid_431_v2.shp
```

## 2. Population calibration and node features

### 2.1 Using official CSV for Fukuoka

Similar to Liverpool which uses repo's shp file and [GBR 2025 WorldPop TIF][2] for population estimate, for Fukuoka I use the [JPN 2025 WorldPop TIF][3]:

```python
# ./generate_od/generator.py
def construct_inputs(self):
    # ...
    if self.city_name == FUKUOKA:  # no longer used
        worldpop = build_fukuoka_features_from_csv(
            self.area,
            csv_path=FUKUOKA_POPULATION_CSV,
            ward_col="N03_005",
        )
    else:  # Fukuoka-shi and Liverpool
        worldpop = self._fetch_worldpop(self.area)
    # ...
```

The function `build_fukuoka_features_from_csv` joins the grid with a CSV containing ward-level population and distributes population **equally** to area down to grid cells. This is a limitation and likely **underestimates intra-ward heterogeneity**. (But `build_fukuoka_features_from_csv` is not used
anymore)

So here `worldpop[:, 0]` and `worldpop[:, 1]` are estimated population per grid cell and area per grid cell in km² respectively, derived from the CSV data.

### 2.2 Calibration to official totals

Since the polygon shape and Worldpop TIF-derived cell population and area features may not exactly sum to the official Fukuoka totals,
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

## 3. Sampling hyperparameter for Fukuoka

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

After hyperparameter experiment in a data-rich city (see [liverpool_reproduction.md](liverpool_reproduction.md)), I then reused the same configuration in a data-poor city (Fukuoka Shi). For Fukuoka experiment I used the following parameters:

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

Given the real Fukuoka OD data does not exist, there is no ground truth for objective evaluation. After OD generation of Fukuoka using [s2_generate_odf.py](../s2_generate_odf.py), I created the script [s4_evaluate_fukuoka.py](../s4_evaluate_fukuoka.py) and analysed:

- shape of the (431, 431) OD matrix;
- total flows: sum of all OD entries;
- overall outflows and inflows per cell statistics (mean, max, std);
- correlation between outflows / inflows and population;
  with results below:

```
           pop  out_flow  in_flow  out_per_capita  in_per_capita
count    431.0     431.0    431.0           431.0   4.310000e+02
mean    3896.0   22064.0  22064.0       3925781.0   3.585492e+08
std     5062.0   22089.0  19100.0      11891519.0   1.083002e+09
min        0.0       8.0    799.0             3.0   2.000000e+00
25%       18.0     312.0   4754.0             6.0   5.000000e+00
50%      639.0   14132.0  17411.0            15.0   2.600000e+01
75%     7635.0   42648.0  35140.0            30.0   2.590000e+02
max    19158.0   77410.0  95226.0      68000000.0   6.820000e+09
Corr(pop, out_flow) = 0.895
Corr(pop, in_flow)  = 0.852
```

To complement the above numerics, I plot two graphs using the same script:

- ![pop vs inflows](img/fukuoka_pop_vs_in.png "pop vs inflows")
- ![pop vs outflows](img/fukuoka_pop_vs_out.png "pop vs outflows")

We could see a correlation of 0.895 between population and outgoing commuters. This indicates:

- cells with higher population tend strongly to have more commuters leaving them.
- commuting is roughly "people living here go somewhere else".

And a correlation of 0.852 between population and incoming commuters, which indicates:

- dense areas also tend to be destinations (mixed residential + employment).
- this is slightly lower than the outflow correlation, so not every dense residential area is a big job centre, which is realistic.

These values are high, but not "trivially 1.0", so:

- the OD flows are strongly shaped by population,
- but not purely proportional. Other factors such as distance and traffics are also in the game.

For Liverpool, I quantify performance against the reference OD in repo via CPC and MAE and compare directly with Rong et al. (2025).
But for Fukuoka, I lack ground-truth OD matrices, so I cannot compute the same error metrics.

[//]: # (Instead, we apply the model to Fukuoka’s administrative regions and analyse the plausibility and structural properties of the generated flows &#40;e.g., flows concentrating on Tenjin/Hakata as employment hubs&#41;. This demonstrates contextual transfer of the method, but not validated accuracy.)

### 4.2 Spatial pattern with visualization (qualitative)

```
    Name  Cell Index  In Flow  Out Flow   Population  In Rank  Out Rank
  Tenjin         293  44292.0   50476.0 16733.093750     60.0      50.0
Nishijin         229  65775.0   58347.0 16545.968750     12.0      20.0
  Yakuin         312  80860.0   51019.0 15944.148438      3.0      47.0
  Ohashi         348  95226.0   59860.0 13796.330078      1.0      18.0
  Hakata         332  31446.0   57823.0 13387.197266    137.0      22.0
```

Fukuoka is pretty polycentric. The above table shows major Fukuoka-shi primary CBDs, and the corresponding inflow outflow rank among the 431 regions. All of them are within the top 34% (with Hakata the worst among them).

![](img/fukuoka_pop_flow_heatmaps.png)

The above 3 plots showing population, incoming flow and outgoing flow heatmap respectively with star marked on the 5 major CBDs. We can see patterns like:

- existence of strong CBDs (e.g. Tenjin and Ohashi) attracting many commuters;
- Fukuoka is of polycentricity (multiple centres) vs a single dominant CBD;
- peripheral cells with high outgoing flows to central zones (long-distance commuters).

| ![](img/fukuoka_ego_origin_inflow_0.png) | ![](img/fukuoka_ego_origin_outflow_0.png) |
|-------------------------------------------------|--------------------------------------------------|
| ![](img/fukuoka_ego_origin_inflow_1.png) | ![](img/fukuoka_ego_origin_outflow_1.png) |
| ![](img/fukuoka_ego_origin_inflow_2.png) | ![](img/fukuoka_ego_origin_outflow_2.png) |
| ![](img/fukuoka_ego_origin_inflow_3.png) | ![](img/fukuoka_ego_origin_outflow_3.png) |
| ![](img/fukuoka_ego_origin_inflow_4.png) | ![](img/fukuoka_ego_origin_outflow_4.png) |

The above plots show 5 pairs of the CBDs' OD inflow and outflow ego networks for Fukuoka-shi. I plot their top 50 strongest OD links. This gives a sense of where people coming to and from these CBDs.

![](img/fukuoka_3_bands.png)

The figure above breaks down the generated Fukuoka OD flows into three quantile bands. 

Left (≥95th percentile, top 5% of flows, and the line color represents the strength: blue(75%) < red(97%) < yellow(100%)): only the busiest commuting corridors remain. These connections mostly link dense residential areas in the south and southwest to the main CBD around Tenjin, Hakata, and Ohashi. They form the backbone of the commuting network. 

Middle (90–95th percentile, next 5%): more strong links show up within the central wards. This reveals secondary hubs like Nishijin and Yakuin and improves connections across the CBD in the urban core. 

Right (75–90th percentile, next 15%): a dense network of medium-strength flows fills in the rest of the built-up area. Many east-west and radial trips come from outer residential areas into the central employment zone. This supports the view of Fukuoka as a polycentric city instead of a single-CBD structure.

⸻

## 5. Implications for SDG 11 / 10 and limitations

### 5.1 SDG 11 – Sustainable Cities and Communities

Potential usage:

- Identify high-demand areas where improvements in public transport, like more rail or bus services and dedicated lanes, could have the biggest effect.
- Explore accessibility: which neighborhoods have limited access to major job centers, such as long average commute times with few strong OD links?

Caveats:

- The WeDAN model uses US commuting data, not data from Japan, so the commuting habits, job-housing balance, and transport systems are different.
- There is no confirmed ground truth for the OD matrix. The generated origin-destination data is modeled, not observed, and should be seen as a scenario or hypothesis, not as established fact.

### 5.2 SDG 10 – Reduced Inequalities

The Fukuoka OD matrix could help us think about inequality in:

- spatial mismatch: areas with a high outflow but few local jobs.
- potential transport poverty: peripheral zones that need long trips to reach employment centers.

However, to truly study inequality, we would need more information:

- income, rent, and job quality per cell.
- public transport access and travel times, not just straight-line distances.

### 5.3 Model limitations

Key limitations when using GlODGen and WeDAN for Fukuoka:

1. Domain shift: the WeDAN model trained on US areas [7]. Japanese land-use and building patterns may not be fully represented by the same RemoteCLIP and WeDAN mapping.
2. Zoning and grid design: the 431-cell grid is my design choice and will influence OD patterns, for example, the Modifiable Areal Unit Problem.
3. Calibration assumptions:
    - I rescale population and area of each cell to match official city figures.
4. Uncertainty not fully quantified:
    - The 50 diffusion samples are averaged, but I do not formally account for uncertainty in SDG indicators.
    - A more complete analysis would look at variance across samples.


## 6. Summary

In this case study, I:

- reused the GlODGen pipeline and WeDAN model, which is trained on US commuting data.
- defined a 431-cell grid for Fukuoka-shi and calibrated population features using the latest WorldPop population TIF and government data.
- generated a reasonable commuting OD matrix for Fukuoka and analyzed its global structure and spatial pattern.
- discussed how this model could support SDG 11 and 10 analyses while emphasizing the limitations of cross-country transfer and generative modeling.

This part represents the "adaptation to new context" section of COMP0173 Coursework 2, complementing the repository's Liverpool as a baseline reproduction.

[1]: https://odm.bodik.jp/tl/dataset/401307_population_touroku_population

[2]: https://hub.worldpop.org/geodata/summary?id=49113

[3]: https://hub.worldpop.org/geodata/summary?id=73951

[4]: https://www.city.fukuoka.lg.jp/shisei/mayor/interviews/documents/250723_konzatukanwaproject.pdf#:~:text=%E5%A4%A9%E7%A5%9E%E3%83%93%E3%83%83%E3%82%B0%E3%83%90%E3%83%B3%E3%81%AA%E3%81%A9%E3%81%AE%E3%81%BE%E3%81%A1%E3%81%A5%E3%81%8F%E3%82%8A%E3%81%8C%E9%80%B2%E5%B1%95%E3%81%97%E3%80%81%E4%BB%8A%E5%BE%8C%E3%81%BE%E3%81%99%E3%81%BE%E3%81%99%E4%BA%BA%E3%81%AE%E5%8B%95%E3%81%8D%E3%81%8C%E6%B4%BB%E7%99%BA%E3%81%AB%E3%81%AA%E3%82%8B%E3%81%AA%E3%81%8B%E3%80%81%20%E5%9C%B0%E4%B8%8B%E9%89%84%E3%82%84%E9%81%93%E8%B7%AF%E4%BA%A4%E9%80%9A%E3%81%AE%E6%B7%B7%E9%9B%91%E3%81%AB%E5%AF%BE%E5%BF%9C%E3%81%97%E3%81%A6%E3%81%84%E3%81%8F%E3%81%93%E3%81%A8%E3%81%8C%E9%87%8D%E8%A6%81%E3%81%A8%E3%81%AA%E3%82%8A%E3%81%BE%E3%81%99%E3%80%82%20%E3%81%93%E3%81%AE%E5%BA%A6%E6%94%B9%E5%AE%9A%E3%81%97%E3%81%9F%E3%80%8C%E9%83%BD%E5%B8%82%E4%BA%A4%E9%80%9A%E5%9F%BA%E6%9C%AC%E8%A8%88%E7%94%BB%E3%80%8D%E3%82%84%E6%94%B9%E5%AE%9A%E4%B8%AD%E3%81%AE%E3%80%8C%E9%81%93%E8%B7%AF%E6%95%B4%E5%82%99%E3%82%A2%E3%82%AF%E3%82%B7%E3%83%A7%E3%83%B3%E3%83%97%E3%83%A9%E3%83%B3%EF%BC%92%EF%BC%90%EF%BC%92%EF%BC%98%E3%80%8D%20%E3%82%92%E8%B8%8F%E3%81%BE%E3%81%88%E3%80%81%E3%82%BD%E3%83%95%E3%83%88%E7%9A%84%E3%81%AA%E5%8F%96%E7%B5%84%E3%81%BF%E3%82%84%E4%BA%A4%E9%80%9A%E5%9F%BA%E7%9B%A4%E6%95%B4%E5%82%99%E3%81%AA%E3%81%A9%E3%80%81%E6%B7%B7%E9%9B%91%E7%B7%A9%E5%92%8C%E3%81%AB%E5%90%91%E3%81%91%E3%81%9F%E6%A7%98%E3%80%85%E3%81%AA%E6%96%BD%E7%AD%96%E3%82%92%20%E6%8E%A8%E9%80%B2%E3%81%97%E3%80%81%E8%AA%B0%E3%82%82%E3%81%8C%E5%BF%AB%E9%81%A9%E3%81%AB%E7%A7%BB%E5%8B%95%E3%81%A7%E3%81%8D%E3%82%8B%E4%BA%A4%E9%80%9A%E7%92%B0%E5%A2%83%E3%81%A5%E3%81%8F%E3%82%8A%E3%81%AB%E7%B7%8F%E5%90%88%E7%9A%84%E3%81%AB%E5%8F%96%E3%82%8A%E7%B5%84%E3%82%93%E3%81%A7%E3%81%BE%E3%81%84%E3%82%8A%E3%81%BE%E3%81%99%E3%80%82,%E2%96%A0%E4%BB%8A%E5%BE%8C%E3%81%AE%E4%B8%BB%E3%81%AA%E5%8F%96%E7%B5%84%E3%81%BF%EF%BC%88%E4%B8%AD%E9%95%B7%E6%9C%9F%E7%9A%84%E3%81%AA%E5%8F%96%E7%B5%84%E3%81%BF%E3%82%82%E5%90%AB%E3%82%80%EF%BC%89

[5]: https://www.city.fukuoka.lg.jp/shisei/mayor/interviews/documents/250723_konzatukanwaproject.pdf#:~:text=%E3%80%87%E9%83%BD%E5%BF%83%E9%83%A8%E3%81%AE%E9%81%93%E8%B7%AF%E4%BA%A4%E9%80%9A%E3%81%AE%E5%86%86%E6%BB%91%E5%8C%96%20%E3%83%BB%E7%A6%8F%E5%B2%A1%E5%B8%82%E9%A7%90%E8%BB%8A%E5%A0%B4%E3%83%8A%E3%83%93%E5%A4%A9%E7%A5%9E%E7%89%88%20%E3%83%BB%E5%9F%8E%E6%9D%B1%E6%A9%8B%E4%BA%A4%E5%B7%AE%E7%82%B9%E3%81%AE%E5%8F%B3%E6%8A%98%E3%83%AC%E3%83%BC%E3%83%B3%E8%A8%AD%E7%BD%AE%20%E3%83%BB%E5%A4%A9%E7%A5%9E%E9%80%9A%E7%B7%9A%E3%81%AE%E5%BB%B6%E4%BC%B8%20%E3%83%BB%E9%82%A3%E3%81%AE%E6%B4%A5%E9%80%9A%E3%82%8A%EF%BC%88%E9%82%A3%E3%81%AE%E6%B4%A5%E5%A4%A7%E6%A9%8B%EF%BC%89%EF%BC%96%E8%BB%8A%E7%B7%9A%E5%8C%96,%E3%83%BB%E5%9B%BD%E9%81%93%EF%BC%93%E5%8F%B7%E3%83%90%E3%82%A4%E3%83%91%E3%82%B9%20%E7%AB%8B%E4%BD%93%E5%8C%96%EF%BC%9C%E5%9B%BD%E4%BA%8B%E6%A5%AD%EF%BC%9E%20%E3%83%BB%E7%A6%8F%E5%B2%A1%E9%AB%98%E9%80%9F%EF%BC%93%E5%8F%B7%E7%B7%9A%EF%BC%88%E7%A9%BA%E6%B8%AF%E7%B7%9A%EF%BC%89%E5%BB%B6%E4%BC%B8%EF%BC%9C%E7%A6%8F%E5%8C%97%E5%85%AC%E7%A4%BE%E4%BA%8B%E6%A5%AD%EF%BC%9E%20%E3%83%BB%E3%83%9C%E3%83%88%E3%83%AB%E3%83%8D%E3%83%83%E3%82%AF%E4%BA%A4%E5%B7%AE%E7%82%B9%EF%BC%88%E4%B8%BB%E8%A6%81%E6%B8%8B%E6%BB%9E%E7%AE%87%E6%89%80%EF%BC%89%E3%81%AE%E6%B7%B7%E9%9B%91%E5%AF%BE%E7%AD%96%E6%A4%9C%E8%A8%8E%20%E3%81%AA%E3%81%A9

[6]: https://www.city.fukuoka.lg.jp/shisei/kouhou-hodo/hodo-happyo/2024/documents/fukuokasiminamikunotametikunokoutuujuutaitaisaku.pdf#:~:text=%E7%A6%8F%E5%B2%A1%E5%9B%BD%E9%81%93%E4%BA%8B%E5%8B%99%E6%89%80%20%20%E3%80%87%E5%AE%9F%E6%96%BD%E6%A6%82%E8%A6%81%EF%BC%9A%20%E3%82%AC%E3%83%B3%E3%82%BB%E3%83%B3%E3%82%BF%E3%83%BC%E5%85%A5%E5%8F%A3%E4%BA%A4%E5%B7%AE%E7%82%B9%E3%81%AB%E3%81%8A%E3%81%84%E3%81%A6%E3%80%81%E4%BA%95%E5%B0%BB%E6%96%B9%E9%9D%A2%E3%81%8B%E3%82%89%E5%A4%A9%E7%A5%9E%E3%83%BB%E5%A4%A7%E6%A9%8B%E6%96%B9%E9%9D%A2%E3%81%B8%E3%81%AE%20%E5%8F%B3%E6%8A%98%E8%BB%8A%E4%B8%A1%E3%81%AE%E6%B5%81%E5%85%A5%E3%82%92%E6%8A%91%E5%88%B6%E3%81%99%E3%82%8B%E3%81%9F%E3%82%81%E3%80%81%E4%BA%A4%E9%80%9A%E5%88%86%E6%95%A3%E3%82%92%E6%8E%A8%E5%A5%A8%E3%80%82,%E6%9D%B1%29%E5%87%BA%E5%8F%A3%E3%81%AB%E3%81%8A%E3%81%84%E3%81%A6%E3%80%81%E6%B8%8B%E6%BB%9E%E3%81%8C%E7%99%BA%E7%94%9F%E3%81%97%E3%81%A6%E3%81%84%E3%82%8B%E5%87%BA%E5%8F%A3%E3%81%AE

[7]: https://arxiv.org/abs/2407.15823
