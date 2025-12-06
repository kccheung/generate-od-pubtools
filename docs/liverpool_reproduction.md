# Liverpool Reproduction – COMP0173 Coursework 2

This document describes how I used the public `generate-od` / GlODGen pipeline
to reproduce the **GB_Liverpool** commuting OD matrix and compare it against the
reference OD provided in the original repository.

The focus here is on:

- how the Liverpool area / shapefile is loaded, how is the population data fetched from WorldPop;
- how the OD matrix is generated using `Generator.generate`;
- how my generated matrix compares to `generation.npy` via CPC / RMSE / NRMSE;
- simple visual comparisons (heatmaps, difference plots).

---

## 0. Population data fetching from worldpop

```shell
sh download_worldpop.sh  # this will download the GBR 2025 WorldPop TIF
```

I use this instead because the GlODGen's default WorldPop ArcGIS service to fetch population data would result in inaccurate result.

## 1. Area definition and shapefile loading

For Liverpool I use the original shapefile provided in the GlODGen example data:

- Shapefile: `./assets/example_data/shapefile/GB_Liverpool/regions.shp`
- Number of regions: **252**
- CRS: projected to WGS84 (EPSG:4326) inside the generator.

To generate OD flows for Liverpool, first need to make sure the folowing is set in `s2_generate_odf.py`:

```python
# ...
my_generator.city_name = LIVERPOOL  # local worldpop TIFF for liverpool
# ...
area = gpd.read_file(LIVERPOOL_SHP)
# ...
od_df.to_csv(f"./outputs/od_matrix_liverpool_{dt_str}.csv")
# ...
```

then run the following in terminal:

```shell
python s2_generate_odf.py
```

## 2. Inner workings

Internally, `generator.load_area`:

- checks CRS is defined;
- converts to WGS84 (epsg=4326) to be consistent with WorldPop and ESRI imagery.

### 2.1. Input features and OD generation pipeline

Once the area is loaded, the Generator builds the conditioning inputs for the
WeDAN diffusion model in `_construct_inputs()`:

#### 2.1.1 Population and area (WorldPop features)

For Liverpool, the code segment is:

```python
# ./generate_od/generator.py
# ...
def _fetch_worldpop(self, area_shp: gpd.GeoDataFrame):
    # ...
    if getattr(self, "city_name", None) in [FUKUOKA_SHI, LIVERPOOL, PARIS]:
        # use local WorldPop TIFF instead of ArcGIS ImageServer
        tif_path = MAPPING[self.city_name]  # for Liverpool: GBR_TIF_PATH
        worldpop_feats = worldpop_from_local_tif(area_shp, tif_path)
    else:
        worldpop_feats = worldpop(area_shp)
    # ...
```

where `worldpop_from_local_tif(...)` samples the local GBR 2025 WorldPop TIFF and computes for each region's estimated total population and area in km² (based on pixel coverage).

For Liverpool no extra calibration is currently applied (I only calibrate Fukuoka to Japanese census totals). For Liverpool I only print a sanity check:

```python
# ./generate_od/generator.py
def _construct_inputs(self):
    # ...
    raw_pop_total = worldpop[:, 0].sum()
    raw_area_total = worldpop[:, 1].sum()
    print(f"[Feature check] raw_pop={raw_pop_total:.1f}, raw_area={raw_area_total:.2f}")
    # ...
```

#### 2.1.2 Satellite imagery and RemoteCLIP features

For each region the generator fetches ESRI satellite images:

```python
# ./generate_od/generator.py
def _construct_inputs(self):
    # ...
    imgs = self._fetch_sateimgs(self.area)
    img_feats = extract_imgfeats_RemoteCLIP(self.vision_model,
                                            self.model_name,  # "RN50" which is the recommended default
                                            imgs,
                                            self.config["device"]
                                            )
    # ...
```

The vision backbone is RemoteCLIP-RN50, it is downloaded once into `./checkpoints/models--chendelong--RemoteCLIP`, and loaded into `self.vision_model` and used to embed satellite tiles.

#### 2.1.3 Node and edge features

Node features:

```python
# ./generate_od/generator.py
def _construct_inputs(self):
    # ...
    nfeat = np.concatenate([img_feats, np.log1p(worldpop)], axis=1)
    # n_indim = 97 in self.config, total dim = 1026 + 97 + 97 for (img, noisy_attr, pred_attr)
```

Edge features:

```python
# ./generate_od/generator.py
def _compute_distance(self):
    # ...
    distance = extract_dis_adj_matrix(self.area)


def _construct_inputs(self):
    # ...
    distance = self._compute_distance()
    distance = self.data_scalers["dis"].transform(distance.reshape([-1, 1])).reshape(distance.shape)
```

Then everything is converted to torch tensors and packed as:

- net = (n, e) where n includes image + attributes,
- masks for node/edge,
- distance and batchlization.

#### 2.1.4 OD generation (diffusion sampling)

Final OD generation is done by `Generator.generate`:

```python
# ./s2_generate_odf.py
# ...
od_hat = my_generator.generate(sample_times=50)
# 50 as suggested in the original paper and is the number of independent diffusion samples to average
# ...
```

Internally:

```python
# ./generate_od/generator.py
def generate(self, sample_times: int = 1):
    # ...
    for k in range(sample_times):
        net_hat = self.od_model.DDIM_sample_loop(n.shape, e.shape, c)
        _, e_hat = net_hat
        e_hat_np = e_hat.detach().cpu().numpy()

        if e_hat_mean is None:
            e_hat_mean = e_hat_np
        else:
            e_hat_mean += (e_hat_np - e_hat_mean) / (k + 1)

    e_hat = e_hat_mean
    # ...
```

Then the raw diffusion output is inverse-transformed:

```python
# ./generate_od/generator.py
def generate(self, sample_times: int = 1):
    # ...
    od_hat = self.data_scalers["od"].inverse_transform(od_hat.reshape([-1, 1])).reshape(od_hat.shape)
    od_hat = self.data_scalers["od_normer"].inverse_transform(od_hat)
    np.fill_diagonal(od_hat, 0)  # no self-commuting
    od_hat[od_hat < 0] = 0
    od_hat = np.floor(od_hat)  # integer commuters
    # ...
```

For Liverpool in the final reproduction run I use, DDIM_T_sample = 25 (default); and sample_times = 50 (default).

## 3. Comparing against the reference ./assets/example_data/CommutingOD/GB_Liverpool/generation.npy

The original repo includes a reference OD matrix:

- File: assets/example_data/CommutingOD/GB_Liverpool/generation.npy
- Shape: (252, 252)

I fix and record a random seed for each run, and ran a total of 10 independent runs with different seeds and compute below metrics vs the baseline.

```python
# ./utils.py
# ...
def rmse(F, F_hat):
    return np.sqrt(np.mean((F - F_hat) ** 2))


def nrmse(F, F_hat):
    F_mean = F.mean()
    denom = np.sqrt(np.mean((F - F_mean) ** 2))
    return rmse(F, F_hat) / denom


def cpc(F, F_hat):
    numerator = 2 * np.sum(np.minimum(F, F_hat))
    denominator = np.sum(F) + np.sum(F_hat)
    return numerator / denominator
# ...
```

On my final average result (after scaling):

| Metrics               | Average over 10 runs                          | Interpretation                                                                                   |
|-----------------------|-----------------------------------------------|--------------------------------------------------------------------------------------------------|
| Reference total flows | 4,883,625                                     |                                                                                                  |
| Scaled total flows    | 4,883,625 (matches reference by construction) |                                                                                                  |
| RMSE (scaled)         | ≈ 72.88                                       | commuters per OD pair (on average).                                                              |
| NRMSE (scaled)        | ≈ 0.99                                        | typical error is on the order of the natural variation of true flows.                            |
| CPC (scaled)          | ≈ 0.70                                        | about 70% of total commuting volume is overlapping; remaining 30% is redistributed across cells. |

```
=== 10 runs summary over seeds ===
   scaled_total       rmse     nrmse       cpc  seed
0     4883625.0  70.289939  0.958267  0.714789     0
1     4883625.0  73.514408  1.002226  0.706290     1
2     4883625.0  74.798215  1.019729  0.704687     2
3     4883625.0  73.048566  0.995876  0.702048     3
4     4883625.0  69.508009  0.947607  0.716466     4
5     4883625.0  75.036621  1.022979  0.696702     5
6     4883625.0  72.616525  0.989986  0.701730     6
7     4883625.0  73.806909  1.006214  0.699382     7
8     4883625.0  73.739475  1.005295  0.701381     8
9     4883625.0  72.449223  0.987705  0.704959     9

Mean ± std:
           rmse     nrmse       cpc
mean  72.880789  0.993588  0.704844
std    1.785529  0.024342  0.006340
```

Given the stochastic nature of diffusion sampling and likely version differences between the authors’ internal code and the public Python package, I interpret this as statistical reproduction, not bitwise equality.
Additionally, according to the authors: 
1. ESRI imagery (which is integrated in GlODGen) updates over time as the built environment evolves, so embeddings may also change over time.
2. Due to reviewer deadlines at the time, they did not lock the random seed when generating the sample dataset, so exact replication of sample matrices is not possible.
⸻

## 5. Lessons and limitations (Liverpool)
- GlODGen / WeDAN trained on US commuting data can still generate plausible OD patterns for Liverpool when conditioned on satellite imagery and population.
- Exact cell-by-cell reproduction of the reference matrix is not realistic:
  - diffusion sampling is stochastic;
  - training, package versions and source of population datasets may differ.
- Metrics like CPC and NRMSE provide a meaningful way to judge reproduction quality.
- For my coursework, I treat the Liverpool reproduction as successful in a statistical sense and then use the same pipeline for the unseen Fukuoka case study.
