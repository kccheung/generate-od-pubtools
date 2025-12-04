#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw/worldpop

# Japan
wget -O assets/worldpop/jpn_pop_2025_CN_100m_R2025A_v1.tif \
  "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/JPN/v1/100m/constrained/jpn_pop_2025_CN_100m_R2025A_v1.tif"

# UK
wget -O assets/worldpop/gbr_pop_2025_CN_100m_R2025A_v1.tif \
  "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/GBR/v1/100m/constrained/gbr_pop_2025_CN_100m_R2025A_v1.tif"

# France
wget -O assets/worldpop/fra_pop_2025_CN_100m_R2025A_v1.tif \
  "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/FRA/v1/100m/constrained/fra_pop_2025_CN_100m_R2025A_v1.tif"
