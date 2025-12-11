Use of Generative AI
--------
I have read and understood UCL’s guidance on the use of Generative AI. For this assignment, I used ChatGPT-5 mini and Grok Code Fast 1 as a coding assistant to help debug errors, clarify Python and library syntax, and suggest alternative implementations. All final code, analysis, and written explanations were reviewed, tested, and edited by me, and I take full responsibility for the submitted work.

# COMP0173 Coursework 2 – GlODGen for Liverpool & Fukuoka

This README documents **my own coursework work** built on top of the original
[`generate-od-pubtools`](https://github.com/tsinghua-fib-lab/generate-od-pubtools) / GlODGen codebase.

The goal of this fork is:

1. To **reproduce** the Liverpool commuting OD case used in the GlODGen / WeDAN papers (please check [liverpool_reproduction.md](liverpool_reproduction.md)), and
2. To **adapt** the pipeline to a new context: **Fukuoka-shi (Japan)**, as a case study
   for **AI for Sustainable Development (COMP0173, UCL)**, please check [fukuoka_case_study.md](fukuoka_case_study.md).

The upstream project and models are by **Rong et al.** and collaborators; all mistakes in this
coursework adaptation are mine.

---

## 1. Background & Objectives

This work is part of:

- **MSc AI for Sustainable Development (UCL)**
- Module: **COMP0173 – Artificial Intelligence for Sustainable Development**
- Coursework 2: *Reproduce an SDG-related AI paper and adapt it to a new context.*

Chosen paper & model:

- **GlODGen / WeDAN** – *Satellites Reveal Mobility: A Commuting Origin–Destination Flow Generator for Global Cities* (and the accompanying benchmark dataset + WeDAN diffusion model).

My concrete objectives:

1. **Liverpool baseline reproduction**

    - Use the public `generate-od` Python package to regenerate the **GB_Liverpool** OD matrix.
    - Compare against the reference OD file provided in the repo using:
        - RMSE, NRMSE, CPC (Common Part of Commuting),
        - simple visualisations (heatmaps, top-k flows).

2. **Fukuoka-shi case study**

    - Construct a 431-cell grid for Fukuoka-shi from government open source data: [国土数値情報ダウンロードサイト](https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N03-2025.html).
    - Calibrate population and area features using Japanese stats.
    - Run GlODGen pipepline (which includes RemoteClip-RN50 and WeDAN model) to generate a Fukuoka commuting OD matrix.
    - Interpret the pattern in terms of SDGs (e.g. SDG 11 Sustainable Cities, SDG 10 Inequalities).

---

## 2. Documentation Structure

This README is a **high-level index**. Detailed explanations are split into three files:

```text
docs/
  ├── README_cw2.md
  ├── liverpool_reproduction.md
  ├── fukuoka_case_study.md
  └── imgs/
       ├── ... (images used in the documentation)
```

## 3. Environment Setup

To run the code, please set up a Python environment below:
```bash
pip install generate-od  # this will smoothly install package dependencies
pip uninstall -y generate-od  # but we do not need the official package, just needs the dependencies
pip freeze  # check for installed dependencies
```

