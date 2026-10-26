# Data Mining Project — Climate Weather Forecasting (2026/2027)


## Objective

Develop an end-to-end weather framework using ERA5-Land climate data for Algeria to forecast the 2-meter temperature class using data mining and machine learning techniques.

## Project Structure

```
dataminingproject/
├── data/
│   ├── raw/                    # Raw ERA5-Land NetCDF data
│   └── processed/              # Cleaned CSVs and numpy arrays
├── notebooks/
│   ├── step1_data_analysis_preprocessing.ipynb   # Step 1: EDA & Feature Engineering
│   ├── step2_supervised_learning.ipynb            # Step 2: KNN, SVM, RF, Voting, AdaBoost
│   └── step3_unsupervised_learning.ipynb          # Step 3: K-Means, DBSCAN, AGNES, AE-KMeans
├── src/
│   └── download_data.py        # Script to download ERA5 data from CDS
├── outputs/
│   └── figures/                # Saved plots and visualizations
├── venv/                       # Python virtual environment
├── requirements.txt
└── README.md
```

## Setup

```bash
# Install dependencies 
pip install -r requirements.txt
```

## Data Download

1. Register at https://cds.climate.copernicus.eu/
2. Get your API key from your profile
3. Create `~/.cdsapirc`:
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <your-api-key>
   ```
4. Run: `python src/download_data.py`

## Study Details

- **Area:** Algeria [18°N–38°N, 9°W–12°E]
- **Timeline:** Monthly means from 2015 to Present (includes 2026 data up to latest available month)
- **Dataset:** ERA5-Land Monthly averaged reanalysis
- **Spatial resolution:** Reduced by factor 4 (aggregating neighboring grid points)
