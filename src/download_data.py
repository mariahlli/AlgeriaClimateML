import os
import sys
import zipfile
from pathlib import Path
import cdsapi
import xarray as xr

# Directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_DIR / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

# Algeria Bounding Box [North, West, South, East]
AREA = [38, -9, 18, 12]

# 12 Required Climate Variables
VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_latent_heat_flux",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
    "surface_sensible_heat_flux",
    "runoff",
    "total_evaporation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "total_precipitation",
]

FINAL_OUTPUT = DATA_RAW / "algeria_climate_2015_present.nc"


def setup_cds_credentials():
    """Ensure ~/.cdsapirc exists, prompting the user if missing."""
    rc_path = Path.home() / ".cdsapirc"
    if rc_path.exists():
        return True

    print("=" * 70)
    print("COPERNICUS CDS CREDENTIALS NOT FOUND (~/.cdsapirc)")
    print("=" * 70)
    print("To download directly via Python, you need a free Copernicus CDS account.")
    print("1. Go to: https://cds.climate.copernicus.eu/")
    print("2. Log in and click your profile icon (top right) -> 'Your profile'")
    print("3. Scroll to 'Personal Access Token' and copy your API Key / Token.")
    print("=" * 70)

    try:
        token = input("Paste your Personal Access Token (API Key) here: ").strip()
    except (EOFError, KeyboardInterrupt):
        token = ""

    if not token:
        print("\nNo token entered. You can also download manually from:")
        print("https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means?tab=download")
        return False

    with open(rc_path, "w") as f:
        f.write("url: https://cds.climate.copernicus.eu/api\n")
        f.write(f"key: {token}\n")

    print(f"Saved credentials to {rc_path}\n")
    return True


def unpack_if_zip(file_path):
    """If file is a ZIP archive, unpack it and return the actual NetCDF path."""
    if not zipfile.is_zipfile(file_path):
        return file_path

    print(f"Unzipping {file_path.name}...")
    with zipfile.ZipFile(file_path, "r") as z:
        z.extractall(DATA_RAW)
        extracted = [DATA_RAW / name for name in z.namelist() if name.endswith(".nc")]

    if extracted:
        file_path.unlink(missing_ok=True)
        return extracted[0]
    return file_path


def download_chunk(client, years, months, output_path):
    """Download a chunk of ERA5-Land monthly means."""
    print(f"\nRequesting ERA5-Land data:")
    print(f"  Years:  {years}")
    print(f"  Months: {months}")
    print(f"  Area:   {AREA} (Algeria)")

    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": VARIABLES,
        "year": years,
        "month": months,
        "time": ["00:00"],
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    client.retrieve("reanalysis-era5-land-monthly-means", request, str(output_path))
    return unpack_if_zip(output_path)


def main():
    print("=" * 70)
    print("ERA5-LAND DATASET DOWNLOADER — ALGERIA (2015 TO PRESENT)")
    print("=" * 70)

    if not setup_cds_credentials():
        sys.exit(1)

    client = cdsapi.Client()

    # Step 1: 2015 - 2025 (all 12 months)
    years_main = [str(y) for y in range(2015, 2026)]
    months_full = [f"{m:02d}" for m in range(1, 13)]
    part1_file = DATA_RAW / "era5_2015_2025.nc"

    print("\n>>> DOWNLOADING PART 1: 2015 to 2025 (11 full years, 132 months)...")
    part1_nc = download_chunk(client, years_main, months_full, part1_file)

    # Step 2: 2026 (Available months up to current latency)
    # ERA5-Land monthly means has ~2-3 months latency.
    # Try up to month 07 (July) or 08 (August)
    part2_nc = None
    part2_file = DATA_RAW / "era5_2026.nc"

    for max_month in [8, 7, 6]:
        months_2026 = [f"{m:02d}" for m in range(1, max_month + 1)]
        print(f"\n>>> ATTEMPTING PART 2: 2026 (months 01 to {max_month:02d})...")
        try:
            part2_nc = download_chunk(client, ["2026"], months_2026, part2_file)
            print(f"Successfully downloaded 2026 data up to month {max_month:02d}!")
            break
        except Exception as e:
            print(f"  Month {max_month:02d} not yet available ({e}). Trying earlier months...")

    # Step 3: Merge parts into one unified dataset
    print("\n>>> MERGING DATASETS INTO UNIFIED FILE...")
    datasets = []
    for f in [part1_nc, part2_nc]:
        if f and Path(f).exists():
            d = xr.open_dataset(f)
            if "valid_time" in d.dims:
                d = d.rename({"valid_time": "time"})
            datasets.append(d)

    if len(datasets) == 1:
        merged = datasets[0]
    else:
        merged = xr.concat(datasets, dim="time").drop_duplicates(dim="time").sortby("time")

    # Save final unified dataset
    merged.to_netcdf(FINAL_OUTPUT)
    print(f"\nSUCCESS! Unified dataset saved to: {FINAL_OUTPUT}")
    print(f"Total time span: {str(merged.time.values[0])[:10]} to {str(merged.time.values[-1])[:10]} ({len(merged.time)} months)")
    print(f"Variables: {list(merged.data_vars.keys())}")


if __name__ == "__main__":
    main()
