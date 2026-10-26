

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
import rioxarray  # noqa: F401 — registers the .rio accessor on xarray objects

# ─── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_DIR / "data" / "raw"
DATA_PROCESSED = PROJECT_DIR / "data" / "processed"
SHP_DIR = PROJECT_DIR / "data" / "dz_shp"
FIGURES_DIR = PROJECT_DIR / "outputs" / "figures"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_shapefile():
    """Load the 58-wilaya shapefile and return both individual and dissolved GeoDataFrames."""
    shp_path = SHP_DIR / "dz.shp"
    if not shp_path.exists():
        raise FileNotFoundError(
            f"Algeria shapefile not found at {shp_path}. "
            "Make sure data/dz_shp/ contains dz.shp, dz.shx, dz.dbf, dz.prj."
        )

    wilayas = gpd.read_file(shp_path)
    print(f"Loaded shapefile: {len(wilayas)} wilayas")
    print(f"  CRS: {wilayas.crs}")
    print(f"  Columns: {list(wilayas.columns)}")
    print(f"  Bounds: {wilayas.total_bounds}")  # [minx, miny, maxx, maxy]

    # Dissolved country boundary (union of all wilayas)
    algeria_boundary = wilayas.dissolve()
    return wilayas, algeria_boundary


def load_climate_data(use_raw=True):
    """Load the ERA5-Land NetCDF dataset. Defaults to raw (full-resolution) unaggregated data."""
    if use_raw:
        nc_path = DATA_RAW / "algeria_climate_2015_present.nc"
        label = "raw (full-resolution ~0.1°)"
    else:
        nc_path = DATA_PROCESSED / "algeria_climate_aggregated.nc"
        if not nc_path.exists():
            nc_path = DATA_RAW / "algeria_climate_2015_present.nc"
            label = "raw (aggregated file not found, using raw)"
        else:
            label = "aggregated (coarsened ~0.4°)"

    print(f"\nLoading {label} dataset: {nc_path.name}")
    ds = xr.open_dataset(nc_path)

    # Normalize dimension names
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    print(f"  Grid: {len(ds.latitude)} lat × {len(ds.longitude)} lon")
    print(f"  Time steps: {len(ds.time)}")
    print(f"  Variables: {list(ds.data_vars)}")
    return ds


def clip_with_shapefile(ds, algeria_boundary):
    """
    Mask the xarray dataset using the dissolved Algeria boundary.

    Uses rioxarray to set the CRS and clip. Grid points outside the polygon
    are set to NaN.
    """
    print("\nClipping dataset to Algeria boundary...")

    # Ensure spatial dims are named correctly for rioxarray
    if "latitude" in ds.dims:
        ds = ds.rename({"latitude": "y", "longitude": "x"})

    # Set CRS (WGS84) — must match the shapefile
    ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
    ds = ds.rio.write_crs("EPSG:4326")

    # Clip to the dissolved country boundary
    ds_clipped = ds.rio.clip(algeria_boundary.geometry, algeria_boundary.crs, drop=False)

    # Rename back to latitude/longitude for consistency with the rest of the pipeline
    ds_clipped = ds_clipped.rename({"y": "latitude", "x": "longitude"})

    # Count valid vs masked points at a single time step
    sample_var = list(ds_clipped.data_vars)[0]
    sample = ds_clipped[sample_var].isel(time=0)
    n_total = sample.size
    n_valid = int(sample.notnull().sum())
    n_masked = n_total - n_valid

    print(f"  Total grid points:  {n_total:,}")
    print(f"  Inside Algeria:     {n_valid:,}")
    print(f"  Masked (outside):   {n_masked:,}")
    print(f"  Coverage:           {n_valid / n_total * 100:.1f}%")

    return ds_clipped


def tag_wilayas(ds_clipped, wilayas):
    """
    For each (lat, lon) grid point, determine which wilaya it falls into.

    Returns a 2D DataArray with wilaya names on the (latitude, longitude) grid.
    """
    print("\nTagging grid points with wilaya names...")

    lats = ds_clipped.latitude.values
    lons = ds_clipped.longitude.values

    # Build a grid of Points
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points = gpd.GeoDataFrame(
        {
            "lat_idx": np.repeat(np.arange(len(lats)), len(lons)),
            "lon_idx": np.tile(np.arange(len(lons)), len(lats)),
        },
        geometry=[Point(lo, la) for lo, la in zip(lon_grid.ravel(), lat_grid.ravel())],
        crs="EPSG:4326",
    )

    # Spatial join: assign each grid point to a wilaya
    joined = gpd.sjoin(points, wilayas[["name", "geometry"]], how="left", predicate="within")

    # Build the 2D wilaya array
    wilaya_grid = np.full((len(lats), len(lons)), "", dtype=object)
    for _, row in joined.iterrows():
        name = row.get("name", "")
        if pd.isna(name):
            name = ""
        wilaya_grid[int(row.lat_idx), int(row.lon_idx)] = name

    # Convert to xarray DataArray
    wilaya_da = xr.DataArray(
        wilaya_grid,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
        name="wilaya",
    )

    n_tagged = int((wilaya_da != "").sum())
    unique_wilayas = sorted([w for w in set(wilaya_grid.ravel()) if w and isinstance(w, str)])
    print(f"  Grid points tagged:      {n_tagged}")
    print(f"  Unique wilayas covered:  {len(unique_wilayas)}")

    return wilaya_da


def save_outputs(ds_clipped, wilaya_da):
    """Save the clipped NetCDF and a flat DataFrame with wilaya tags."""
    # 1. Save clipped NetCDF
    nc_out = DATA_PROCESSED / "algeria_climate_clipped.nc"
    # Drop the spatial_ref coordinate added by rioxarray (not needed downstream)
    if "spatial_ref" in ds_clipped.coords:
        ds_clipped = ds_clipped.drop_vars("spatial_ref")
    if nc_out.exists():
        nc_out.unlink()
    ds_clipped.to_netcdf(nc_out)
    print(f"\nSaved clipped NetCDF:  {nc_out}")

    # 2. Save wilaya grid as NetCDF
    wilaya_nc = DATA_PROCESSED / "wilaya_grid.nc"
    if wilaya_nc.exists():
        wilaya_nc.unlink()
    wilaya_da.to_dataset(name="wilaya").to_netcdf(wilaya_nc)
    print(f"Saved wilaya grid:     {wilaya_nc}")

    # 3. Flatten to DataFrame (for supervised/unsupervised pipelines)
    print("\nBuilding flat DataFrame with wilaya labels...")
    df = ds_clipped.to_dataframe().reset_index()

    # Merge wilaya names
    wilaya_df = wilaya_da.to_dataframe().reset_index()
    df = df.merge(wilaya_df, on=["latitude", "longitude"], how="left")

    # Drop rows that are fully NaN (masked ocean / outside Algeria)
    data_vars = [v for v in ds_clipped.data_vars]
    df = df.dropna(subset=data_vars, how="all")

    csv_out = DATA_PROCESSED / "algeria_climate_clipped.csv"
    if csv_out.exists():
        csv_out.unlink()
    df.to_csv(csv_out, index=False)
    print(f"Saved flat CSV:        {csv_out}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    return df


def plot_clipping_result(ds_clipped, algeria_boundary, wilayas):
    """Generate a quick visual check of the clipping result."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Left: clipped temperature map
        sample_var = "t2m" if "t2m" in ds_clipped.data_vars else list(ds_clipped.data_vars)[0]
        ds_clipped[sample_var].isel(time=0).plot(
            ax=axes[0], cmap="RdYlBu_r", add_colorbar=True
        )
        wilayas.boundary.plot(ax=axes[0], color="black", linewidth=0.5)
        axes[0].set_title("Clipped Climate Data + Wilaya Borders", fontsize=13)
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")

        # Right: wilaya boundaries colored
        wilayas.plot(ax=axes[1], column="name", cmap="tab20", edgecolor="black", linewidth=0.3, legend=False)
        axes[1].set_title("58 Wilayas of Algeria", fontsize=13)
        axes[1].set_xlabel("Longitude")
        axes[1].set_ylabel("Latitude")

        plt.tight_layout()
        fig_path = FIGURES_DIR / "clipping_result.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved figure: {fig_path}")

    except Exception as e:
        print(f"\nWarning: Could not generate plot ({e})")


def main():
    parser = argparse.ArgumentParser(description="Clip ERA5-Land data to Algeria borders")
    parser.add_argument("--aggregated", action="store_true", help="Clip the aggregated dataset instead of raw")
    parser.add_argument("--raw", action="store_true", default=True, help="Use the raw (full-resolution) dataset (default)")
    args = parser.parse_args()

    use_raw = not args.aggregated

    print("=" * 70)
    print("CLIP ERA5-LAND DATA TO ALGERIA SHAPEFILE")
    print("=" * 70)

    # Load data
    wilayas, algeria_boundary = load_shapefile()
    ds = load_climate_data(use_raw=use_raw)

    # Clip
    ds_clipped = clip_with_shapefile(ds, algeria_boundary)

    # Tag wilayas
    wilaya_da = tag_wilayas(ds_clipped, wilayas)

    # Save
    df = save_outputs(ds_clipped, wilaya_da)

    # Plot
    plot_clipping_result(ds_clipped, algeria_boundary, wilayas)

    print("\n" + "=" * 70)
    print("DONE! Clipped data is ready for the pipeline.")
    print("=" * 70)


if __name__ == "__main__":
    main()
