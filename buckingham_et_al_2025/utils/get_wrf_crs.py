import cartopy.crs as ccrs

def get_wrf_crs(ds):
    """
    Extract the projection information from a WRF dataset and construct a corresponding Cartopy CRS.

    This function reads the 'MAP_PROJ' attribute from the input dataset to determine the type of map 
    projection used by the WRF model. Depending on the value of 'MAP_PROJ', it extracts additional 
    projection parameters (such as TRUELAT1, TRUELAT2, STAND_LON, and MOAD_CEN_LAT) and constructs the 
    appropriate Cartopy CRS object. Supported projections include:
      - MAP_PROJ == 1: Lambert Conformal Conic projection.
      - MAP_PROJ == 2: Stereographic projection.
      - MAP_PROJ == 3: Mercator projection.
      - MAP_PROJ == 6: Plate Carree projection (geographic).
    
    Parameters:
      ds: A dataset (e.g., an xarray.Dataset) containing WRF projection attributes such as
          'MAP_PROJ', 'TRUELAT1', 'TRUELAT2', 'STAND_LON', and optionally 'MOAD_CEN_LAT'.
    
    Returns:
      A Cartopy CRS object representing the WRF projection.
    
    Raises:
      ValueError: If the MAP_PROJ attribute is not one of the supported projection types.
    """
    map_proj = int(ds.MAP_PROJ)

    if map_proj == 1:
        truelat1 = float(ds.TRUELAT1)
        truelat2 = float(ds.TRUELAT2)
        stand_lon = float(ds.STAND_LON)
        moad_cen_lat = float(ds.MOAD_CEN_LAT)

        wrf_crs = ccrs.LambertConformal(
            central_longitude=stand_lon,
            central_latitude=moad_cen_lat,
            standard_parallels=(truelat1, truelat2)
        )
    elif map_proj == 2:
        truelat1 = float(ds.TRUELAT1)
        stand_lon = float(ds.STAND_LON)

        wrf_crs = ccrs.Stereographic(
            central_longitude=stand_lon,
            central_latitude=90 if truelat1 > 0 else -90,
            true_scale_latitude=truelat1
        )
    elif map_proj == 3:
        truelat1 = float(ds.TRUELAT1)
        stand_lon = float(ds.STAND_LON)

        wrf_crs = ccrs.Mercator(
            central_longitude=stand_lon,
            latitude_true_scale=truelat1
        )
    elif map_proj == 6:
        wrf_crs = ccrs.PlateCarree()
    else:
        raise ValueError(f"Unsupported MAP_PROJ value: {map_proj}")

    return wrf_crs