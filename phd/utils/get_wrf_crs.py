import cartopy.crs as ccrs

def get_wrf_crs(ds):
    """
    Extract the projection information from the WRF dataset and construct a Cartopy CRS.
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