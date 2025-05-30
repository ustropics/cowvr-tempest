from imports import *
from config import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import traceback
import matplotlib.axes


def plot_cowvr_data(wind_data, output_dir, date, file_path=None, is_combined=False):
    """
    Plot COWVR filtered wind speed, wind vectors, and/or cloud liquid water on a map.

    Args:
        wind_data (dict): Dictionary containing wind speed, u, v, lat, lon, valid_mask, clw
        output_dir (str): Directory to save the plot
        date (datetime): Date of the data
        file_path (str, optional): Path to the granule file for unique naming
        is_combined (bool): Whether this is a combined swath plot
    """
    logging.debug(
        f"Starting plot generation (combined={is_combined}, mode={PLOT_MODE})")

    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Use filtered data
        wind_speed = wind_data['wind_speed']
        u = wind_data['u']
        v = wind_data['v']
        lat = wind_data['lat']
        lon = wind_data['lon']
        valid_mask = wind_data['valid_mask']
        clw = wind_data['clw']

        # Validate data shapes
        expected_shape = (601, 1801)
        if wind_speed.shape != expected_shape or clw.shape != expected_shape or lon.shape != expected_shape or lat.shape != expected_shape:
            logging.error(
                f"Invalid data shapes: wind_speed {wind_speed.shape}, clw {clw.shape}, lon {lon.shape}, lat {lat.shape}")
            raise ValueError("Data shapes must match expected (601, 1801)")

        # Clip data to 30° x 30° extent to reduce plotting overhead
        half_extent = MAP_EXTENT_DEGREES / 2
        lon_min = CENTRAL_LON - half_extent
        lon_max = CENTRAL_LON + half_extent
        lat_min = CENTRAL_LAT - half_extent
        lat_max = CENTRAL_LAT + half_extent
        lon_min = ((lon_min + 180) % 360) - 180
        lon_max = ((lon_max + 180) % 360) - 180
        if lon_max <= lon_min:
            lon_max += 360

        mask = (lon >= lon_min) & (lon <= lon_max) & (
            lat >= lat_min) & (lat <= lat_max)
        if not np.any(mask):
            logging.warning(
                f"No data within map extent for {file_path or 'combined'}")
            return

        wind_speed = np.where(mask, wind_speed, np.nan)
        u = np.where(mask, u, np.nan)
        v = np.where(mask, v, np.nan)
        clw = np.where(mask, clw, np.nan)
        valid_mask = valid_mask & mask

        # Generate output filename
        filter_str = 'strict' if valid_mask.sum() < wind_speed.size else 'relaxed'
        if is_combined:
            file_id = 'combined'
        else:
            file_id = wind_data.get('file_id', 'unknown') if file_path is None else os.path.basename(
                file_path).split('.')[1]
        output_file = os.path.join(
            output_dir, f'cowvr_{PLOT_MODE}_{date.strftime("%Y%m%d")}_{file_id}_{filter_str}_filtered.png')

        # Check if image already exists and OVERWRITE_PLOTS is False
        if os.path.exists(output_file) and not OVERWRITE_PLOTS:
            logging.debug(f"Image already exists, skipping: {output_file}")
            return

        # Log data shapes and ranges
        logging.debug(
            f"Plotting data shapes: wind_speed {wind_speed.shape}, clw {clw.shape}, lon {lon.shape}, lat {lat.shape}")
        logging.debug(
            f"wind_speed range: min {np.nanmin(wind_speed):.2f}, max {np.nanmax(wind_speed):.2f}")
        logging.debug(
            f"clw range: min {np.nanmin(clw):.2f}, max {np.nanmax(clw):.2f}")
        logging.debug(
            f"lon range: min {np.nanmin(lon):.2f}, max {np.nanmax(lon):.2f}")
        logging.debug(
            f"lat range: min {np.nanmin(lat):.2f}, max {np.nanmax(lat):.2f}")

        # Check for valid data to plot
        wind_points = np.sum(~np.isnan(wind_speed) & valid_mask)
        clw_points = np.sum(~np.isnan(clw) & ~valid_mask)
        logging.debug(f"Valid wind points to plot: {wind_points}")
        logging.debug(f"Valid clw points to plot: {clw_points}")
        if (PLOT_MODE in ['wind', 'both'] and wind_points == 0) or (PLOT_MODE in ['clw', 'both'] and clw_points == 0 and PLOT_MODE != 'wind'):
            logging.warning(
                f"No valid data to plot for {output_file} (mode={PLOT_MODE})")
            return

        # Validate central coordinates
        if not (-90 <= CENTRAL_LAT <= 90):
            logging.error(
                f"Invalid central latitude: {CENTRAL_LAT}. Must be between -90 and 90 degrees.")
            raise ValueError(
                "Central latitude must be between -90 and 90 degrees.")
        if not (-180 <= CENTRAL_LON <= 180):
            logging.error(
                f"Invalid central longitude: {CENTRAL_LON}. Must be between -180 and 180 degrees.")
            raise ValueError(
                "Central longitude must be between -180 and 180 degrees.")

        # Create figure and axis with Cartopy projection
        try:
            fig = plt.figure(figsize=(15, 12))
            ax = plt.axes(projection=ccrs.PlateCarree())
        except Exception as e:
            logging.error(
                f"Failed to initialize Cartopy projection: {str(e)}\n{traceback.format_exc()}")
            raise

        # Add map features with configured appearance
        ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR)
        ax.add_feature(cfeature.COASTLINE, color=COASTLINE_COLOR,
                       linewidth=COASTLINE_LINEWIDTH)
        ax.add_feature(cfeature.BORDERS, color=BORDERS_COLOR,
                       linewidth=BORDERS_LINEWIDTH)
        gl = ax.gridlines(draw_labels={
                          'bottom': True, 'left': True, 'top': False, 'right': False}, linestyle='--')

        # Set map extent to the 30° x 30° region
        ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                      crs=ccrs.PlateCarree())

        # Plot data based on PLOT_MODE
        sc_wind = None
        sc_clw = None
        if PLOT_MODE in ['wind', 'both']:
            # Plot wind speed
            wind_cmap_name = f"{WIND_CMAP}_r" if WIND_CMAP_REVERSED else WIND_CMAP
            cmap_wind = plt.get_cmap(wind_cmap_name)
            sc_wind = ax.pcolormesh(lon, lat, wind_speed, cmap=cmap_wind,
                                    vmin=WIND_VMIN, vmax=WIND_VMAX,
                                    transform=ccrs.PlateCarree(), shading='auto')
            # Plot wind vectors
            step = WIND_VECTOR_STEP
            scale = WIND_VECTOR_SCALE
            lon_sub = lon[::step, ::step]
            lat_sub = lat[::step, ::step]
            u_sub = u[::step, ::step]
            v_sub = v[::step, ::step]
            logging.debug(
                f"Quiver plot: step={step}, scale={scale}, subsampled shape={u_sub.shape}")
            ax.quiver(lon_sub, lat_sub, u_sub, v_sub,
                      transform=ccrs.PlateCarree(), scale=scale, color='black',
                      width=WIND_VECTOR_WIDTH, headwidth=WIND_VECTOR_HEADWIDTH,
                      headlength=WIND_VECTOR_HEADLENGTH, headaxislength=WIND_VECTOR_HEADAXISLENGTH)

        if PLOT_MODE in ['clw', 'both']:
            # Plot clw where wind data is masked
            clw_masked = np.where(~valid_mask, clw, np.nan)
            clw_masked_points = np.sum(~np.isnan(clw_masked))
            logging.debug(f"CLW masked points to plot: {clw_masked_points}")
            clw_cmap_name = f"{CLW_CMAP}_r" if CLW_CMAP_REVERSED else CLW_CMAP
            cmap_clw = plt.get_cmap(clw_cmap_name)
            sc_clw = ax.pcolormesh(lon, lat, clw_masked, cmap=cmap_clw, vmin=0, vmax=1,
                                   transform=ccrs.PlateCarree(), shading='auto')

        # Create colorbars based on PLOT_MODE with adjusted spacing
        divider = make_axes_locatable(ax)
        if PLOT_MODE == 'wind':
            cax_wind = divider.append_axes(
                "right", size="2.5%", pad=0.2, axes_class=matplotlib.axes.Axes)
            cbar_wind = fig.colorbar(
                sc_wind, cax=cax_wind, label='Wind Speed (m/s)')
        elif PLOT_MODE == 'clw':
            cax_clw = divider.append_axes(
                "right", size="2.5%", pad=0.2, axes_class=matplotlib.axes.Axes)
            cbar_clw = fig.colorbar(
                sc_clw, cax=cax_clw, label='Cloud Liquid Water (mm)')
        else:  # both
            cax_wind = divider.append_axes(
                "right", size="2.5%", pad=0.2, axes_class=matplotlib.axes.Axes)
            cax_clw = divider.append_axes(
                "right", size="2.5%", pad=0.5, axes_class=matplotlib.axes.Axes)
            cbar_wind = fig.colorbar(
                sc_wind, cax=cax_wind, label='Wind Speed (m/s)')
            cbar_clw = fig.colorbar(
                sc_clw, cax=cax_clw, label='Cloud Liquid Water (mm)')

        # Set centered figure title
        title = f'COWVR {PLOT_MODE.capitalize()} - {date.strftime("%Y%m%d")} ({filter_str.capitalize()}, Filtered{", Combined" if is_combined else ""})'
        fig.suptitle(title, fontsize=16, y=0.98)

        # Optimize layout
        fig.tight_layout()

        # Save plot
        try:
            plt.savefig(output_file, bbox_inches='tight', dpi=150)
            logging.debug(f"Plot saved to {output_file}")
        except Exception as e:
            logging.error(
                f"Failed to save plot {output_file}: {str(e)}\n{traceback.format_exc()}")
            raise

        plt.close()

    except Exception as e:
        logging.error(
            f"Failed to generate plot {output_file}: {str(e)}\n{traceback.format_exc()}")
        raise
