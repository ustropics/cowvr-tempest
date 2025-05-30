from imports import *
from config import *


def process_cowvr_data(file_path, strict_filters=True):
    """
    Process COWVR EDR HDF5 file to extract filtered wind speed, direction, and cloud liquid water.

    Args:
        file_path (str): Path to the HDF5 file
        strict_filters (bool): Whether to apply strict quality filters (rain, land)

    Returns:
        dict: Processed data including wind speed, u, v, lat, lon, valid_mask, file_id, and clw
    """
    # Configure data logging for data.log
    os.makedirs(LOG_DIR, exist_ok=True)
    data_logger = logging.getLogger("data_logger")
    data_logger.setLevel(logging.INFO)
    if not data_logger.handlers:
        data_handler = logging.FileHandler(os.path.join(LOG_DIR, "data.log"))
        data_handler.setFormatter(logging.Formatter("%(message)s"))
        data_logger.addHandler(data_handler)

    logging.debug(f"Processing HDF5 file: {file_path}")

    try:
        with h5py.File(file_path, 'r') as f:
            # Debug HDF5 structure
            def log_datasets(group, prefix=""):
                for key, item in group.items():
                    path = f"{prefix}/{key}"
                    if isinstance(item, h5py.Dataset):
                        logging.debug(
                            f"Dataset: {path}, Shape: {item.shape}, Dtype: {item.dtype}")
                    elif isinstance(item, h5py.Group):
                        log_datasets(item, path)

            logging.debug("HDF5 file structure:")
            log_datasets(f)

            # Check for required datasets
            required_datasets = [
                '/EnvDataRecords/wind_speed',
                '/EnvDataRecords/wind_dir',
                '/EnvDataRecords/wind_speed_flag',
                '/EnvDataRecords/wind_dir_flag',
                '/GriddedGeolocationAndFlags/grid_rain_flag',
                '/GriddedGeolocationAndFlags/grid_land_flag',
                '/GriddedGeolocationAndFlags/grid_lat',
                '/GriddedGeolocationAndFlags/grid_lon',
                '/EnvDataRecords/clw_aft'  # Added for cloud liquid water
            ]

            for dataset in required_datasets:
                if dataset not in f:
                    logging.error(f"Missing dataset: {dataset}")
                    raise KeyError(f"Dataset {dataset} not found in HDF5 file")

            # Access gridded data
            wind_speed = f['/EnvDataRecords/wind_speed'][:]  # m/s
            # degrees clockwise from North
            wind_dir = f['/EnvDataRecords/wind_dir'][:]
            wind_speed_flag = f['/EnvDataRecords/wind_speed_flag'][:]
            wind_dir_flag = f['/EnvDataRecords/wind_dir_flag'][:]
            rain_flag = f['/GriddedGeolocationAndFlags/grid_rain_flag'][:]
            land_flag = f['/GriddedGeolocationAndFlags/grid_land_flag'][:]
            lat = f['/GriddedGeolocationAndFlags/grid_lat'][:]  # degrees
            lon = f['/GriddedGeolocationAndFlags/grid_lon'][:]  # degrees
            clw = f['/EnvDataRecords/clw_aft'][:]  # cloud liquid water (mm)

            # Log shapes
            logging.debug(f"wind_speed shape: {wind_speed.shape}")
            logging.debug(f"wind_dir shape: {wind_dir.shape}")
            logging.debug(f"clw shape: {clw.shape}")
            logging.debug(f"lat shape: {lat.shape}")
            logging.debug(f"lon shape: {lon.shape}")

            # Check for fill values (e.g., -9999)
            fill_value = -9999
            wind_speed = np.where(wind_speed == fill_value, np.nan, wind_speed)
            wind_dir = np.where(wind_dir == fill_value, np.nan, wind_dir)
            clw = np.where(clw == fill_value, np.nan, clw)
            logging.debug(
                f"Fill values replaced: wind_speed NaN count {np.isnan(wind_speed).sum()}, "
                f"wind_dir NaN count {np.isnan(wind_dir).sum()}, "
                f"clw NaN count {np.isnan(clw).sum()}")

            # Create 2D coordinate grids
            if len(lon.shape) == 1 and len(lat.shape) == 1:
                lon_2d, lat_2d = np.meshgrid(lon, lat)
                logging.debug(
                    f"Created 2D grids: lon_2d shape {lon_2d.shape}, lat_2d shape {lat_2d.shape}")
            else:
                logging.error(
                    f"Expected 1D lon and lat, got shapes: lon {lon.shape}, lat {lat.shape}")
                raise ValueError("Longitude and latitude must be 1D arrays")

            # Check compatibility with wind_speed and clw
            if wind_speed.shape != lon_2d.shape or clw.shape != lon_2d.shape:
                logging.error(
                    f"Shape mismatch: wind_speed {wind_speed.shape}, clw {clw.shape}, lon_2d {lon_2d.shape}")
                raise ValueError(
                    "Wind speed and clw shapes must match coordinate grid")

            # Apply quality flags for wind data
            total_points = wind_speed.size
            if strict_filters:
                valid_mask = (wind_speed_flag == 0) & (wind_dir_flag == 0) & \
                    (rain_flag <= 1) & (land_flag == 0)  # Ocean, no/heavy rain
            else:
                relaxed_data = []
                valid_mask = (wind_speed_flag == 0) & (
                    wind_dir_flag == 0)  # Minimal filters

            # Log filter impact
            logging.debug(f"Quality filter impact (strict={strict_filters}):")
            logging.debug(f"Total grid points: {total_points}")
            logging.debug(
                f"Valid points after wind_speed_flag: {(wind_speed_flag == 0).sum()} ({(wind_speed_flag == 0).sum()/total_points*100:.1f}%)")
            logging.debug(
                f"Valid points after wind_dir_flag: {(wind_dir_flag == 0).sum()} ({(wind_dir_flag == 0).sum()/total_points*100:.1f}%)")
            logging.debug(
                f"Valid points after rain_flag <= 1: {(rain_flag <= 1).sum()} ({(rain_flag <= 1).sum()/total_points*100:.1f}%)")
            logging.debug(
                f"Valid points after land_flag == 0: {(land_flag == 0).sum()} ({(land_flag == 0).sum()/total_points*100:.1f}%)")
            logging.debug(
                f"Final valid wind points: {valid_mask.sum()} ({valid_mask.sum()/total_points*100:.1f}%)")
            logging.debug(
                f"Valid clw points (where wind is masked): {(~valid_mask & ~np.isnan(clw)).sum()}")

            wind_speed = np.where(valid_mask, wind_speed, np.nan)
            wind_dir = np.where(valid_mask, wind_dir, np.nan)

            # Log data ranges
            logging.debug(
                f"Filtered wind_speed range: min {np.nanmin(wind_speed):.2f}, max {np.nanmax(wind_speed):.2f}")
            logging.debug(
                f"clw range: min {np.nanmin(clw):.2f}, max {np.nanmax(clw):.2f}")
            logging.debug(
                f"lon_2d range: min {np.min(lon_2d):.2f}, max {np.max(lon_2d):.2f}")
            logging.debug(
                f"lat_2d range: min {np.min(lat_2d):.2f}, max {np.max(lat_2d):.2f}")
            logging.debug(f"Valid data points: {np.sum(valid_mask)}")

            # Log valid data points to data.log
            data_logger.info("longitude,latitude,wind_speed,wind_direction")
            for i in range(lon_2d.shape[0]):
                for j in range(lon_2d.shape[1]):
                    if valid_mask[i, j]:
                        data_logger.info(
                            f"{lon_2d[i, j]:.2f},{lat_2d[i, j]:.2f},{wind_speed[i, j]:.2f},{wind_dir[i, j]:.2f}")

            # Convert wind direction to u, v components for plotting
            wind_dir_rad = np.deg2rad(wind_dir)
            u = wind_speed * np.sin(wind_dir_rad)  # Wind blowing toward
            v = wind_speed * np.cos(wind_dir_rad)

            # Log time coverage (if available)
            if '/GriddedGeolocationAndFlags/grid_time_tai93_aft' in f:
                time_aft = f['/GriddedGeolocationAndFlags/grid_time_tai93_aft'][:]
                logging.debug(
                    f"Time range (TAI93 aft): min {np.nanmin(time_aft):.2f}, max {np.nanmax(time_aft):.2f}")

            # Extract file_id for naming
            file_id = os.path.basename(file_path).split('.')[1]

            logging.debug("Data processing completed")

            return {
                'wind_speed': wind_speed,
                'u': u,
                'v': v,
                'lat': lat_2d,
                'lon': lon_2d,
                'valid_mask': valid_mask,
                'file_id': file_id,
                'clw': clw  # Added clw data
            }
    except Exception as e:
        logging.error(f"Failed to process HDF5 file {file_path}: {str(e)}")
        raise
