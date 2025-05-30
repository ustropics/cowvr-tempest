from imports import *
import shutil
import multiprocessing as mp
from config import *
from data_get import fetch_cowvr_data
from data_prc import process_cowvr_data
from data_plt import plot_cowvr_data

# Set Matplotlib to use non-interactive Agg backend
plt.switch_backend('Agg')


def combine_swath_data(all_wind_data, strict_filters):
    """
    Combine filtered wind data and clw from multiple granules.

    Args:
        all_wind_data (list): List of dictionaries containing wind and clw data
        strict_filters (bool): Whether strict filters were applied

    Returns:
        dict: Combined data dictionary
    """
    if not all_wind_data:
        logging.warning("No data to combine")
        return None

    # Initialize combined arrays with NaN
    sample_data = all_wind_data[0]
    shape = sample_data['wind_speed'].shape
    combined_data = {
        'wind_speed': np.full(shape, np.nan),
        'u': np.full(shape, np.nan),
        'v': np.full(shape, np.nan),
        'clw': np.full(shape, np.nan),
        'lat': sample_data['lat'],
        'lon': sample_data['lon'],
        'valid_mask': np.zeros(shape, dtype=bool)
    }

    # Combine data
    valid_wind_points = 0
    valid_clw_points = 0
    for data in all_wind_data:
        if data['valid_mask'].sum() > 0:
            # Wind data: use points where wind is valid and non-NaN
            wind_mask = ~np.isnan(data['wind_speed']) & data['valid_mask']
            if wind_mask.shape != shape:
                logging.error(
                    f"Shape mismatch in granule wind_mask: {wind_mask.shape}, expected {shape}")
                continue
            combined_data['wind_speed'][wind_mask] = data['wind_speed'][wind_mask]
            combined_data['u'][wind_mask] = data['u'][wind_mask]
            combined_data['v'][wind_mask] = data['v'][wind_mask]
            combined_data['valid_mask'][wind_mask] = True
            valid_wind_points += np.sum(wind_mask)
            logging.debug(
                f"Added {np.sum(wind_mask)} valid wind points from granule")

            # CLW data: use points where clw is non-NaN
            clw_mask = ~np.isnan(data['clw'])
            if clw_mask.shape != shape:
                logging.error(
                    f"Shape mismatch in granule clw_mask: {clw_mask.shape}, expected {shape}")
                continue
            combined_data['clw'][clw_mask] = data['clw'][clw_mask]
            valid_clw_points += np.sum(clw_mask)
            logging.debug(
                f"Added {np.sum(clw_mask)} valid clw points from granule")

    logging.debug(
        f"Combined {len(all_wind_data)} swaths with {valid_wind_points} valid wind points for strict_filters={strict_filters}")
    logging.debug(
        f"Total valid clw points in combined data: {np.sum(~np.isnan(combined_data['clw']))}")
    logging.debug(
        f"Valid clw points to plot (where wind is masked): {np.sum(~combined_data['valid_mask'] & ~np.isnan(combined_data['clw']))}")
    if valid_wind_points == 0 and valid_clw_points == 0:
        logging.warning("No valid wind or clw data points in combined swaths")
        return None

    return combined_data


def process_and_plot_granule(args):
    """
    Process and plot a single granule with given filter.

    Args:
        args (tuple): (file_path, strict, output_dir, date)
    """
    file_path, strict, output_dir, date = args
    try:
        logging.debug(
            f"Processing granule: {file_path} with strict_filters={strict}")
        wind_data = process_cowvr_data(file_path, strict_filters=strict)
        if wind_data is None:
            logging.debug(f"Skipping granule {file_path}: no data in extent")
            return None, strict
        logging.debug(
            f"Plotting filtered COWVR data with strict_filters={strict}")
        plot_cowvr_data(wind_data, output_dir, date, file_path=file_path)
        return wind_data, strict
    except Exception as e:
        logging.error(
            f"Failed to process or plot granule {file_path} with strict_filters={strict}: {str(e)}")
        return None, strict


def main():
    # Configure logging
    configure_logging()
    logging.debug(
        f"Starting COWVR data processing pipeline (PLOT_MODE={PLOT_MODE})")

    try:
        # Clear output directory if OVERWRITE_PLOTS is True
        if OVERWRITE_PLOTS:
            if os.path.exists(OUTPUT_DIR):
                try:
                    shutil.rmtree(OUTPUT_DIR)
                    logging.debug(f"Cleared output directory: {OUTPUT_DIR}")
                except PermissionError as e:
                    logging.error(
                        f"Permission denied when clearing {OUTPUT_DIR}: {str(e)}")
                    raise
            os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Step 1: Fetch data
        logging.debug("Fetching COWVR data")
        file_paths = fetch_cowvr_data(
            DATE, DATA_DIR, NASA_EARTHDATA_URL, DATASET_TITLE)

        if not file_paths:
            logging.warning("No files available to process")
            return

        logging.debug(f"Processing {len(file_paths)} granules")

        # Step 2: Process and plot granules in parallel
        strict_data = []
        relaxed_data = []
        num_processes = min(mp.cpu_count(), 8)  # Use up to 8 cores
        logging.debug(
            f"Using {num_processes} processes for parallel processing")

        tasks = [(file_path, strict, OUTPUT_DIR, DATE)
                 for file_path in file_paths
                 for strict in [True, False]]

        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(process_and_plot_granule, tasks)

        # Collect results
        for wind_data, strict in results:
            if wind_data is not None:
                if strict:
                    strict_data.append(wind_data)
                else:
                    relaxed_data.append(wind_data)

        # Step 3: Plot combined swaths
        logging.debug("Generating combined swath plots")
        for strict, data_list in [(True, strict_data), (False, relaxed_data)]:
            logging.debug(
                f"Combining {len(data_list)} swaths for strict_filters={strict}")
            combined_data = combine_swath_data(
                data_list, strict_filters=strict)
            if combined_data:
                logging.debug(
                    f"Plotting combined filtered data with strict_filters={strict}")
                plot_cowvr_data(combined_data, OUTPUT_DIR,
                                DATE, is_combined=True)

        logging.debug("Pipeline completed successfully")
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
