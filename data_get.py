from imports import *
from config import *


def fetch_cowvr_data(date, data_dir, earthdata_url, dataset_title):
    """
    Fetch all COWVR EDR data granules for a given date from NASA Earth Data Portal using earthaccess.

    Args:
        date (datetime): Date for which to fetch data
        data_dir (str): Directory to store downloaded data
        earthdata_url (str): NASA Earth Data Portal URL (unused, kept for compatibility)
        dataset_title (str): Dataset title for COWVR data (unused, kept for compatibility)

    Returns:
        list: Paths to downloaded or existing HDF5 files
    """
    # Read credentials from credentials.txt
    credentials_file = "credentials.txt"
    if not os.path.exists(credentials_file):
        logging.error(f"Credentials file {credentials_file} not found")
        raise FileNotFoundError(
            f"Please create {credentials_file} in project root with:\n"
            "username=<your_username>\n"
            "password=<your_password>"
        )

    try:
        with open(credentials_file, "r") as f:
            lines = f.readlines()
            credentials = {}
            for line in lines:
                key, value = line.strip().split("=", 1)
                credentials[key] = value
            username = credentials.get("username")
            password = credentials.get("password")
            if not username or not password:
                raise ValueError(
                    "Missing username or password in credentials.txt")
    except Exception as e:
        logging.error(f"Failed to read credentials: {str(e)}")
        raise

    # Set environment variables for earthaccess
    os.environ["EARTHDATA_USERNAME"] = username
    os.environ["EARTHDATA_PASSWORD"] = password

    # Authenticate with Earthdata
    try:
        auth = earthaccess.login(strategy="environment")
        if not auth.authenticated:
            logging.error("Authentication failed with provided credentials")
            raise Exception(
                "Authentication failed. Verify credentials in credentials.txt.")
    except Exception as e:
        logging.error(f"Failed to authenticate with Earthdata: {str(e)}")
        raise

    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Check for existing files
    date_str = date.strftime("%Y%m%d")
    file_pattern = os.path.join(data_dir, f"COWVR_EDR.*.{date_str}T*.h5")
    existing_files = glob.glob(file_pattern)

    if existing_files:
        logging.warning(
            f"Found {len(existing_files)} existing files matching {file_pattern}: {existing_files}")
        logging.warning("Using existing files. Skipping download.")
        return existing_files

    # Construct date range for the specified day
    start_time = date.strftime("%Y-%m-%d 00:00:00")
    end_time = (date + datetime.timedelta(days=1)
                ).strftime("%Y-%m-%d 00:00:00")

    # Search for COWVR EDR data
    logging.debug(f"Searching for COWVR data for {start_time} to {end_time}")
    results = earthaccess.search_data(
        short_name="COWVR_STPH8_L2_EDR_V10.0",
        temporal=(start_time, end_time)
    )

    # Normalize results check and provide more informative logging
    results_count = 0 if not results else len(results)
    if results_count == 0:
        logging.error(
            f"No data found for {start_time} to {end_time} (short_name=COWVR_STPH8_L2_EDR_V10.0)."
        )
        raise ValueError(
            f"No data found for {start_time} to {end_time} (short_name=COWVR_STPH8_L2_EDR_V10.0)"
        )

    # Download the data with error handling
    logging.debug(
        f"Attempting to download {len(results)} COWVR data granules to {data_dir}")
    downloaded_files = []
    for granule in results:
        try:
            files = earthaccess.download([granule], data_dir)
            if files:
                downloaded_files.extend(files)
                logging.debug(f"Successfully downloaded: {files}")
        except HTTPError as e:
            logging.error(
                f"Failed to download granule {granule.data_links()[0]}: {str(e)}")
            continue
        except Exception as e:
            logging.error(
                f"Unexpected error downloading granule {granule.data_links()[0]}: {str(e)}")
            continue

    if not downloaded_files:
        logging.error(
            f"No files downloaded for {start_time} to {end_time} after attempting {results_count} search results"
        )
        raise ValueError(
            f"Failed to download any data for {start_time} to {end_time} (searched {results_count} granules)"
        )

    logging.debug(
        f"Downloaded {len(downloaded_files)} files: {downloaded_files}")
    return downloaded_files
