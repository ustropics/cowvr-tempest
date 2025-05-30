from imports import *

# Set date for data retrieval
DATE = datetime.datetime(2024, 10, 7)

# NASA Earthdata URL for access
NASA_EARTHDATA_URL = "https://search.earthdata.nasa.gov/search"

# Set title for output
DATASET_TITLE = "COWVR STP-H8 Surface Wind Vector and Column-Integrated Atmospheric Water Measurements"

# Main directories
DATA_DIR = "./cowvr_data"
OUTPUT_DIR = "./images/swaths"
LOG_DIR = "./log"

# Central coordinates for map zoom (latitude, longitude in degrees)
CENTRAL_LAT = 22
CENTRAL_LON = -92.0
# Size of the map extent (20 x 20 degrees for instance)
MAP_EXTENT_DEGREES = 20.0

# Plotting options
OVERWRITE_PLOTS = True  # Overwrite existing plot files
# Options: 'wind' (wind speed only), 'clw' (cloud liquid water only), 'both' (wind and clw)
PLOT_MODE = 'both'

# Wind vector plotting parameters
WIND_VECTOR_STEP = 2  # Step size for subsampling wind vectors
WIND_VECTOR_SCALE = 1200
WIND_VECTOR_WIDTH = 0.001  # Width of the arrow shaft
WIND_VECTOR_HEADWIDTH = 4  # Width of the arrowhead
WIND_VECTOR_HEADLENGTH = 3  # Length of the arrowhead
WIND_VECTOR_HEADAXISLENGTH = 2.5  # Length of the arrowhead axis

# Colormap settings
WIND_CMAP = 'RdYlBu'  # Colormap for wind speed
CLW_CMAP = 'BuGn'  # Colormap for cloud liquid water
WIND_CMAP_REVERSED = True  # Whether to reverse the wind colormap
CLW_CMAP_REVERSED = False  # Whether to reverse the CLW colormap
WIND_VMIN = 0.0  # Minimum value for wind speed colormap (m/s)
WIND_VMAX = 25.0  # Maximum value for wind speed colormap (m/s)

# Map feature appearance
LAND_COLOR = 'black'  # Color for land (continent) fill
COASTLINE_COLOR = 'black'  # Color for coastlines
COASTLINE_LINEWIDTH = 0.5  # Line width for coastlines
BORDERS_COLOR = 'black'  # Color for borders
BORDERS_LINEWIDTH = 0.5  # Line width for borders

# Validate PLOT_MODE
VALID_PLOT_MODES = ['wind', 'clw', 'both']
if PLOT_MODE not in VALID_PLOT_MODES:
    raise ValueError(
        f"Invalid PLOT_MODE: {PLOT_MODE}. Must be one of {VALID_PLOT_MODES}")

# Validate colormaps
if WIND_CMAP not in plt.colormaps():
    raise ValueError(
        f"Invalid WIND_CMAP: {WIND_CMAP}. Must be a valid Matplotlib colormap.")
if CLW_CMAP not in plt.colormaps():
    raise ValueError(
        f"Invalid CLW_CMAP: {CLW_CMAP}. Must be a valid Matplotlib colormap.")
# Validate reversed colormaps
if WIND_CMAP_REVERSED and f"{WIND_CMAP}_r" not in plt.colormaps():
    raise ValueError(
        f"Invalid reversed WIND_CMAP: {WIND_CMAP}_r. Must be a valid Matplotlib colormap.")
if CLW_CMAP_REVERSED and f"{CLW_CMAP}_r" not in plt.colormaps():
    raise ValueError(
        f"Invalid reversed CLW_CMAP: {CLW_CMAP}_r. Must be a valid Matplotlib colormap.")

# Validate wind colormap range
if WIND_VMIN >= WIND_VMAX:
    raise ValueError(
        f"WIND_VMIN ({WIND_VMIN}) must be less than WIND_VMAX ({WIND_VMAX})")

# Custom filter to exclude INFO level messages


class ExcludeInfoFilter(logging.Filter):
    def filter(self, record):
        # Allow DEBUG, WARNING, ERROR, and CRITICAL; exclude INFO
        return record.levelno != logging.INFO

# Configure logging for main.log


def configure_logging():
    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    # Create logger
    logger = logging.getLogger()
    # Capture DEBUG and above (includes WARNING, ERROR)
    logger.setLevel(logging.DEBUG)

    # Create file handler for main.log
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "main.log"))
    file_handler.setLevel(logging.DEBUG)  # Capture DEBUG and above
    file_handler.addFilter(ExcludeInfoFilter())  # Exclude INFO messages

    # Create stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)  # Capture DEBUG and above

    # Set formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Clear existing handlers to avoid duplication
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Suppress Matplotlib font manager and PIL debug messages
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)
