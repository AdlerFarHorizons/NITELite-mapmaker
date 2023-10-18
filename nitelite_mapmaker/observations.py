'''This module handles data from observations, e.g. NITELite flights.
'''

from typing import Tuple, Union

import glob
import os

import cv2
import numpy as np
from osgeo import gdal
import pandas as pd
import pyproj
import scipy

import matplotlib.pyplot as plt


class Flight:

    def __init__(
        self,
        image_dir: str,
        img_log_fp: str,
        imu_log_fp: str,
        gps_log_fp: str,
        referenced_dir: str = None,
        img_shape: Tuple[int, int] = (1200, 1920),
        bit_precisions: dict[str, int] = {
            '.raw': 12,
            '.tiff': 16,
            '.tif': 16,
        },
        metadata_tz_offset_in_hr: float = 5.,
        cart_crs_code: str = 'EPSG:3857',
        latlon_crs_code: str = 'EPSG:4326',
    ):
        '''
        Args:
            image_dir: Directory containing raw image files.
            img_log_fp: Location of log containing image metadata.
            imu_log_fp: Location of log containing IMU metadata.
            gps_log_fp: Location of log containing GPS metadata.
            referenced_dir: Directory containing any existing
                georeferenced files.
            img_shape: Shape of image.
            bit_precisions: Bit types used for the data.
            metadata_tz_offset_in_hr: Difference between timezone of the GPS
                clock and the timezone of area the flight took place in.
            cart_crs_code: Cartesian coordinate reference system code.
                Defaults to the one used for Google maps.
            latlon_crs_code: Latitude/longituded coordinate reference system
                code. Defaults to WGS84, a common standard.
        '''

        self.image_dir = image_dir
        self.img_log_fp = img_log_fp
        self.imu_log_fp = imu_log_fp
        self.gps_log_fp = gps_log_fp
        self.referenced_dir = referenced_dir
        self.img_shape = img_shape
        self.bit_precisions = bit_precisions
        self.metadata_tz_offset_in_hr = metadata_tz_offset_in_hr

        # Establish CRS and conversions
        self.cart_crs_code = cart_crs_code
        self.latlon_crs_code = latlon_crs_code
        self.cart_crs = pyproj.CRS(cart_crs_code)
        self.latlon_crs = pyproj.CRS(latlon_crs_code)
        self.cart_to_latlon = pyproj.Transformer.from_crs(
            self.cart_crs,
            self.latlon_crs
        )
        self.latlon_to_cart = pyproj.Transformer.from_crs(
            self.latlon_crs,
            self.cart_crs
        )

        # TODO: This is commented out because raw images without metadata
        #   are not particularly useful.
        # Get list of image filepaths.
        # self.image_fps = glob.glob(os.path.join(self.image_dir, '*.raw'))

    def load_img_log(self, img_log_fp: str = None) -> pd.DataFrame:
        '''Load the images log.

        Args:
            img_log_fp: Location of the image log.
                Defaults to the one provided at init.
        '''

        if img_log_fp is None:
            img_log_fp = self.img_log_fp

        # Load data
        # Column names are known and input ad below.
        img_log_df = pd.read_csv(
            img_log_fp,
            names=[
                'odroid_timestamp',
                'obc_timestamp',
                'camera_num',
                'serial_num',
                'exposure_time',
                'sequence_ind',
                'internal_temp',
                'filename',
            ] + ['Unnamed: {}'.format(i + 1) for i in range(12)]
        )

        # Parse the timestamp
        # We use a combination of the odroid timestamp and the obc
        # timestamp because the odroid timestamp is missing the year but
        # the obc_timestamp has the wrong month.
        timestamp_split = img_log_df['obc_timestamp'].str.split('_')
        img_log_df['obc_timestamp'] = pd.to_datetime(
            timestamp_split.apply(lambda x: '_'.join(x[:2])),
            format=' %Y%m%d_%H%M%S'
        )
        img_log_df['timestamp'] = pd.to_datetime(
            img_log_df['obc_timestamp'].dt.year.astype(str)
            + ' '
            + img_log_df['odroid_timestamp']
        )
        img_log_df['timestamp_id'] = timestamp_split.apply(
            lambda x: x[-1]
        ).astype(int)
        # Correct for overflow
        img_log_df.loc[img_log_df['timestamp_id'] < 0, 'timestamp_id'] += 2**32

        # Drop unnamed columns
        img_log_df = img_log_df.drop(
            [column for column in img_log_df.columns if 'Unnamed' in column],
            axis='columns'
        )

        # Get filepaths
        def get_filepath(filename):
            return os.path.join(
                self.image_dir,
                filename,
            )
        img_log_df['obc_filename'] = img_log_df['filename'].copy()
        img_log_df['filename'] = img_log_df['obc_filename'].apply(
            os.path.basename
        )
        img_log_df['filepath'] = img_log_df['filename'].apply(get_filepath)
        # TODO: Currently flight only deals with one camera at a time.
        # Filepaths that are for other cameras are invalid.
        img_log_df['valid_filepath'] = img_log_df['filename'].isin(
            os.listdir(self.image_dir)
        )

        self.img_log_df = img_log_df
        return img_log_df

    def load_imu_log(self, imu_log_fp: str = None) -> pd.DataFrame:
        '''Load the IMU log.

        Args:
            imu_log_fp: Location of the IMU log.
                Defaults to the one provided at init.
        '''

        if imu_log_fp is None:
            imu_log_fp = self.imu_log_fp

        imu_log_df = pd.read_csv(imu_log_fp, low_memory=False)

        # Remove the extra header rows, and the nan rows
        imu_log_df.dropna(subset=['CurrTimestamp', ], inplace=True)
        imu_log_df.drop(
            imu_log_df.index[imu_log_df['CurrTimestamp'] == 'CurrTimestamp'],
            inplace=True
        )

        # Handle some situations where the pressure is negative
        ac_columns = ['TempC', 'pressure', 'mAltitude']
        imu_log_df.loc[imu_log_df['pressure'].astype(float) < 0] = np.nan

        # Convert to datetime and sort
        imu_log_df['CurrTimestamp'] = pd.to_datetime(
            imu_log_df['CurrTimestamp']
        )
        imu_log_df.sort_values('CurrTimestamp', inplace=True)

        # Assign dtypes
        skipped_cols = []
        for column in imu_log_df.columns:
            if column == 'CurrTimestamp':
                continue

            imu_log_df[column] = imu_log_df[column].astype(float)

        # Now also handle when the altitude or temperature are weird
        imu_log_df.loc[imu_log_df['TempC'] < -273, ac_columns] = np.nan
        imu_log_df.loc[imu_log_df['mAltitude'] < 0., ac_columns] = np.nan
        imu_log_df.loc[imu_log_df['mAltitude'] > 20000., ac_columns] = np.nan

        self.imu_log_df = imu_log_df
        return imu_log_df

    def load_gps_log(self, gps_log_fp: str = None) -> pd.DataFrame:
        '''Load the GPS log.

        Args:
            gps_log_fp: Location of the GPS log.
                Defaults to the one provided at init.
        '''

        if gps_log_fp is None:
            gps_log_fp = self.gps_log_fp

        gps_log_df = pd.read_csv(gps_log_fp)

        # Remove the extra header rows and the empty rows
        gps_log_df.dropna(subset=['CurrTimestamp', ], inplace=True)
        gps_log_df.drop(
            gps_log_df.index[gps_log_df['CurrTimestamp'] == 'CurrTimestamp'],
            inplace=True
        )

        # Remove the empty rows
        empty_timestamp = '00.00.0000 00:00:00000'
        gps_log_df.drop(
            gps_log_df.index[gps_log_df['CurrTimestamp'] == empty_timestamp],
            inplace=True
        )

        # Convert to datetime and sort
        gps_log_df['CurrTimestamp'] = pd.to_datetime(
            gps_log_df['CurrTimestamp']
        )
        gps_log_df.sort_values('CurrTimestamp', inplace=True)

        # Assign dtypes
        for column in gps_log_df.columns:
            if column in ['CurrTimestamp', 'GPSTime']:
                continue

            gps_log_df[column] = gps_log_df[column].astype(float)

        # Coordinates
        gps_log_df['sensor_x'], gps_log_df['sensor_y'] = \
            self.latlon_to_cart.transform(
                gps_log_df['GPSLat'],
                gps_log_df['GPSLong']
            )

        self.gps_log_df = gps_log_df
        return gps_log_df

    def combine_logs(
        self,
        img_log_df: pd.DataFrame = None,
        imu_log_df: pd.DataFrame = None,
        gps_log_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        '''Combine the different logs

        Args:
            img_log_df: DataFrame containing image metadata.
            imu_log_df: DataFrame containing IMU metadata.
            gps_log_df: DataFrame containing GPS metadata.

        Returns:
            log_df: Combined dataframe containing IMU and GPS metadata
                for each image.
        '''

        if img_log_df is None:
            img_log_df = self.img_log_df
        if imu_log_df is None:
            imu_log_df = self.imu_log_df
        if gps_log_df is None:
            gps_log_df = self.gps_log_df

        dfs_interped = [img_log_df, ]
        source_log_names = ['imu', 'gps']
        for i, df_to_include in enumerate([imu_log_df, gps_log_df]):

            source_log_name = source_log_names[i]
            df_to_include = df_to_include.copy()

            # This doesn't interpolate well unless converted
            if 'GPSTime' in df_to_include.columns:
                del df_to_include['GPSTime']

            # Get the timestamps in the right time zone
            df_to_include['CurrTimestamp_in_img_tz'] = (
                df_to_include['CurrTimestamp']
                - pd.Timedelta(self.metadata_tz_offset_in_hr, 'hr')
            )
            df_to_include = df_to_include.dropna(
                subset=['CurrTimestamp_in_img_tz']
            ).set_index('CurrTimestamp_in_img_tz').sort_index()
            df_to_include['timestamp_int_{}'.format(source_log_name)] = \
                df_to_include['CurrTimestamp'].astype(int)
            del df_to_include['CurrTimestamp']

            # Interpolate
            interp_fn = scipy.interpolate.interp1d(
                df_to_include.index.astype(int),
                df_to_include.values.transpose()
            )
            interped = interp_fn(img_log_df['timestamp'].astype(int))
            df_interped = pd.DataFrame(
                interped.transpose(),
                columns=df_to_include.columns
            )

            dfs_interped.append(df_interped)

        log_df = pd.concat(dfs_interped, axis='columns', )

        return log_df

    def prep_metadata(
        self,
        img_log_fp: str = None,
        imu_log_fp: str = None,
        gps_log_fp: str = None,
    ) -> pd.DataFrame:
        '''Load the image, IMU, and GPS metadata and correlate them.

        Args:
            img_log_fp: Location of the image log.
            imu_log_fp: Location of the IMU log.
            gps_log_fp: Location of the GPS log.
        '''

        if img_log_fp is None:
            img_log_fp = self.img_log_fp
        if imu_log_fp is None:
            imu_log_fp = self.imu_log_fp
        if gps_log_fp is None:
            gps_log_fp = self.gps_log_fp

        img_log_df = self.load_img_log()
        imu_log_df = self.load_imu_log()
        gps_log_df = self.load_gps_log()

        self.metadata = self.combine_logs(
            img_log_df,
            imu_log_df,
            gps_log_df,
        )

        if self.referenced_dir is not None:
            _ = self.get_manually_georeferenced_filepaths(self.referenced_dir)

        return self.metadata

    def get_manually_georeferenced_filepaths(
        self,
        manually_georeferenced_dir: str,
        camera_num: int = 1,
    ) -> pd.Series:
        '''Associate the image files with image files that have been
        manually georeferenced, assuming the timestamp is reliable.

        Args:
            manually_georeferenced_dir:
                Location of the manually-georeferenced files.
            camera_num:
                What camera to use.

        Returns:
            manually_referenced_fps:
                Filepath containing the manually-georeferenced file for each
                image, if it exists. Invalids are marked with pd.NA.
        '''

        # Get the timestamp IDs for the manually-referenced images
        man_fps = glob.glob(os.path.join(manually_georeferenced_dir, '*.tif'))
        man_fps = pd.Series(man_fps)
        pattern = r'(\d+)_{}.*.tif'.format(camera_num)
        man_ts_ids = man_fps.str.findall(pattern).str[0].astype('Int64')

        # Create a dataframe to work with
        man_df = pd.DataFrame(
            {
                'manually_referenced_fp': man_fps,
                'timestamp_id': man_ts_ids,
            }
        )
        man_df = man_df.drop_duplicates('timestamp_id')
        man_df.set_index('timestamp_id', inplace=True)

        # Correlate
        is_mr = self.metadata['timestamp_id'].isin(man_df.index)
        found_ts_ids = self.metadata['timestamp_id'].loc[is_mr].values
        found_man_fps = man_df.loc[found_ts_ids, 'manually_referenced_fp']
        found_man_fps = found_man_fps.values

        # Add a new column to the metadata
        self.metadata['manually_referenced_fp'] = pd.NA
        self.metadata.loc[is_mr, 'manually_referenced_fp'] = found_man_fps

        return self.metadata['manually_referenced_fp']

    def load_preexisting_metadata(self, fp: str = None) -> pd.DataFrame:

        if fp is None:
            fp = self.metadata_fp

        metadata = pd.read_csv(fp)

        # Handle any extra whitespace
        metadata.rename(columns=lambda x: x.strip(), inplace=True)

        self.metadata = metadata
        return metadata

    def get_observation(self, ind_or_fn: Union[int, str]):

        # Identify the matching ind
        if isinstance(ind_or_fn, str):
            fn_matches = self.metadata['filename'].str.find(ind_or_fn) != -1
            ind = self.metadata.index[fn_matches][0]
        # Use the passed-in ind
        else:
            ind = ind_or_fn

        return Observation(self, ind)

    def get_referenced_observation(self, ind_or_fn: Union[int, str]):
        '''
        TODO: Combine this with get_observation?
        '''

        # Identify the matching ind
        if isinstance(ind_or_fn, str):
            fps = self.metadata['manually_referenced_fp']
            fn_matches = fps.str.find(ind_or_fn) != -1
            ind = self.metadata.index[fn_matches][0]
        # Use the passed-in ind
        else:
            ind = ind_or_fn

        return ReferencedObservation(self, ind)


class Observation:

    def __init__(self, flight: Flight, ind: int, *args, **kwargs):
        '''Class for handling individual observations.
        Not intended for use independent of a flight.
        
        Args:
            flight: The flight this image was taken as part of.
            ind: The index of this observation
        Kwargs:
        Returns:
        '''
        self.flight = flight
        self.ind = ind
        self.img_shape = self.flight.img_shape
        self.bit_precisions = self.flight.bit_precisions

    @property
    def metadata(self) -> pd.Series:
        '''Easy reference for the specific metadata row corresponding to this
        observation.

        Returns:
            self.metadata: A row of the metadata DataFrame
        '''
        return self.flight.metadata.loc[self.ind]

    @property
    def img(self) -> np.ndarray[float]:
        '''Quick access, using default options for getting the image.
        For more-controlled access call get_img.
        '''
        if not hasattr(self, '_img'):
            self._img = self.get_img()
        return self._img

    @property
    def img_int(self) -> np.ndarray[int]:
        '''Quick access, using default options for getting the image.
        For more-controlled access call get_img.
        '''
        if not hasattr(self, '_img_int'):
            self._img_int = self.get_img_int()
        return self._img_int

    def get_img(
        self,
        conversion_method: str = 'opencv'
    ) -> np.ndarray[float]:
        '''Load one of the images.

        Args:
            conversion_method: How to convert the raw images. Options are
                'opencv': Let opencv do the work.
                'crude': Homebrewed method for verification.

        Returns:
            img: (n,m,3) array containing the image in RGB format
        '''

        fp = self.metadata['filepath']

        return load_rgb_img(
            fp,
            img_shape=self.img_shape,
            bit_precisions=self.bit_precisions,
            conversion_method=conversion_method,
        )

    def get_img_int(self, *args, **kwargs) -> np.ndarray[int]:

        img_int = cv2.normalize(
            self.img,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U
        )

        return img_int

    def show(self, ax=None):

        if ax is None:
            fig = plt.figure(figsize=np.array(self.img.shape[:2]) / 60.)
            ax = plt.gca()

        ax.imshow(self.img)


class ReferencedObservation(Observation):
    '''
    Assumes the file is saved as a GeoTIFF.
    '''

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # We don't inherit our image shape from the flight when referenced
        del self.img_shape

    @property
    def dataset(self):
        '''GDAL dataset.'''
        if not hasattr(self, '_dataset'):
            fp = self.metadata['manually_referenced_fp']
            self._dataset = gdal.Open(fp)
        return self._dataset

    def get_img(self, k_rot=0):

        fp = self.metadata['manually_referenced_fp']

        # Load the manually-referenced image
        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        img = img[:, :, ::-1] / 2**16  # Formatting

        return img

    def get_bounds(self, crs: pyproj.CRS) -> Tuple[np.ndarray, np.ndarray]:
        '''Get image bounds in a given coordinate system.
        
        Args:
            crs: Desired coordinate system.

        Returns:
            x_bounds: x_min, x_max of the image in the target coordinate system
            y_bounds: y_min, y_max of the image in the target coordinate system
        '''

        # Get the coordinates
        x_min, px_width, x_rot, y_max, y_rot, px_height = \
            self.dataset.GetGeoTransform()
        x_max = x_min + px_width * self.dataset.RasterXSize
        y_min = y_max + px_height * self.dataset.RasterYSize

        # Convert to desired crs
        file_crs = pyproj.CRS(self.dataset.GetProjection())
        file_to_desired = pyproj.Transformer.from_crs(
            file_crs,
            crs,
            always_xy=True
        )
        x_bounds, y_bounds = file_to_desired.transform(
            [x_min, x_max],
            [y_min, y_max]
        )

        return x_bounds, y_bounds

    def get_latlon_bounds(self):
        return self.get_bounds(self.flight.latlon_crs)

    def get_cart_bounds(self):
        return self.get_bounds(self.flight.cart_crs)

    def get_cart_coordinates(self):

        x_bounds, y_bounds = self.get_cart_bounds()

        xs = np.linspace(x_bounds[0], x_bounds[1], self.img.shape[1])
        ys = np.linspace(y_bounds[0], y_bounds[1], self.img.shape[0])

        return xs, ys

    def get_pixel_coordinates(self):

        pxs = np.arange(self.img.shape[1])
        pys = np.arange(self.img.shape[0])

        return pxs, pys

    def show(self, ax=None, cartesian=False):
        '''
        TODO: Make this more consistent with naming of other functions?
        '''

        # Use existing functionality
        if not cartesian:
            return super().show(ax=ax)

        if ax is None:
            fig = plt.figure(figsize=np.array(self.img.shape[:2]) / 60.)
            ax = plt.gca()

        xs, ys = self.get_cart_coordinates()

        ax.pcolormesh(
            xs,
            ys,
            self.img
        )

        ax.set_aspect('equal')

    def add_to_folium_map(
        self,
        m,
        label: str = 'referenced',
        include_corner_markers: bool = False
    ):
        '''Add to a folium map.
        
        Args:
            m (folium map): The map to add to.
        '''

        # Let's keep this as an optional import for now.
        import folium

        lon_bounds, lat_bounds = self.get_latlon_bounds()
        bounds = [
            [lat_bounds[0], lon_bounds[0]],
            [lat_bounds[1], lon_bounds[1]]
        ]

        folium.raster_layers.ImageOverlay(
            self.dataset.ReadAsArray().transpose(1, 2, 0),
            bounds=bounds,
            name=label,
        ).add_to(m)

        # Markers for the corners so we can understand how the image pixels
        # get flipped around
        if include_corner_markers:
            bounds_group = folium.FeatureGroup(name=f'{label} bounds')
            minmax_labels = ['min', 'max']
            for ii in range(2):
                for jj in range(2):
                    hover_text = (
                        f'(x_{minmax_labels[jj]}, '
                        f'y_{minmax_labels[ii]})'
                    )
                    folium.Marker(
                        [lat_bounds[ii], lon_bounds[jj]],
                        popup=hover_text,
                        icon=folium.Icon(
                            color='gray',
                            # TODO: Incorporate colors
                            # icon_color=palette.as_hex()[jj * 2 + ii]
                        ),
                    ).add_to(bounds_group)
            bounds_group.add_to(m)


def load_rgb_img(
    fp: str,
    img_shape: Tuple,
    bit_precisions: dict[int],
    conversion_method: str = 'opencv',
) -> np.ndarray[float]:
    '''General function for loading an rgb image from file,
    with minimal defaults. Not specific to a particular flight.
    '''

    ext = os.path.splitext(fp)[1]

    # Load and reshape raw image data.
    if ext == '.raw':
        raw_img = np.fromfile(fp, dtype=np.uint16)
        raw_img = raw_img.reshape(img_shape)

        # Get RGB image
        # Good method (does interpolation).
        if conversion_method == 'opencv':
            img = cv2.cvtColor(raw_img, cv2.COLOR_BAYER_BG2RGB)
        # Homebrew, understandable method.
        elif conversion_method == 'crude':
            # Since the raw files are bayer sampled
            # the red pixels are at "[0::2,0::2]" locations,
            # green are at "[0::2,1::2]" and "[1::2,0::2]",
            # and blue at "[1::2,1::2]"
            red = raw_img[0::2, 0::2]
            green1 = raw_img[0::2, 1::2]
            green2 = raw_img[1::2, 0::2]
            green_avg = 0.5 * (green1 + green2)
            blue = raw_img[1::2, 1::2]
            img = np.array([red, green_avg, blue, ]).transpose(1, 2, 0)
        else:
            raise ValueError('Invalid conversion method.')

    elif ext in ['.tiff', '.tif']:
        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)

        # CV2 defaults to BGR, but RGB is more standard for our purposes
        img = img[:, :, ::-1]

    else:
        raise IOError('Cannot read filetype {}'.format(ext))

    if img is None:
        return img

    # Scale RGB image to 0 to 1 for each channel.
    # The values are saved as integers, so we need to divide out.
    max_val = 2 ** bit_precisions[ext] - 1
    img = img / max_val

    # float32 is what OpenCV expects
    img = img.astype('float32')

    return img
