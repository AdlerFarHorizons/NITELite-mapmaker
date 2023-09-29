'''This module handles data from observations, e.g. NITELite flights.
'''

from typing import Tuple
import warnings

import glob
import os

import cv2
import numpy as np
import pandas as pd
import scipy


class Flight:

    def __init__(
        self,
        image_dir: str,
        img_log_fp: str,
        imu_log_fp: str,
        gps_log_fp: str,
        img_shape: Tuple[int, int] = (1200, 1920),
        bit_precisions: dict[str, int] = {
            '.raw': 12,
            '.tiff': 16,
            '.tif': 16,
        },
        metadata_tz_offset_in_hr: float = 5.,
    ):
        '''
        Args:
            image_dir: Directory containing raw image files.
            img_log_fp: Location of log containing image metadata.
            imu_log_fp: Location of log containing IMU metadata.
            gps_log_fp: Location of log containing GPS metadata.
            img_shape: Shape of image.
            bit_precisions: Bit types used for the data.
        '''

        self.image_dir = image_dir
        self.img_log_fp = img_log_fp
        self.imu_log_fp = imu_log_fp
        self.gps_log_fp = gps_log_fp
        self.img_shape = img_shape
        self.bit_precisions = bit_precisions
        self.metadata_tz_offset_in_hr = metadata_tz_offset_in_hr

        # Get list of image filepaths.
        self.image_fps = glob.glob(os.path.join(self.image_dir, '*.raw'))

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

        metadata = self.combine_logs(
            img_log_df,
            imu_log_df,
            gps_log_df,
        )

        self.metadata = metadata
        return metadata

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

    def get_rgb_img(
        self,
        fp: str,
        conversion_method: str = 'opencv'
    ) -> np.ndarray[float]:
        '''Load one of the images.

        Args:
            fp: Image location.
            conversion_method: How to convert the raw images. Options are
                'opencv': Let opencv do the work.
                'crude': Homebrewed method for verification.

        Returns:
            img: (n,m,3) array containing the image in RGB format
        '''

        ext = os.path.splitext(fp)[1]

        # Load and reshape raw image data.
        if ext == '.raw':
            raw_img = np.fromfile(fp, dtype=np.uint16)
            raw_img = raw_img.reshape(self.img_shape)

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
        max_val = 2**self.bit_precisions[ext] - 1
        img = img / max_val

        # float32 is what OpenCV expects
        img = img.astype('float32')

        return img
