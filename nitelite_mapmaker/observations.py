'''This module handles data from observations, e.g. NITELite flights.
'''

from typing import Tuple

import glob
import os

import cv2
import numpy as np
import pandas as pd


class Flight:

    def __init__(
        self,
        image_dir: str,
        img_log_fp: str,
        imu_log_fp: str,
        gps_log_fp: str,
        img_shape: Tuple[int, int] = (1200, 1920),
        max_val: int = 4091,
    ):
        '''
        Args:
            image_dir: Directory containing raw image files.
            img_log_fp: Location of log containing image metadata.
            imu_log_fp: Location of log containing IMU metadata.
            gps_log_fp: Location of log containing GPS metadata.
            img_shape: Shape of image.
            max_val: Maximum value of raw image data. This corresponds to
                the integer type. Defaults to 4091 = 2**12 - 1 (i.e. 12-bit).
                It's not clear why it's a 12 bit integer, but it is.
        '''

        self.image_dir = image_dir
        self.img_log_fp = img_log_fp
        self.imu_log_fp = imu_log_fp
        self.gps_log_fp = gps_log_fp
        self.img_shape = img_shape
        self.max_val = max_val

        # Get list of image filepaths.
        self.image_fps = glob.glob(os.path.join(self.image_dir, '*.raw'))

    def load_img_log(self, img_log_fp: str = None):
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

        # Drop unnamed columns
        img_log_df = img_log_df.drop(
            [column for column in img_log_df.columns if 'Unnamed' in column],
            axis='columns'
        )

        self.img_log_df = img_log_df
        return img_log_df

    def load_imu_log(self, imu_log_fp: str = None):

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

        # We'll skip this step for now because we filter on the date later
        # DEPRECATED
        '''
        # Convert to datetime, toss out IMU recordings not associated with the 5-13 flight.
        imu_log_df['CurrTimestamp'] = pd.to_datetime(imu_log_df['CurrTimestamp'])
        imu_log_df.drop(
            imu_log_df.index[imu_log_df['CurrTimestamp'] < pd.to_datetime('2022-5-13 20')],
            inplace=True,
        )
        '''

        # Sort by datetime
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

    def prep_metadata(self):
        '''Load the image, IMU, and GPS metadata and correlate them.
        '''

        img_log_metadata = self.load_img_log_metadata()
        imu_metadata = self.load_imu_metadata()
        gps_metadata = self.load_gps_metadata()

        self.metadata = self.combine_metadata(
            img_log_metadata,
            imu_metadata,
            gps_metadata,
        )

    def load_preexisting_metadata(self, fp: str = None):

        if fp is None:
            fp = self.metadata_fp

        metadata = pd.read_csv(fp)

        # Handle any extra whitespace
        metadata.rename(columns=lambda x: x.strip(), inplace=True)

        self.metadata = metadata
        return metadata

    def get_rgb_img(self, fp: str, conversion_method: str = 'opencv'):

        # Load and reshape raw image data.
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

        # Scale RGB image to 0 to 1 for each channel.
        img = img / self.max_val

        return img
