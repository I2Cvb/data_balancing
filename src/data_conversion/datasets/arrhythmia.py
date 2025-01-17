""" Cardiac Arrhythmia Database

The original dataset and further information can be found here:

    https://archive.ics.uci.edu/ml/datasets/Arrhythmia

Brief description
-----------------


This data contains 452 observations on 279 variables (206 linear valued
+ 73 nominal) on ECG readings. The data was collected to determine the
type of arrhythmia based on the ECG.


7. Attribute Information:
   -- Complete attribute documentation:
      1 Age: Age in years , linear
      2 Sex: Sex (0 = male; 1 = female) , nominal
      3 Height: Height in centimeters , linear
      4 Weight: Weight in kilograms , linear
      5 QRS duration: Average of QRS duration in msec., linear
      6 P-R interval: Average duration between onset of P and Q waves
        in msec., linear
      7 Q-T interval: Average duration between onset of Q and offset
        of T waves in msec., linear
      8 T interval: Average duration of T wave in msec., linear
      9 P interval: Average duration of P wave in msec., linear
     Vector angles in degrees on front plane of:, linear
     10 QRS
     11 T
     12 P
     13 QRST
     14 J

     15 Heart rate: Number of heart beats per minute ,linear

     Of channel DI:
      Average width, in msec., of: linear
      16 Q wave
      17 R wave
      18 S wave
      19 R' wave, small peak just after R
      20 S' wave

      21 Number of intrinsic deflections, linear

      22 Existence of ragged R wave, nominal
      23 Existence of diphasic derivation of R wave, nominal
      24 Existence of ragged P wave, nominal
      25 Existence of diphasic derivation of P wave, nominal
      26 Existence of ragged T wave, nominal
      27 Existence of diphasic derivation of T wave, nominal

     Of channel DII:
      28 .. 39 (similar to 16 .. 27 of channel DI)
     Of channels DIII:
      40 .. 51
     Of channel AVR:
      52 .. 63
     Of channel AVL:
      64 .. 75
     Of channel AVF:
      76 .. 87
     Of channel V1:
      88 .. 99
     Of channel V2:
      100 .. 111
     Of channel V3:
      112 .. 123
     Of channel V4:
      124 .. 135
     Of channel V5:
      136 .. 147
     Of channel V6:
      148 .. 159

     Of channel DI:
      Amplitude , * 0.1 milivolt, of
      160 JJ wave, linear
      161 Q wave, linear
      162 R wave, linear
      163 S wave, linear
      164 R' wave, linear
      165 S' wave, linear
      166 P wave, linear
      167 T wave, linear

      168 QRSA , Sum of areas of all segments divided by 10,
          ( Area= width * height / 2 ), linear
      169 QRSTA = QRSA + 0.5 * width of T wave * 0.1 * height of T
          wave. (If T is diphasic then the bigger segment is
          considered), linear

     Of channel DII:
      170 .. 179
     Of channel DIII:
      180 .. 189
     Of channel AVR:
      190 .. 199
     Of channel AVL:
      200 .. 209
     Of channel AVF:
      210 .. 219
     Of channel V1:
      220 .. 229
     Of channel V2:
      230 .. 239
     Of channel V3:
      240 .. 249
     Of channel V4:
      250 .. 259
     Of channel V5:
      260 .. 269
     Of channel V6:
      270 .. 279

8. Missing Attribute Values: Several.  Distinguished with '?'.

9. Class Distribution:
       Database:  Arrhythmia

       Class code :   Class   :                       Number of instances:
       01             Normal				          245
       02             Ischemic changes (Coronary Artery Disease)   44
       03             Old Anterior Myocardial Infarction           15
       04             Old Inferior Myocardial Infarction           15
       05             Sinus tachycardy			           13
       06             Sinus bradycardy			           25
       07             Ventricular Premature Contraction (PVC)       3
       08             Supraventricular Premature Contraction	    2
       09             Left bundle branch block 		            9
       10             Right bundle branch block		           50
       11             1. degree AtrioVentricular block	            0
       12             2. degree AV block		            0
       13             3. degree AV block		            0
       14             Left ventricule hypertrophy 	            4
       15             Atrial Fibrillation or Flutter	            5
       16             Others				           22
Original Owner and Donor
------------------------

H. Altay Guvenir, PhD., and, Burak Acar, M.S., and Haldun Muderrisoglu, M.D., Ph.D.,

Bilkent University,
06533 Ankara, Turkey

Email: guvenir@cs.bilkent.edu.tr
Email: buraka@ee.bilkent.edu.tr


References
----------

H. Altay Guvenir, Burak Acar, Gulsen Demiroz, Ayhan Cekin
"A Supervised Machine Learning Algorithm for Arrhythmia Analysis"
Proceedings of the Computers in Cardiology Conference,
Lund, Sweden, 1997.

#TODO: explain that we use class=14
"""


# Authors: Joan Massich and Guillaume Lemaitre
# License: MIT

from os.path import join, exists
from os import makedirs
try:
    # Python 2
    from urllib2 import urlretrieve
except ImportError:
    # Python 3+
    from urllib import urlretrieve

import numpy as np

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/"\
           "arrhythmia/arrhythmia.data"
RAW_DATA_LABEL = 'arrhythmia'

def get_dataset_home(data_home=None, dir=RAW_DATA_LABEL):
    return join(get_data_home(data_home=data_home), dir)

def fetch_arrhythmia(data_home=None, download_if_missing=True):
    """Fetcher for xxxxxxxxxxxxxxxxxxxxx.

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        the original datasets for this `data_balance` study are stored at
        `../data/raw/` subfolders.

    download_if_missing: optional, True by default
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    """
    data_home = get_dataset_home(data_home=data_home)
    if not exists(data_home):
        makedirs(data_home)
    print('downloading Arrhythmia data from %s to %s' % (DATA_URL, data_home))
    urlretrieve(DATA_URL, join(data_home,'data.csv'))

def process_arrhythmia(target=14):
    """Process data of the CoIL 2000 dataset.

    Parameters
    ----------
    target: the target class [0..16]

    Returns
    -------
    (data, label)

    #TODO: check if files exist
    #TODO: a generic file managing using get_data_home
    #TODO:
    """

    #TODO: assert target
    f = join(get_data_home, 'data.csv')

    tmp_input = np.loadtxt(f, delimiter=',')
    return (tmp_input[:, :-1], tmp_input[:, -1])

def convert_arrhythmia_14():
    d, l = process_arrhythmia(target=14)
    np.savez('../data/clean/uci-arrythmia_14.npz', data=d, label=l)

if __name__ == '__main__':
    convert_arrhythmia_14()
