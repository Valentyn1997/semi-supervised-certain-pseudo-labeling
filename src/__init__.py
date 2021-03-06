from os.path import abspath, dirname
import logging

logging.basicConfig(level=logging.INFO)

ROOT_PATH = dirname(dirname(abspath(__file__)))
DATA_PATH = f'{ROOT_PATH}/data'
CONFIG_PATH = f'{ROOT_PATH}/config'
GLOBAL_ARTIFACTS_PATH = '/nfs/data3/obermeier/sscpl/'
