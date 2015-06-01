PROJECT_DATA_PATH = '../data/raw/' # it assumes that code is executed from $PROJECT/src

def get_data_home(data_home=None):
    """ Return the path to the experiment's data dir.

    This folder used to avoid repetitive downloads of the dataset
    is placed at '$PROJECT_DRI/data/raw/'

    If the folder does not already exist, it is automatically created.

    This function is a wrapper of `sklearn.datasets.get_data_home`
    """
    from sklearn.datasets import get_data_home as gdh

    return gdh(data_home=PROJECT_DATA_PATH)
