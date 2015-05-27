custom_data_home = '../data/raw/'

from sklearn.datasets import fetch_mldata
for dataset_name in {'uci-20070111 abalone', 'uci-20070111 letter', 'uci-20070111 diabetes'}:
    xx = fetch_mldata(dataset_name, data_home=custom_data_home)
    print '\n' + dataset_name
    for k, v in xx.items():
        try :
            print '\t{0}: {1}'.format(k, v.shape)
        except AttributeError:
            print '\t{0}: {1}'.format(k, v)

