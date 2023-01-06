import os
import tarfile
import argparse
import urllib.request

parser = argparse.ArgumentParser(description='Download and extract the datasets')
parser.add_argument('--dataset', default='im4sketch', type=str, metavar='', help='choose dataset')
parser.add_argument('--download', default='yes', type=str, metavar='', help='download the dataset')
parser.add_argument('--extract', default='yes', type=str, metavar='', help='extract tar')
parser.add_argument('--models', default='yes', type=str, metavar='', help='download the models')
parser.add_argument('--delete', default='no', type=str, metavar='', help='delete tar')
args = parser.parse_args()

def extract_tar(archive_file, target_dir=None):
    if not target_dir:
        target_dir, _ = os.path.split(archive_file)
    try:
        with tarfile.open(archive_file, 'r:gz') as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, target_dir)
    except:
        print(archive_file, 'was not extracted')

def dataset_files(dataset):
    if dataset.lower() in ['im4sketch', 'i4s', 'all']:
        data_list = ['PACS.tar.gz', 'Sketchy.tar.gz', 'TU-Berlin.tar.gz', 'DomainNet.tar.gz', 
                     'ImageNet_1.tar.gz', 'ImageNet_2.tar.gz', 'ImageNet_3.tar.gz', 
                     'ImageNet_4.tar.gz', 'ImageNet_5.tar.gz']
    elif dataset.lower() == 'pacs':
        data_list = ['PACS.tar.gz']
    elif dataset.lower() == 'sketchy':
        data_list = ['Sketchy.tar.gz']
    elif dataset.lower() in ['tu-berlin', 'tub']:
        data_list = ['TU-Berlin.tar.gz']
    elif dataset.lower() in ['domainnet', 'domainet', 'dn']:
        data_list = ['DomainNet.tar.gz']
    elif dataset.lower() in ['imagenet', 'in']:
        data_list = ['ImageNet_1.tar.gz', 'ImageNet_2.tar.gz', 'ImageNet_3.tar.gz', 
                     'ImageNet_4.tar.gz', 'ImageNet_5.tar.gz']

    return data_list

data_files = dataset_files(dataset=vars(args)['dataset'])

if not os.path.exists('./Data'): 
    os.makedirs('./Data')

if vars(args)['models'].lower() in ['yes', 'true', 'y', 't']:
    if not os.path.exists('./Pretrained_Models'):
        os.makedirs('./Pretrained_Models')
    print('downloading the models')
    urllib.request.urlretrieve('http://ptak.felk.cvut.cz/im4sketch/Models/opencv_extra.yml.gz', 
                               './Pretrained_Models/opencv_extra.yml.gz')
    urllib.request.urlretrieve('http://ptak.felk.cvut.cz/im4sketch/Models/im4sketch_model.pt', 
                               './Pretrained_Models/im4sketch_model.pt')

if vars(args)['download'].lower() in ['yes', 'true', 'y', 't']:
    for data in data_files:
        print('downloading', data)
        urllib.request.urlretrieve('http://ptak.felk.cvut.cz/im4sketch/'+data, './Data/'+data)

if vars(args)['extract'].lower() in ['yes', 'true', 'y', 't']:
    for data in data_files:
        print('extracting', data)
        if data not in ['ImageNet_2.tar.gz', 'ImageNet_3.tar.gz', 
                        'ImageNet_4.tar.gz', 'ImageNet_5.tar.gz']:
            if os.path.isfile('./Data/'+data): 
                extract_tar('./Data/'+data, './Data/')
            else:
            	print(data, 'file not found')
        else:
            if not os.path.exists('./Data/ImageNet'): 
                os.makedirs('./Data/ImageNet')
            if not os.path.exists('./Data/ImageNet/real_train'): 
                os.makedirs('./Data/ImageNet/real_train')
            if os.path.isfile('./Data/'+data):
                extract_tar('./Data/'+data, './Data/ImageNet/real_train/')
            else:
                print(data, 'file not found')

if vars(args)['delete'].lower() in ['yes', 'true', 'y', 't']:
    for data in data_files:
        print('Deleting', data)
        if os.path.isfile('./Data/'+data):
            os.remove('./Data/'+data)
        else:
            print(data, 'file not found')
