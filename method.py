"""
This is the main module for training and testing the Im2RBTE method
"""
import os
import shutil
import sys
import argparse
import warnings
import utils as ut
import network as nw
import cv2 as cv
import torchvision as tv 
import torch
#import pdb


def train_test():
    # Read the terminal variables
    cv.setNumThreads(0)
    parser = argparse.ArgumentParser(description='Training the Model')
    parser.add_argument('--run', default='', type=str, metavar='TrRun', help='Input training file')
    parser.add_argument('--gpu', type=int, metavar='gpu', help='gpu id')
    args = parser.parse_args()
    if not sys.warnoptions:
        warnings.simplefilter('ignore')
    run = ut.read_yaml('./runs/' + vars(args)['run'])
    if vars(args)['gpu'] is not None:
        run['train']['gpu'] = vars(args)['gpu']
    test_results = []
    for idx in range(run['no_trainings']):
        direct = run['io_var']['save_dir'] + run['io_var']['save_folder_name'] + '/' \
        + run['io_var']['save_subfolder_name'] + '_' + str(idx + 1)
        if not os.path.exists(direct): 
            os.makedirs(direct)
        nms_model = cv.ximgproc.createStructuredEdgeDetection(run['io_var']['nms_model'])

        # Save Training Info
        if not os.path.exists(direct+'/Data'): 
            os.makedirs(direct+'/Data')
        if not os.path.exists(direct + '/Code'): 
            os.makedirs(direct + '/Code')
        if not os.path.isfile(direct + '/Code/' + os.path.basename(__file__)): 
            shutil.copy(__file__, direct + '/Code/' + os.path.basename(__file__))
        if not os.path.isfile(direct + '/Code/augmentations.py'): 
            shutil.copy('./augmentations.py', direct + '/Code/augmentations.py')
        if not os.path.isfile(direct + '/Code/utils.py'): 
            shutil.copy('./utils.py', direct + '/Code/utils.py')
        if not os.path.isfile(direct + '/Code/network.py'): 
            shutil.copy('./network.py', direct + '/Code/network.py')
        if not os.path.isfile(direct + '/Hyperparameters.yaml'): 
            ut.save_yaml({'run': 'python ' + ' '.join(sys.argv), 'io_var': run['io_var'],
                          'train': run['train'], 
                          'extra_options': run['extra_options'],
                          'train_transforms': run['train_transforms'], 
                          'val_transforms': run['val_transforms']}, 
                         direct + '/Hyperparameters.yaml')

        # Datasets, Network, Optimizer, Scheduler and Loss
        train_dataset_list, train_list_class = ut.dirlist(run['io_var']['train_dataset_list'],
                                                          'training_data', direct+'/Data')
        val_dataset_list, val_list_class = ut.dirlist(run['io_var']['val_dataset_list'], 
                                                      'validation_data', direct+'/Data')
        train_tr = ut.create_transform(input_list=run['train_transforms'], nms_model=nms_model)
        val_tr = ut.create_transform(input_list=run['val_transforms'], nms_model=nms_model)
        train_dataset = ut.ImageFolderSubsetPath(image_list=train_dataset_list, 
                                                 class_list=train_list_class, 
                                                 transform=tv.transforms.Compose(train_tr))
        val_dataset = ut.ImageFolderSubsetPath(image_list=val_dataset_list, 
                                               class_list=val_list_class,
                                               transform=tv.transforms.Compose(val_tr))
        train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                        batch_size=run['train']['batch'],
                                                        num_workers=run['train']['workers'],
                                                        shuffle=True)    
        val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                                      batch_size=run['train']['batch'], 
                                                      num_workers=run['train']['workers'],
                                                      shuffle=False)
        #pdb.set_trace()
        net = nw.initizalize_network(architecture=run['train']['architecture'], 
                                     no_classes=len(train_dataset.classes), 
                                     load_model=run['io_var']['load_model'], 
                                     pretrained=run['train']['pretrained'])
        optimizer = torch.optim.Adam(net.parameters(), lr=run['train']['learning_rate'],
                                     betas=(0.9, 0.999), 
                                     weight_decay=run['train']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=run['train']['schedule'], 
                                                    gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss()

        # Learning Rate Search
        if run['extra_options']['lr_mode'] and idx == 0:
            ut.lr_tool(net=net, criterion=criterion, original_optimizer=optimizer, 
                       train_data_loader=train_data_loader, gpu_id=run['train']['gpu'],
                       direct=direct, start_lr=0.0000001, end_lr=10, step_size=50, gamma=10)

        # Training
        ut.set_all_seeds(idx)
        last_model_name = ut.training(net=net, criterion=criterion, optimizer=optimizer,
                                      scheduler=scheduler, train_data_loader=train_data_loader,
                                      val_data_loader=val_data_loader, 
                                      total_epochs=run['train']['epochs'], 
                                      gpu_id=run['train']['gpu'], direct=direct)

        # Testing
        testing_name = []
        for testing_data in run['io_var']['test_dataset_list']:
            for testing_transforms in run['test_transforms']:
                print('Testing ' + testing_data['name'] + ' ' + testing_transforms['name'])
                test_dataset_list, test_list_class = ut.dirlist(testing_data['directory'], 
                                                                'test_data_'+testing_data['name'],
                                                                direct+'/Data')
                test_tr = ut.create_transform(input_list=testing_transforms['transform'], 
                                              nms_model=nms_model)
                test_dataset = ut.ImageFolderSubsetPath(image_list=test_dataset_list, 
                                                        class_list=test_list_class, 
                                                        transform=tv.transforms.Compose(test_tr))
                test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                               batch_size=run['train']['batch'],
                                                               num_workers=run['train']['workers'],
                                                               shuffle=False) 
                performance_dict, _, _ = ut.testing(net=net, test_data_loader=test_data_loader, 
                                                    direct=direct, model_name=last_model_name, 
                                                    gpu_id=run['train']['gpu'], 
                                                    train_class_list=train_dataset.classes,
                                                    dataset_list=testing_data['directory'], 
                                                    transform=testing_transforms['transform'],
                                                    name=testing_data['name']+'_'\
                                                    +testing_transforms['name'])
                if performance_dict['name'] not in testing_name:
                    testing_name.append(performance_dict['name'])
                test_results.append(performance_dict)
        del net, optimizer, scheduler, criterion
        with torch.cuda.device('cuda:' + str(run['train']['gpu'])):
            torch.cuda.empty_cache()

    if run['no_trainings'] > 1:
        ut.save_average_testing(testing_name=testing_name, test_results=test_results, 
                                save_dir=run['io_var']['save_dir']\
                                +run['io_var']['save_folder_name']+'/'\
                                +run['io_var']['save_subfolder_name']\
                                +'_'+str(run['no_trainings']))


if __name__ == '__main__':
    train_test()
