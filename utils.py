import csv
import math
import os
import time
import PIL
import yaml
import augmentations as aug
import torch
import torchvision as tv
import random
import numpy as np

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_transform(input_list, nms_model=None):
    transform_list = []
    for tr_module in input_list:
        if tr_module['name'] == 'square':
            transformation = aug.Square()
        elif tr_module['name'] == 'pad':
            transformation = aug.Pad(percent=tr_module['percent'], color=tr_module['color'])
        elif tr_module['name'] == 'resize':
            transformation = tv.transforms.Resize(size=tr_module['size'])
        elif tr_module['name'] == 'random_affine':
            transformation = tv.transforms.RandomAffine(degrees=tr_module['degrees'], 
                                                        shear=tr_module['shear'])
        elif tr_module['name'] == 'random_resized_crop':
            transformation = tv.transforms.RandomResizedCrop(size=tr_module['size'], 
                                                             scale=float_tuple(tr_module['scale']),
                                                             ratio=float_tuple(tr_module['ratio']))
        elif tr_module['name'] == 'random_horizontal_flip':
            transformation = tv.transforms.RandomHorizontalFlip(p=tr_module['p'])
        elif tr_module['name'] == 'edge_detector':
            transformation = aug.EdgeDetector(edge_mode=tr_module['edge_mode'])
        elif tr_module['name'] == 'ori_nms':
            transformation = aug.OriNMS(model=nms_model, prob=tr_module['prob'], 
                                        radious=tr_module['radious'], 
                                        bound_radious=tr_module['bound_radious'], 
                                        multi=tr_module['multi'])
        elif tr_module['name'] == 'thresholder':
            transformation = aug.Thresholder(thresh_rand=tr_module['thresh_rand'], 
                                             thinning=tr_module['thinning'], 
                                             thresh_mode=tr_module['thresh_mode'], 
                                             hyst_prob=tr_module['hyst_prob'], 
                                             hyst_par=tr_module['hyst_par'], 
                                             hyst_pert=tr_module['hyst_pert'])
        elif tr_module['name'] == 'cleaner':
            transformation = aug.Cleaner(percent_of_cc=tr_module['percent_of_cc'], 
                                         del_less_than=tr_module['delete_less_than'])
        elif tr_module['name'] == 'multi_scale':
            transformation = aug.MultiScale(size=tr_module['size'], 
                                            size_multipliers=tr_module['size_multipliers'], 
                                            return_white_bg=tr_module['return_white_bg'])
        elif tr_module['name'] == 'black_background':
            transformation = aug.BlackBackground()
        elif tr_module['name'] == 'resize_np':
            transformation = aug.ResizeNP(size=tr_module['size'])
        elif tr_module['name'] == 'square_np':
            transformation = aug.SquareNP()
        transform_list.append(aug.ListTransform(transformation))
    transform_list.append(aug.ToStackTensor())
    return transform_list


def float_tuple(value):
    try:
        if isinstance(value, float):
            value = (value, value)
        elif isinstance(value, str):
            value = (float(value), float(value))
        elif isinstance(value, list):
            if len(value) < 2:
                value = (float(value[0]), float(value[0]))
            else:
                value = tuple([float(i) for i in value])
        return value
    except ValueError:
        print('input cannot change to float')


def dirlist(dirs, datatype, save):
    if dirs == 'None':
        return None, None

    if isinstance(dirs, str):
        dirs = [dirs]
    images = []
    classes = []
    for current_dir in dirs:
        with open(current_dir, mode='r', encoding='utf-8') as file:
            for line in file:
                if line != '':
                    if len(line.split()) > 1:
                        images.append(os.path.dirname(current_dir)+'/'+line.split()[0])
                        classes.append(str(line.split()[1]))
    rows = zip(images, classes)
    with open(save+'/'+datatype+'.csv', 'w') as file:
        writer = csv.writer(file, delimiter=' ')
        for row in rows:
            writer.writerow(row)
    return images, classes


def pil_loader(path):
    with open(path, 'rb') as file:
        img = PIL.Image.open(file)
        return img.convert('RGB')

IMAGE_EXTS = ['.jpg', '.png', '.jpeg']

class ImageFolderSubsetPath(tv.datasets.ImageFolder):
    def __init__(self, image_list, class_list, transform=None, target_transform=None, 
                 loader=pil_loader):
        if image_list == [] or not isinstance(image_list, list): 
            raise RuntimeError('Need a list of image directiries')
        if class_list == [] or not isinstance(class_list, list): 
            raise RuntimeError('Need a list of classes')
        if len(class_list) != len(image_list):
            raise RuntimeError('Image and class lists must be of same size')

        classes = list(set(class_list.copy()))
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = []

        for i, path in enumerate(image_list):
            target = class_to_idx[class_list[i]]
            self.samples.append((path, target))

        self.classes = classes
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.samples[index]
        if os.path.isfile(path):
            image = self.loader(path)
        else:
            if path[0] == '.':
                tmp_image_dir = '.'+path.split('.')[1]
            else:
                tmp_image_dir = path.split('.')[0]
            for ext in IMAGE_EXTS:
                if os.path.isfile(tmp_image_dir+ext):
                    image = self.loader(tmp_image_dir+ext)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            image = self.transform(image)

        return image, target, path

    def __len__(self):
        return len(self.samples)


def save_architecture(net, direct, name = 'architecture'):
    with open(direct + '/' + name + '.txt', 'w') as file:
        print(net, file=file)
        print('', file=file)
        for param_name, param in net.named_parameters():
            print(param_name, param.requires_grad, file=file)


def freeze_network(model, mode = ['fc', 'bn', 'conv']):
    if not isinstance(mode, list):
        mode = [mode]
    mode = [x.lower() for x in mode]
    if 'all' in mode:
        for name, par in model.named_parameters():
            par.requires_grad = False
    else:
        for component in mode:
            for name, par in model.named_parameters():
                if component in name:
                    par.requires_grad = False        


def set_batchnorm_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_log(log_list, direct, name=None):
    ''' 
    Inputs a list (log list) of lists (evaluation lists) of dictionaries (epoch dictionaries) 
    or a list of dictionaries of epoch performances and a directory. It saves the performances 
    to the directory. The outer list can contain an abstract number of lists. Each one of them 
    is a list of epoch dictionaries for a spesific evaluation. 
    '''
    if not isinstance(log_list[0], list):
        log_list = [log_list]
    if name is None:
        name = 'All_Metrics_Log'
    cell_names = ['Epoch']
    for eval_lists in log_list:
        cell_names = cell_names + [eval_lists[0]['name'] + '_Acc', 
                                   eval_lists[0]['name'] + '_Acc_Top-5',
                                   eval_lists[0]['name'] + '_Loss', 
                                   eval_lists[0]['name'] + '_Grad', 
                                   eval_lists[0]['name'] + '_Load_Sec', 
                                   eval_lists[0]['name'] + '_Rest_Sec']
    epoch = list(range(1, len(log_list[0]) + 1))
    data = [epoch]
    for evaluation_lists in log_list:
        acc, acc5, loss, grad, load, rest = ([] for i in range(6))
        for epoch_dict in evaluation_lists:
            acc.append(epoch_dict['accuracy'])
            acc5.append(epoch_dict['accuracy_top5'])
            loss.append(epoch_dict['loss'])
            grad.append(epoch_dict['grad'])
            load.append(epoch_dict['loading_seconds'])
            rest.append(epoch_dict['rest_seconds'])
        data = data + [acc, acc5, loss, grad, load, rest]
    rows = zip(*data)
    try:
        with open(direct + '/'+str(name)+'.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(cell_names)
            for row in rows:
                writer.writerow(row)
        return()
    except PermissionError:
        return()


def save_input(inputs, save_no, direct, name = ''):
    save_no = min(save_no, inputs.shape[0])
    nrows = math.ceil(math.sqrt(save_no))

    if len(inputs.shape) < 5:
        tv.utils.save_image(inputs[0:save_no].cpu(), 
                            direct+'/inputs'+name+'.png', nrow=nrows)
    else:
        tv.utils.save_image(inputs[0:save_no][0].cpu(), 
                            direct+'/inputs1'+name+'.png', nrow=nrows)
        tv.utils.save_image(inputs[0:save_no][1].cpu(), 
                            direct+'/inputs2'+name+'.png', nrow=nrows)
        tv.utils.save_image(inputs[0:save_no][2].cpu(), 
                            direct+'/inputs3'+name+'.png', nrow=nrows)


def grads(net):
    if hasattr(net, 'backbone'):
        architecture = net.backbone.__class__.__name__.lower()
        if architecture == 'resnet':
            conv1_grad = torch.norm(net.backbone.conv1.weight.grad, 2).item()
        else:
            conv1_grad = -1
    else:
        architecture = net.__class__.__name__.lower()
        if architecture == 'resnet':
            conv1_grad = torch.norm(net.conv1.weight.grad, 2).item()
        else:
            conv1_grad = -1
    return conv1_grad


def batch_run(net, optimizer, criterion, train, image, target, gpu_id):
    if train:
        net.train().cuda(gpu_id)
        torch.set_grad_enabled(train)
        optimizer.zero_grad()
    inputs = torch.autograd.Variable(image).cuda(gpu_id) 
    classes = torch.autograd.Variable(target).cuda(gpu_id)
    outputs = net(inputs).cuda(gpu_id)
    if train:
        loss = criterion(outputs, classes)
        loss.backward()
        optimizer.step()
    net.eval().cuda(gpu_id)
    with torch.no_grad():
        outputs = net(inputs).detach().clone().cuda(gpu_id)
        _, top5_class = torch.topk(outputs, 5, dim=1)
        loss_save = criterion(outputs, classes).item()
        if train:
            grad = grads(net)
        else:
            grad = -1

    return inputs.detach().clone().cuda(gpu_id), outputs, top5_class, classes, loss_save, grad


def epoch_run(net, data_loader, criterion, gpu_id, optimizer, name=None, train=True, 
              schedule=None):
    start = time.time()
    net.eval().cuda(gpu_id)
    torch.set_grad_enabled(train)
    sum_top5_corr, no_elements, sum_corr, sum_loss, sum_grad = 0, 0, 0, 0, 0
    loading_seconds, rest_seconds = 0, 0
    for i, (image, target, *_) in enumerate(data_loader, 0):
        end = time.time()
        loading_seconds += end - start
        inputs, outputs, top5_class, classes, loss_save, grad = batch_run(net=net, 
                                                                          optimizer=optimizer, 
                                                                          criterion=criterion, 
                                                                          train=train, 
                                                                          image=image, 
                                                                          target=target, 
                                                                          gpu_id=gpu_id)
        sum_grad += grad
        sum_top5_corr += (top5_class.permute(1, 0) == classes).float().sum().item()
        sum_corr += (outputs.argmax(dim=1) == classes).float().sum().item()
        no_elements += classes.numel() 
        sum_loss += loss_save
        if i == 0:
            inputs_example = inputs
        start = time.time()
        rest_seconds += start - end
    if schedule is not None and train:
        schedule.step()
    epoch_results = {
        'name': str(name),
        'train': str(train), 
        'accuracy': sum_corr/no_elements,
        'accuracy_top5': sum_top5_corr/no_elements, 
        'grad': sum_grad/len(data_loader), 
        'loss': sum_loss/len(data_loader), 
        'inputs_example': inputs_example, 
        'loading_seconds': loading_seconds, 
        'rest_seconds': rest_seconds
        }
    return epoch_results


def save_model(net, optimizer, epoch, direct, name=''):
    state = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, direct+'/'+name +'.pt') 


def load_model(net, model_location):
    ''' Inputs an architecture and a checkpoint path and loads the checkpoint '''
    device = next(net.parameters()).device
    state = torch.load(model_location, map_location=device)
    net.load_state_dict(state['state_dict'])
    net.eval()


def print_epoch(epoch_list, epoch, learning_rate):
    total_seconds = 0
    for result in epoch_list:
        total_seconds = total_seconds + result['loading_seconds'] + result['rest_seconds']
    print('_'*60)
    print('')
    print('Epoch: ', epoch)
    print('Learning Rate: ', learning_rate)
    print('_'*30)
    print('')
    for result in epoch_list:
        print(result['name'] + ' Accuracy: ', '%.2f' % result['accuracy'], 
              ' Top-5 ' + result['name'] + ' Accuracy: ', '%.2f' % result['accuracy_top5'])
    print('_'*45)
    print('')
    for result in epoch_list:
        print(result['name'] + ' Loss:', '%.5f' % result['loss'])
    print('_'*30)
    print('')
    print('Total time        : ', '%.0f' % (total_seconds//60), 
          'min', '%.0f' % (total_seconds%60), 'sec')
    for result in epoch_list:
        print(result['name'] + ' loading time: ', '%.0f' % (result['loading_seconds']//60), 
              'min', '%.0f' % (result['loading_seconds']%60), 'sec')
        print(result['name'] + ' rest time     : ', '%.0f' % (result['rest_seconds']//60), 
              'min', '%.0f' % (result['rest_seconds']%60), 'sec')
    print('_'*60)
    print('')


def print_dict(perf_dict):
    print('_'*50)
    print('')
    print(yaml.dump(perf_dict, sort_keys=True, indent=4))
    print('_'*50)


def save_lr_log(direct, lr, loss, acc1, acc5, grad):
    rows = zip(lr, loss, acc1, acc5, grad)
    try:
        with open(direct + '/All_Metrics_LR_Log.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Learning_Rate', 'Loss', 'Accuracy_Top-1', 
                             'Accuracy_Top-5', 'Gradient_L2_Norm'])
            for row in rows:
                writer.writerow(row)
        return()
    except PermissionError:
        return()


def lr_tool(net, criterion, original_optimizer, train_data_loader, gpu_id, direct, 
            start_lr=0.0000001, end_lr=10, step_size=50, gamma=10):
    if (os.path.isfile(direct + '/All_Metrics_LR_Log.csv') 
            and os.path.isfile(direct + '/Starting_Network.pt')):
        print('The learning rate test is already done')
        return
    save_model(net=net, optimizer=original_optimizer, epoch=0, direct=direct, 
               name='Starting_Network')
    optimizer = torch.optim.Adam(net.parameters(), lr=start_lr, betas=(0.9, 0.999), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    loss_list, lr_list, acc1_list, acc5_list, grad_list = ([] for i in range(5))

    while get_lr(optimizer) <= end_lr: 
        for i, (image, target, *_) in enumerate(train_data_loader, 0):

            _, outputs, top5_class, classes, loss, grad = batch_run(net=net, 
                                                                    optimizer=optimizer, 
                                                                    criterion=criterion, 
                                                                    train=True, 
                                                                    image=image, 
                                                                    target=target, 
                                                                    gpu_id=gpu_id)
            no_elements = classes.numel() 
            grad_list.append(grad)
            loss_list.append(loss)
            acc1_list.append((outputs.argmax(dim=1) == classes).float().sum().item()/no_elements)
            acc5_list.append((top5_class.permute(1, 0) == classes).float().sum().item()/no_elements)
            lr_list.append(get_lr(optimizer))

            scheduler.step()
            print(round(loss, 4), round(get_lr(optimizer), 10), ' '*15, end='\r')
            if len(loss_list) > 5:
                if sum(loss_list[-5:]) > 10*sum(loss_list[0:5]):
                    save_lr_log(direct, lr_list, loss_list, acc1_list, acc5_list, grad_list)
                    load_model(net=net, model_location=direct+'/Starting_Network.pt')
                    return
    save_lr_log(direct, lr_list, loss_list, acc1_list, acc5_list, grad_list)
    load_model(net=net, model_location=direct+'/Starting_Network.py')
    return

def training(net, criterion, optimizer, scheduler, train_data_loader, val_data_loader, 
             total_epochs, gpu_id, direct):
    if os.path.isfile(direct + '/' + 'Epoch_'+str(total_epochs)+'.pt'):
        print('The training was already done')
        load_model(net, direct + '/' + 'Epoch_'+str(total_epochs)+'.pt')
        return 'Epoch_'+str(total_epochs)+'.pt'
    train_epoch_list, val_epoch_list = ([] for i in range(2))
    save_architecture(net, direct)
    representations = direct + '/representations'
    if not os.path.exists(representations):
        os.makedirs(representations)

    for epoch in range(1, total_epochs + 1):
        train_epoch_results = epoch_run(net=net, data_loader=train_data_loader, 
                                        criterion=criterion, gpu_id=gpu_id, optimizer=optimizer, 
                                        name='Train', train=True, schedule=scheduler)
        val_epoch_results = epoch_run(net=net, data_loader=val_data_loader, criterion=criterion, 
                                      gpu_id=gpu_id, optimizer=optimizer, name='Val', train=False, 
                                      schedule=scheduler)
        train_epoch_list.append(train_epoch_results)
        val_epoch_list.append(val_epoch_results)
        print_epoch(epoch_list=[train_epoch_results, val_epoch_results], epoch=epoch, 
                    learning_rate=round(get_lr(optimizer), 8))
        save_log([train_epoch_list, val_epoch_list], direct)
        if epoch == total_epochs:
            print('Saving the model')
            save_model(net=net, optimizer=optimizer, epoch=epoch, direct=direct, 
                       name='Epoch_'+str(epoch))
        epoch_dir = representations + '/epoch_'+str(epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        save_input(inputs=train_epoch_results['inputs_example'], save_no=9, direct=epoch_dir, 
                   name=train_epoch_results['name'])
        save_input(inputs=val_epoch_results['inputs_example'], save_no=9, direct=epoch_dir, 
                   name=val_epoch_results['name'])
        train_epoch_list[len(train_epoch_list)-1]['inputs_example'] = None
        val_epoch_list[len(val_epoch_list)-1]['inputs_example'] = None

    return 'Epoch_'+str(total_epochs)+'.pt'


def testing(net, test_data_loader, direct, model_name, gpu_id, dataset_list, transform, 
            train_class_list=None, name=''):
    save_dir = direct + '/' + model_name[:-3]+'_'+name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if (os.path.isfile(save_dir +'/Classification_Report_Full_Argmax.csv') 
            and os.path.isfile(save_dir + '/conf_and_hist.yaml') 
            and os.path.isfile(save_dir + '/Performance.yaml')):
        try:
            perf_dict = read_yaml(save_dir + '/Performance.yaml')
            conf_and_hist = read_yaml(save_dir + '/conf_and_hist.yaml')
            print('Testing was already done')
            print_dict(perf_dict)
            return (perf_dict, conf_and_hist['confusion_full'], 
                    conf_and_hist['histogram_full'])

        except (PermissionError, KeyError):
            print('There are unusual files from previouse testing. Running the test again')

    net.eval().cuda(gpu_id)

    save_yaml({'test_dataset_list': dataset_list, 'test_transforms': transform}, 
              save_dir + '/Test_hyperparameters.yaml')
    save_architecture(net, save_dir, 'architecture_1')
    try:
        net = net.backbone
        print('The backbone was extracted from the network')
    except:
        print('The network was already just backbone')
    save_architecture(net, save_dir, 'architecture_2')

    test_class_list = test_data_loader.dataset.classes
    if train_class_list is None:
        train_class_list = test_class_list
    train_class_list.sort()
    test_class_list.sort()
    mask = make_mask(train_class_list, test_class_list)

    all_dir, all_actual_class, all_actual_prob, all_class_confidences = ([] for i in range(4))
    all_class_predictions, all_actual_prob_less = ([] for i in range(2))
    all_class_conf_less, all_class_pred_less = ([] for i in range(2))
    with torch.no_grad():
        for j, (image, target, path) in enumerate(test_data_loader, 0):
            
            inputs = torch.autograd.Variable(image).cuda(gpu_id)
            labels = torch.autograd.Variable(target).cuda(gpu_id)
            print('Testing: ', str(int(100*(j + 1)/len(test_data_loader))) + '%', end='\r')
            if j < 20: 
                save_input(inputs=inputs, save_no=16, direct=save_dir, name=str(j))

            if len(inputs.size()) > 4:
                batch, crops, channel, height, width = inputs.size()
                outputs_all = net(inputs.view(-1, channel, height, width)) # fuse crops to batch
                outputs_all = outputs_all.view(batch, crops, -1).mean(1) # avg over crops
            else:
                outputs_all = net(inputs).cuda(gpu_id)
            
            outputs_all = torch.nn.functional.softmax(outputs_all, dim=1)
            outputs_less = masked_output(outputs_all, mask)

            top5_value, top5_class = torch.topk(outputs_all, 5, dim=1)
            top5_value_less, top5_class_less = torch.topk(outputs_less, 5, dim=1)
            for i in range(inputs.shape[0]): # Batch loop
                actual = test_class_list[labels[i].item()]
                actual_idx_all_names = train_class_list.index(actual)

                all_dir.append(path[i])
                all_actual_class.append(actual)
                all_actual_prob.append(round(outputs_all[i][actual_idx_all_names].item(), 4))
                
                all_class_confidences.append([round(k.item(), 4) for k in top5_value[i]])
                all_class_predictions.append([train_class_list[k.item()] for k in top5_class[i]])

                all_actual_prob_less.append(round(outputs_less[i][actual_idx_all_names].item(), 4))
                all_class_conf_less.append([round(k.item(), 4) for k in top5_value_less[i]])
                all_class_pred_less.append([train_class_list[k.item()] for k in top5_class_less[i]])
            
    perf_dict = {}
    perf_dict['full_argmax'], hist_full, confusion_full = metrics(class_names=train_class_list, 
                                                                  actual=all_actual_class, 
                                                                  top5_pred=all_class_predictions)
    perf_dict['less_argmax'], hist_less, confusion_less = metrics(class_names=train_class_list, 
                                                                  actual=all_actual_class, 
                                                                  top5_pred=all_class_pred_less)
    perf_dict['name'] = name
    classification_report(save_dir=save_dir, all_dir=all_dir, all_actual_class=all_actual_class, 
                          all_actual_prob=all_actual_prob, 
                          all_class_predictions=all_class_predictions, 
                          all_class_confidences=all_class_confidences, name='Full_Argmax')
    classification_report(save_dir=save_dir, all_dir=all_dir, all_actual_class=all_actual_class, 
                          all_actual_prob=all_actual_prob_less, 
                          all_class_predictions=all_class_pred_less, 
                          all_class_confidences=all_class_conf_less, name='Less_Argmax')

    with open(save_dir + '/Performance.yaml', 'w') as file:
        documents = yaml.dump(perf_dict, file)
    
    print_dict(perf_dict)
    save_yaml({'histogram_full': hist_full, 'histogram_less': hist_less, 
               'confusion_full': confusion_full, 'confusion_less':confusion_less}, 
              save_dir + '/conf_and_hist.yaml')

    return perf_dict, confusion_full, hist_full


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def save_yaml(structure, direct):
    with open(direct, 'w') as file:
        documents = yaml.dump(structure, file, width=10000, Dumper=NoAliasDumper)


def read_yaml(direct):
    with open(direct) as file:
        structure = yaml.full_load(file)
    return structure


def make_mask(train_class_list, test_class_list):
    mask = []
    for i in train_class_list:
        if i in test_class_list:
            mask.append(1)
        else:
            mask.append(0)
    return mask


def masked_output(outputs_all, mask):
    outputs_less = outputs_all.clone()
    min_output = torch.min(outputs_less)
    for i, mask_element in enumerate(mask):
        if mask_element == 0:
            outputs_less[:, i] = min_output
    return outputs_less


def metrics(class_names, actual, top5_pred):

    correct, correct_top5 = 0, 0
    performance = {}

    confusion = dict.fromkeys(list(class_names))
    for true_class in class_names:
        confusion[true_class] = {}
        for predicted_class in class_names:
            confusion[true_class][predicted_class] = 0

    histogram = dict.fromkeys(list(class_names))
    for true_class in class_names:
        histogram[true_class] = {}
        for prediction in ('Correct', 'Wrong'):
            histogram[true_class][prediction] = 0

    for actual_class, top5_pred_class in zip(actual, top5_pred):
        confusion[actual_class][top5_pred_class[0]] += 1

        if actual_class in top5_pred_class:
            correct_top5 += 1

        if actual_class == top5_pred_class[0]:
            histogram[actual_class]['Correct'] += 1
            correct += 1
        else:
            histogram[actual_class]['Wrong'] += 1

    performance['acc'] = round(correct/len(actual), 4)
    performance['acc_top5'] = round(correct_top5/len(actual), 4)

    class_no = len(set(actual))
    class_acc_list, class_acc_top5_list = ([] for i in range(2))
    for i in set(actual):
        class_total, class_correct, class_correct_top5 = 0, 0, 0
        for count, j in enumerate(actual):
            if i == j:
                class_total += 1
                if j in top5_pred[count]:
                    class_correct_top5 += 1
                    if j == top5_pred[count][0]:
                        class_correct += 1
        class_acc_list.append(class_correct/class_total)
        class_acc_top5_list.append(class_correct_top5/class_total)
    performance['acc_class'] = round(sum(class_acc_list)/class_no, 4)
    performance['acc_class_top5'] = round(sum(class_acc_top5_list)/class_no, 4)

    return(performance, histogram, confusion)


def classification_report(save_dir, all_dir, all_actual_prob, all_actual_class, 
                          all_class_predictions, all_class_confidences, name=''):
    if name != '': 
        name = '_' + name
    rows = zip(all_dir, all_actual_class, all_actual_prob, 
               [col[0] for col in all_class_predictions], 
               [col[0] for col in all_class_confidences], 
               [col[1] for col in all_class_predictions], 
               [col[1] for col in all_class_confidences], 
               [col[2] for col in all_class_predictions], 
               [col[2] for col in all_class_confidences], 
               [col[3] for col in all_class_predictions], 
               [col[3] for col in all_class_confidences], 
               [col[4] for col in all_class_predictions], 
               [col[4] for col in all_class_confidences])
    try:
        with open(save_dir+'/Classification_Report'+name+'.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Directory', 'Actual', 'Actual_Probability', 'Predicted', 
                             'Predicted_Probability', 'Predicted_2', 'Predicted_2_Probability', 
                             'Predicted_3', 'Predicted_3_Probability', 'Predicted_4', 
                             'Predicted_4_Probability', 'Predicted_5', 'Predicted_5_Probability'])
            for row in rows:
                writer.writerow(row)
    except PermissionError:
        print('An exception occurred')


def save_average_testing(testing_name, test_results, save_dir):
    for name in testing_name:
        counter = 0
        result = {'name': name, 
                  'full_argmax': {'acc': 0, 'acc_class': 0, 
                                  'acc_class_top5': 0, 'acc_top5': 0},
                  'full_argmax_sd': {'acc': 0, 'acc_class': 0, 
                                     'acc_class_top5': 0, 'acc_top5': 0}, 
                  'full_cumul_sq': {'acc': 0, 'acc_class': 0, 
                                    'acc_class_top5': 0, 'acc_top5': 0}, 
                  'less_argmax': {'acc': 0, 'acc_class': 0, 
                                  'acc_class_top5': 0, 'acc_top5': 0},
                  'less_argmax_sd': {'acc': 0, 'acc_class': 0, 
                                     'acc_class_top5': 0, 'acc_top5': 0},
                  'less_cumul_sq': {'acc': 0, 'acc_class': 0, 
                                    'acc_class_top5': 0, 'acc_top5': 0}
                 }
        for performance in test_results:
            if result['name'] == performance['name']:
                result['full_argmax'] = {k: result['full_argmax'].get(k, 0) \
                                         + performance['full_argmax'].get(k, 0) \
                                         for k in set(result['full_argmax'])}
                result['less_argmax'] = {k: result['less_argmax'].get(k, 0) \
                                         + performance['less_argmax'].get(k, 0) \
                                         for k in set(result['less_argmax'])}
                result['full_cumul_sq'] = {k: result['full_cumul_sq'].get(k, 0) \
                                           + pow(performance['full_argmax'].get(k, 0), 2) \
                                           for k in set(result['full_cumul_sq'])}
                result['less_cumul_sq'] = {k: result['less_cumul_sq'].get(k, 0) \
                                           + pow(performance['less_argmax'].get(k, 0), 2) \
                                           for k in set(result['less_cumul_sq'])}
                counter += 1
        result['full_argmax'] = {k: result['full_argmax'].get(k, 0) / counter \
                                 for k in set(result['full_argmax'])}
        result['less_argmax'] = {k: result['less_argmax'].get(k, 0) / counter \
                                 for k in set(result['less_argmax'])}

        result['full_argmax_sd'] = {k: round(math.sqrt((result['full_cumul_sq'].get(k, 0) \
                                    - pow(result['full_argmax'].get(k, 0), 2)*counter) \
                                    / (counter-1)), 4) for k in set(result['full_argmax_sd'])}
        result['less_argmax_sd'] = {k: round(math.sqrt((result['less_cumul_sq'].get(k, 0) \
                                    - pow(result['less_argmax'].get(k, 0), 2)*counter) \
                                    / (counter-1)), 4) for k in set(result['less_argmax_sd'])}
        result['full_argmax'] = {k: round(result['full_argmax'].get(k, 0), 4) \
                                 for k in set(result['full_argmax'])}
        result['less_argmax'] = {k: round(result['less_argmax'].get(k, 0), 4) \
                                 for k in set(result['less_argmax'])}
        del result['full_cumul_sq']
        del result['less_cumul_sq']
        with open(save_dir + '_' + result['name'] + '_performance.yaml', 'w') as file:
            documents = yaml.dump(result, file)
