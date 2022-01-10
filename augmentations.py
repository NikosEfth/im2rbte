import random 
import PIL
import numpy as np
import torch
import cv2 as cv
import skimage.morphology
import skimage.filters
import skimage.measure
import skimage.transform

cv.setNumThreads(0)

def to_tuple(var, size=2):
    if isinstance(var, (int, float)):
        if size == 2:
            var = (var, var)
        elif size == 3:
            var = (var, var, var)
    elif isinstance(var, list):
        var = tuple(var)
    return var


def find_bounding_box(img):
    try:
        positive = np.where(img != 0)
        bbox = np.min(positive[0]), np.max(positive[0]), np.min(positive[1]), np.max(positive[1])
    except Exception:
        bbox = (0, img.shape[0], 0, img.shape[1])
    return bbox


def max_one(image):
    image = np.array(image)
    if np.amax(image) > 1: 
        image = image/255
    return image


def keep_first_channel(image):
    image = max_one(image)
    if len(image.shape) > 2: 
        image = image[:, :, 0]
    return image


def thresh_function(thresh_mode):
    if isinstance(thresh_mode, list):
        thresh_mode = [x.lower() for x in thresh_mode]
        if 'all' in thresh_mode or 'normal' in thresh_mode:
            thresh_mode = 'all'
        else:
            thresh_mode = thresh_mode[random.randint(0, len(thresh_mode)-1)]
    if isinstance(thresh_mode, str):
        try:
            thresh_mode = int(thresh_mode)
        except Exception:
            thresh_mode = thresh_mode.lower()
    if thresh_mode in ['normal', 'all', 'n', 'a']:
        thresh_mode_function = random.choice([skimage.filters.threshold_otsu, 
                                              skimage.filters.threshold_yen, 
                                              skimage.filters.threshold_mean, 
                                              skimage.filters.threshold_isodata, 
                                              skimage.filters.threshold_li])
    elif thresh_mode in ['yen', 'y']:
        thresh_mode_function = skimage.filters.threshold_yen
    elif thresh_mode in ['mean', 'm']:
        thresh_mode_function = skimage.filters.threshold_mean
    elif thresh_mode in ['isodata', 'i', 'id']:
        thresh_mode_function = skimage.filters.threshold_isodata
    elif thresh_mode in ['li', 'l']:
        thresh_mode_function = skimage.filters.threshold_li
    else:
        thresh_mode_function = skimage.filters.threshold_otsu
        if thresh_mode not in ['otsu', 'o']:
            print(thresh_mode, 'not a valid thresh mode, Otsu was used by default')
    return thresh_mode_function


class ToStackTensor:
    
    def __init__(self):
        self.to_tensor = ToTensor()

    def __call__(self, images):
        if isinstance(images, list):
            return torch.stack([self.to_tensor(image) for image in images])
        return self.to_tensor(images)


class ListTransform:
    
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, images):
        if isinstance(images, list):
            return [self.transform(image) for image in images]
        return self.transform(images)


class Square(object):

    def __call__(self, image):
        height, width = image.size
        size = max(height, width)
        new_im = PIL.Image.new('RGB', (size, size), (0, 0, 0))
        new_im.paste(image, (int((size - height) / 2), int((size - width) / 2)))
        return new_im


class SquareNP(object):

    def __call__(self, image):
        image = keep_first_channel(image)
        height, width = image.shape
        size = max(height, width)
        offset = ((size - height) // 2, (size - width) // 2)
        new_image = np.zeros((size, size))
        if np.sum(image) > np.sum(1-image):
            new_image = np.ones((size, size))
        new_image[offset[0]:offset[0]+height, offset[1]:offset[1]+width] = image
        return new_image


class ResizeNP(object):
    
    def __init__(self, size=224):
        self.size = size

    def __call__(self, image):
        image = keep_first_channel(image)
        height, width = image.shape
        if height >= width:
            new_image = skimage.transform.resize(image, (int(self.size), 
                                                         int((width/height)*self.size)))
        else:
            new_image = skimage.transform.resize(image, (int((height/width)*self.size), 
                                                         int(self.size)))
        return new_image


class Pad(object):
    
    def __init__(self, percent=(0, 12), color=(0, 0, 0)):

        self.percent = to_tuple(percent, 2)
        self.color = to_tuple(color, 3)

    def __call__(self, image):
        percent = random.randint(self.percent[0], self.percent[1])/100
        height, width = image.size

        new_height = int(height + percent*height)
        new_width = int(width + percent*width)
        new_im = PIL.Image.new('RGB', (new_height, new_width), self.color)
        new_im.paste(image, (int((new_height - height) / 2), int((new_width - width) / 2)))
        return new_im


class BlackBackground(object):
    def __call__(self, image):
        image = max_one(image)
        if np.sum(image) > np.sum(1-image):
            image = 1-image
        return image


class MultiScale(object):
    
    def __init__(self, size=224, size_multipliers=(0.9, 0.75, 0.5), return_white_bg=False):
        if isinstance(size_multipliers, tuple):
            size_multipliers = list(size_multipliers)
        elif isinstance(size_multipliers, (int, float)):
            size_multipliers = [size_multipliers]
        self.size = size
        self.size_multipliers = size_multipliers
        self.return_white_bg = return_white_bg

    def __call__(self, image):
        image = max_one(image)
        if np.sum(image) > np.sum(1-image):
            image = 1-image
        images = [image]
        if len(image.shape) > 2:
            images = [image[:, :, 0]] 

        for count, img in enumerate(images):
            height, width = img.shape
            img[:, :3], img[:, height-3:], img[:3, :], img[width-3:, :] = 0, 0, 0, 0
            max_value = np.amax(img)
            img[img < max_value*0.1] = 0
            box = find_bounding_box(img)
                
            img = img[box[0]:box[1], box[2]:box[3]]
            height, width = img.shape
            images[count] = img
        new_images = []
        for count, img in enumerate(images):
            for multi in self.size_multipliers:
                new_image = np.zeros((self.size, self.size))
                try:
                    height, width = img.shape
                    if height >= width:
                        tmpimg = skimage.transform.resize(img, 
                                                          (int(self.size*multi), 
                                                           int((width/height)*self.size*multi)))
                    else:
                        tmpimg = skimage.transform.resize(img, 
                                                          (int((height/width)*self.size*multi), 
                                                           int(self.size*multi)))
                except Exception:
                    print('one image was empty')
                    tmpimg = np.zeros((int(self.size*multi), int(self.size*multi)))
                height, width = tmpimg.shape
                offset = ((self.size - height) // 2, (self.size - width) // 2)
                new_image[offset[0]:offset[0]+height, offset[1]:offset[1]+width] = tmpimg
                if self.return_white_bg:
                    new_image = 1-new_image
                new_images.append(new_image)
        return new_images


class EdgeDetector(object):

    def __init__(self, edge_mode='normal'):
        self.edge_mode = edge_mode

    def __call__(self, image):
        image = max_one(image)
        if isinstance(self.edge_mode, list):
            edge_mode = [x.lower() for x in self.edge_mode]
            if 'all' in edge_mode or 'normal' in edge_mode or 'a' in edge_mode or 'n' in edge_mode:
                edge_mode = random.choice(['dollar', 'hed', 'bdcn'])
            else:
                edge_mode = edge_mode[random.randint(0, len(edge_mode)-1)]
        else:
            edge_mode = self.edge_mode.lower()
        if edge_mode in ['all', 'normal', 'a', 'n']:
            edge_mode = random.choice(['dollar', 'hed', 'bdcn'])
        if edge_mode in ['bdcn', 'b']:
            image = image[:, :, 0]
        elif edge_mode in ['hed', 'h']:
            image = image[:, :, 1]
        else:
            image = image[:, :, 2]
            if edge_mode not in ['dollar', 'dol', 'd']:
                print(self.edge_mode, 'not a valid edge mode, Dollar was used by default')

        image = np.dstack((image, image, image))
        return image


class OriNMS(object):
    def __init__(self, model=None, prob=100, radious=2, bound_radious=0, multi=1.0):
        self.model = model
        self.prob = prob
        self.bound_radious = bound_radious
        self.radious = radious
        self.multi = multi

    def __call__(self, image):
        image = keep_first_channel(image)
        if self.model is not None:
            edgemap_original = np.float32(np.array(image))
            orimap_original = self.model.computeOrientation(edgemap_original)
            if random.randint(1, 100) <= self.prob:
                edgemap_original = self.model.edgesNms(edgemap_original, orimap_original,
                                                       r=self.radious, s=self.bound_radious, 
                                                       m=self.multi)
            return np.dstack((edgemap_original, edgemap_original, edgemap_original))
        return np.dstack((image, image, image))


class Thresholder(object):
    def __init__(self, thresh_rand=10, thresh_mode='normal', hyst_par=(0.5, 1.5), hyst_pert=0.2, 
                 hyst_prob=100, thinning=False):
        self.thresh_mode = thresh_mode
        self.thresh_rand = thresh_rand
        self.hyst_par = to_tuple(hyst_par, 2)
        self.hyst_pert = hyst_pert
        self.hyst_prob = hyst_prob
        self.thinning = thinning

    def __call__(self, image):

        image = keep_first_channel(image)
        image = image*255
        
        if isinstance(self.thresh_mode, int):
            thresh = self.thresh_mode + random.normalvariate(0, self.thresh_rand/2)
            thresh = max(min(thresh, np.amax(image)), np.amin(image))
        else:
            try:
                exact_thresh = thresh_function(self.thresh_mode)(image)
            except Exception:
                exact_thresh = 0.5
            thresh = exact_thresh + random.normalvariate(0, self.thresh_rand/2)    
            if thresh >= np.amax(image) or thresh <= np.amin(image):
                thresh = exact_thresh 
        if random.randint(1, 100) <= self.hyst_prob:
            per = random.normalvariate(0, self.hyst_pert)
            lower, upper = [max(0.1, self.hyst_par[0] -  per), min(2, self.hyst_par[1] + per)]
            if lower > upper: 
                lower, upper = upper, lower 
            binary = skimage.filters.apply_hysteresis_threshold(image/255, lower*thresh/255,
                                                                upper*thresh/255)*1
            if np.amin(binary) == 1:
                binary = skimage.filters.apply_hysteresis_threshold(image/255,
                                                                    self.hyst_par[0]*thresh/255,
                                                                    self.hyst_par[1]*thresh/255)*1
            if self.thinning:
                binary = skimage.morphology.skeletonize(binary)
            
        else:
            binary = (image > thresh)*1
            if self.thinning:
                binary = skimage.morphology.skeletonize(binary)
        binary = np.dstack((binary, binary, binary))

        return binary


class Cleaner(object):
    def __init__(self, percent_of_cc=(80, 100), del_less_than=(0, 10)):

        self.percent_of_cc = to_tuple(percent_of_cc, 2)
        self.del_less_than = to_tuple(del_less_than, 2)

    def __call__(self, image):
        image = keep_first_channel(image)
        percent_of_cc = random.randint(self.percent_of_cc[0], self.percent_of_cc[1])
        del_less_than = random.randint(self.del_less_than[0], self.del_less_than[1])

        blobs_labels = skimage.measure.label(image > 0.5, background=0)
        unique, counts = np.unique(blobs_labels, return_counts=True)
        hist = dict(zip(unique, counts))
        hist[0] = 0

        colors = []
        counts = []
        counter = 0
        for component in sorted(hist, key=hist.get, reverse=True):
            if counter == 0:
                colors.append(component)
                counts.append(hist[component])
            elif (sum(counts)/sum(hist.values()) < percent_of_cc/100 
                  and hist[component] > del_less_than):
                colors.append(component)
                counts.append(hist[component])
            counter += 1

        outimage = np.zeros_like(blobs_labels)
        for i in colors:
            outimage = outimage + (blobs_labels == i)
        return np.dstack((outimage, outimage, outimage))


class ToTensor(object):
    def __call__(self, image):
        image = np.array(image)
        image = keep_first_channel(image)
        image = np.dstack((image, image, image))
        image = torch.from_numpy(image).type('torch.FloatTensor').permute(2, 0, 1)
        return image
