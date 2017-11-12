"""
OAR Database
"""

import cv2
import os
import numpy as np
import cPickle
import PIL.Image as Image
from imdb import IMDB
import glob
from ..processing.bbox_transform import bbox_overlaps
import SimpleITK as sitk

def fast_hist(a, b, n):
    a = np.array(map(int, a)).astype(np.int32)
    b = np.array(map(int, b)).astype(np.int32)
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k] + b[k], minlength=n**2).reshape(n, n)

def get_hist(prediction, label, num_class):
    return fast_hist(label.flatten(), prediction.flatten(), num_class)

def per_class_dice(hist, num_classes, class_name):
    class_dice = []
    acc_total = np.diag(hist).sum() / hist.sum()
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    for ii in range(num_classes):
        jaccard = np.diag(hist)[ii] / (hist.sum(1)[ii] + hist.sum(0)[ii] - np.diag(hist)[ii])
        dice = 2.0 * jaccard / (1+ jaccard)
        class_dice.append(dice)
    return class_dice

def writeLabelImage(image, filename):
    """ store label data to colored image """
    BrainStem = [128,128,128]
    Chiasm = [128,0,0]
    Cochlea = [192,192,128]
    Eye = [128,64,128]
    InnerEars = [128,128,0]
    Larynx = [64,64,128]
    Lens = [64,0,128]
    OpticNerve = [0,128,192]
    SpinalCord = [255,255,255]
    Background = [0,0,0]
    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array([
        Background, BrainStem, Chiasm, Cochlea, Eye, InnerEars, Larynx, Lens, OpticNerve,
        SpinalCord
    ])
    for l in range(0,len(label_colours)):
        r[image==l] = label_colours[l,0]
        g[image==l] = label_colours[l,1]
        b[image==l] = label_colours[l,2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:,:,0] = r/1.0
    rgb[:,:,1] = g/1.0
    rgb[:,:,2] = b/1.0
    im = Image.fromarray(np.uint8(rgb))
    im.save(filename)

class OAR(IMDB):
    def __init__(self, image_set, root_path, dataset_path):
        """
        fill basic information to initialize imdb
        :param image_set: train or val or trainval or test
        :param root_path: 'cache' and 'rpn_data'
        :param dataset_path: data and results
        :return: imdb object
        """
        super(OAR, self).__init__('oar', image_set, root_path, dataset_path)
        self.image_set = image_set
        self.root_path = root_path
        self.data_path = dataset_path
        self.classes = ['__background__', 'Brain Stem', 'Chiasm', 'Cochlea', 'Eye',
                        'Inner Ears', 'Larynx', 'Lens',
                        'Optic Nerve', 'Spinal Cord']
        # self.classes = ['__background__', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
        self.class_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.num_classes = len(self.classes)
        self.patients, self.image_set_index = self.load_patient_ids()
        # self.image_set_index = self.patients
        self.num_images = len(self.image_set_index)
        print 'num_images', self.num_images

    def load_patient_ids(self):
        """
        Load patient ids in train, val, test file
        """
        patient_ids_file = os.path.join(self.data_path, 'patients', self.image_set + '.lst')
        assert os.path.exists(patient_ids_file), 'Path does not exist: {}'.format(patient_ids_file)
        patient_ids = []
        total_images = []
        with open(patient_ids_file, 'r') as f:
            for line in f:
                label = line.strip()
                #prefix = os.path.join("/data/dataset/oar/ct_segmentation_data_2", self.image_set, label, "/*.pkl")
                prefix = "/data/dataset/oar/ct_segmentation_data_2/" + self.image_set + "/" + label + "/*.pkl"
                total_images += glob.glob(prefix)
                patient_ids.append(label)

        return patient_ids, total_images

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        image_set_index_file = os.path.join(self.data_path, 'imglists', self.image_set + '.lst')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        image_set_index = []
        with open(image_set_index_file, 'r') as f:
            for line in f:
                if len(line) > 1:
                    label = line.strip().split('\t')
                    image_set_index.append(label[1])
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.data_path, index)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'ins_id', 'ins_seg', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                roidb = cPickle.load(f)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        gt_roidb = self.load_OAR_annotations()
        with open(cache_file, 'wb') as f:
            cPickle.dump(gt_roidb, f, cPickle.HIGHEST_PROTOCOL)

        return gt_roidb
    
    def load_from_OAR_seg(self, image_path):
        annotation_file = image_path
        seg_gt = annotation_file
        boxes = []
        gt_classes = []
        ins_id = []
        gt_overlaps = []
        """
        finalData = {
            'id': image_id,
            'image': imData['pixel_array'],
            'height': imData['columns'],
            'width': imData['rows'],
            'num_instance': len(bboxes),
            'bboxes': bboxes, # un-normalized [ymin, xmin, ymax, xmax]
            'masks': instance_masks, 
            'mask': mask,
            'label': labels,
        }
        """
        with open(annotation_file, 'rb') as f:
            _file = cPickle.load(f)
            height = _file['height']
            width = _file['width']
            image = image_path
            boxes = []
            _boxes = _file['bboxes'] # N * 4
            for bb in _boxes:
                ymin, xmin, ymax, xmax = bb
                boxes.append([xmin, ymin, xmax, ymax])
            # need xmin, ymin, xmax, ymax
            gt_classes = _file['label']
            instance_num = np.zeros(self.num_classes) # num of label 
            for index, la in enumerate(gt_classes):
                overlaps = np.zeros(self.num_classes)
                overlaps[la] = 1
                gt_overlaps.append(overlaps)
                ins_id.append(instance_num[la])
                # wtf ins_id ?
                instance_num[la] += 1

        return np.asarray(boxes), np.asarray(gt_classes), np.asarray(ins_id), seg_gt, np.asarray(gt_overlaps), image, height, width

    def load_from_seg(self, ins_seg_path):
        seg_gt = os.path.join(self.data_path, ins_seg_path)
        print seg_gt
        assert os.path.exists(seg_gt), 'Path does not exist: {}'.format(seg_gt)
        im = Image.open(seg_gt)
        pixel = list(im.getdata())
        pixel = np.array(pixel).reshape([im.size[1], im.size[0]])
        print im.size
        boxes = []
        gt_classes = []
        ins_id = []
        gt_overlaps = []
        for c in range(1, len(self.class_id)):
            px = np.where((pixel >= self.class_id[c] * 1000) & (pixel < (self.class_id[c] + 1) * 1000))
            # ex: label range => 26000 < [26013] < 27000, get all label
            if len(px[0]) == 0:
                continue
            ids = np.unique(pixel[px])
            for id in ids:
                px = np.where(pixel == id)
                x_min = np.min(px[1])
                y_min = np.min(px[0])
                x_max = np.max(px[1])
                y_max = np.max(px[0])
                if x_max - x_min <= 1 or y_max - y_min <= 1:
                    continue
                boxes.append([x_min, y_min, x_max, y_max])
                gt_classes.append(c)
                ins_id.append(id % 1000)
                overlaps = np.zeros(self.num_classes) # one-hot
                overlaps[c] = 1
                gt_overlaps.append(overlaps)
        return np.asarray(boxes), np.asarray(gt_classes), np.asarray(ins_id), seg_gt, np.asarray(gt_overlaps)
    
    def load_OAR_annotations(self):
        # Need to estbalish data format
        imgfiles_list = []
        # dataset path = "/data/dataset/oar/{pat_id}/slice_0.pkl...etc"
        prefix = "/data/dataset/oar/ct_segmentation_data_2/" + self.image_set
        for _id in self.patients:
            patient_folder = os.path.join(prefix, _id)
            print("folder ", patient_folder)
            slice_list = glob.glob(patient_folder+"/*.pkl")
            for _slice in slice_list:
                file_list = dict()
                file_list['img_id'] = _slice.split("/")[-1].split(".")[0]
                file_list['img_path'] = _slice
                file_list['ins_seg_path'] = _slice
                imgfiles_list.append(file_list)
        roidb = []
        for im in range(len(imgfiles_list)):
            print '===============================', imgfiles_list[im]['img_id'], '====================================='
            roi_rec = dict()
            ### Need config
            #roi_rec['image'] = os.path.join(self.data_path, imgfiles_list[im]['img_path'])
            #size = cv2.imread(roi_rec['image']).shape
            #roi_rec['height'] = size[0]
            #roi_rec['width'] = size[1]
            ### Need config
            boxes, gt_classes, ins_id, pixel, gt_overlaps, image, height, width = self.load_from_OAR_seg(imgfiles_list[im]['ins_seg_path'])
            if boxes.size == 0:
                total_num_objs = 0
                boxes = np.zeros((total_num_objs, 4), dtype=np.uint16)
                gt_overlaps = np.zeros((total_num_objs, self.num_classes), dtype=np.float32)
                gt_classes = np.zeros((total_num_objs, ), dtype=np.int32)
            roi_rec['image'] = image
            roi_rec['height'] = height
            roi_rec['width'] = width
            roi_rec['boxes'] = boxes
            roi_rec['gt_classes'] = gt_classes
            roi_rec['gt_overlaps'] = gt_overlaps
            roi_rec['ins_id'] = ins_id
            roi_rec['ins_seg'] = pixel
            roi_rec['max_classes'] = gt_overlaps.argmax(axis=1)
            roi_rec['max_overlaps'] = gt_overlaps.max(axis=1)
            roi_rec['flipped'] = False
            assert len(roi_rec) == 11
            roidb.append(roi_rec)
        return roidb

    def load_cityscape_annotations(self):
        """
        for a given index, load image and bounding boxes info from a single image list
        :return: list of record['boxes', 'gt_classes', 'ins_id', 'ins_seg', 'gt_overlaps', 'flipped']
        """
        imglist_file = os.path.join(self.data_path, 'imglists', self.image_set + '.lst')
        assert os.path.exists(imglist_file), 'Path does not exist: {}'.format(imglist_file)
        imgfiles_list = []
        with open(imglist_file, 'r') as f:
            for line in f:
                file_list = dict()
                label = line.strip().split('\t')
                file_list['img_id'] = label[0]
                file_list['img_path'] = label[1]
                file_list['ins_seg_path'] = label[2].replace('labelTrainIds', 'instanceIds')
                imgfiles_list.append(file_list)

        assert len(imgfiles_list) == self.num_images, 'number of boxes matrix must match number of images'
        roidb = []
        for im in range(self.num_images):
            print '===============================', im, '====================================='
            roi_rec = dict()
            roi_rec['image'] = os.path.join(self.data_path, imgfiles_list[im]['img_path'])
            size = cv2.imread(roi_rec['image']).shape
            roi_rec['height'] = size[0]
            roi_rec['width'] = size[1]
            boxes, gt_classes, ins_id, pixel, gt_overlaps = self.load_from_seg(imgfiles_list[im]['ins_seg_path'])
            if boxes.size == 0:
                total_num_objs = 0
                boxes = np.zeros((total_num_objs, 4), dtype=np.uint16)
                gt_overlaps = np.zeros((total_num_objs, self.num_classes), dtype=np.float32)
                gt_classes = np.zeros((total_num_objs, ), dtype=np.int32)
            roi_rec['boxes'] = boxes
            roi_rec['gt_classes'] = gt_classes
            roi_rec['gt_overlaps'] = gt_overlaps
            roi_rec['ins_id'] = ins_id
            roi_rec['ins_seg'] = pixel
            roi_rec['max_classes'] = gt_overlaps.argmax(axis=1)
            roi_rec['max_overlaps'] = gt_overlaps.max(axis=1)
            roi_rec['flipped'] = False
            assert len(roi_rec) == 11
            roidb.append(roi_rec)
        return roidb
    
    def append_flipped_images_oar(self, roidb):
        """
        append flipped images to an roidb
        flip boxes coordinates, images will be actually flipped when loading into network
        :param roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        print 'append flipped images to roidb'
        for i in range(self.num_images):
            roi_rec = roidb[i]
            boxes = roi_rec['boxes'].copy()
            if boxes.shape[0] != 0:
                oldx1 = boxes[:, 0].copy()
                oldx2 = boxes[:, 2].copy()
                boxes[:, 0] = roi_rec['width'] - oldx2 - 1
                boxes[:, 2] = roi_rec['width'] - oldx1 - 1

            entry = {'image': roi_rec['image'],
                     'height': roi_rec['height'],
                     'width': roi_rec['width'],
                     'boxes': boxes,
                     'gt_classes': roidb[i]['gt_classes'],
                     'gt_overlaps': roidb[i]['gt_overlaps'],
                     'max_classes': roidb[i]['max_classes'],
                     'max_overlaps': roidb[i]['max_overlaps'],
                     'ins_seg': roidb[i]['ins_seg'],
                     'ins_id': roidb[i]['ins_id'],
                     'flipped': True}
            roidb.append(entry)

        self.image_set_index *= 2
        return roidb

    def append_flipped_images(self, roidb):
        """
        append flipped images to an roidb
        flip boxes coordinates, images will be actually flipped when loading into network
        :param roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        print 'append flipped images to roidb'
        assert self.num_images == len(roidb)
        for i in range(self.num_images):
            roi_rec = roidb[i]
            boxes = roi_rec['boxes'].copy()
            if boxes.shape[0] != 0:
                oldx1 = boxes[:, 0].copy()
                oldx2 = boxes[:, 2].copy()
                boxes[:, 0] = roi_rec['width'] - oldx2 - 1
                boxes[:, 2] = roi_rec['width'] - oldx1 - 1
                assert (boxes[:, 2] >= boxes[:, 0]).all(),\
                    'img_name %s, width %d\n' % (roi_rec['image'], roi_rec['width']) + \
                    np.array_str(roi_rec['boxes'], precision=3, suppress_small=True)
            entry = {'image': roi_rec['image'],
                     'height': roi_rec['height'],
                     'width': roi_rec['width'],
                     'boxes': boxes,
                     'gt_classes': roidb[i]['gt_classes'],
                     'gt_overlaps': roidb[i]['gt_overlaps'],
                     'max_classes': roidb[i]['max_classes'],
                     'max_overlaps': roidb[i]['max_overlaps'],
                     'ins_seg': roidb[i]['ins_seg'],
                     'ins_id': roidb[i]['ins_id'],
                     'flipped': True}
            roidb.append(entry)

        self.image_set_index *= 2
        return roidb

    def evaluate_mask(self, results_pack):
        for result_rec in results_pack['results_list']:
            image_path = result_rec['image']
            im_info = result_rec['im_info']
            detections = result_rec['boxes']
            seg_masks = result_rec['masks']

            filename = image_path.split("/")[-1]
            filename = filename.replace('.png', '')

            result_path = 'data/oar/results/pred/'

            print 'writing results for: ', filename
            result_txt = os.path.join(result_path, filename)
            result_txt = result_txt + '.txt'
            count = 0
            f = open(result_txt, 'w')

            for j, labelID in enumerate(self.class_id):
                if labelID == 0:
                    continue
                dets = detections[j]
                masks = seg_masks[j]
                for i in range(len(dets)):
                    bbox = dets[i, :4]
                    score = dets[i, -1]
                    bbox = map(int, bbox)
                    mask_image = np.zeros((int(im_info[0, 0]), int(im_info[0, 1])))
                    mask = masks[i, :, :]
                    mask = cv2.resize(mask, (bbox[2] - bbox[0], (bbox[3] - bbox[1])), interpolation=cv2.INTER_LINEAR)
                    mask[mask > 0.5] = 200
                    mask[mask <= 0.5] = 0
                    mask_image[bbox[1]: bbox[3], bbox[0]: bbox[2]] = mask
                    cv2.imwrite(os.path.join(result_path, filename) + '_' + str(count) + '.png', mask_image)
                    f.write('{:s} {:s} {:.8f}\n'.format(filename + '_' + str(count) + '.png', str(labelID), score))
                    count += 1
            f.flush()
            f.close()

    def evaluate_mask_mean_iou(self, results_pack):
        prefix = "/data/dataset/oar/ct_segmentation_data_2/" + self.image_set
        gt_masks = {}
        pred_masks = {}
        case_hist = {}
        for _id in self.patients:
            patient_folder = os.path.join(prefix, _id)
            slice_list = glob.glob(patient_folder+"/*.pkl")
            for _slice in slice_list:
                image_id = _slice.split("/")[-1].split(".")[0]
                with open(_slice, 'rb') as f:
                    _file = cPickle.load(f)
                    im = cv2.resize(_file['mask'], None, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
                gt_masks[image_id] = cv2.resize(_file['mask'], None, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        
        nine_organs = {}
        for result_rec in results_pack['results_list']:
            image_path = result_rec['image']
            result_image_id = image_path.split("/")[-1].split(".")[0]
            im_info = result_rec['im_info']
            detections = result_rec['boxes']
            seg_masks = result_rec['masks']

            filename = image_path.split("/")[-1]
            filename = filename.replace('.png', '')

            result_path = 'data/oar/results/pred_semantic/'
            if not (os.path.exists(result_path)):
                os.makedirs(result_path)
            print 'writing results for: ', filename
            result_txt = os.path.join(result_path, filename)
            result_txt = result_txt + '.txt'
            count = 0
            # f = open(result_txt, 'w')
            whole_mask_image = np.zeros((int(im_info[0, 0]), int(im_info[0, 1])))
            for j, labelID in enumerate(self.class_id):
                if labelID == 0:
                    continue
                dets = detections[j]
                masks = seg_masks[j]
                mask_image = np.zeros((int(im_info[0, 0]), int(im_info[0, 1])))
                for i in range(len(dets)):
                    bbox = dets[i, :4]
                    score = dets[i, -1]
                    bbox = map(int, bbox)
                    # mask_image = np.zeros((int(im_info[0, 0]), int(im_info[0, 1])))
                    mask = masks[i, :, :]
                    mask = cv2.resize(mask, (bbox[2] - bbox[0], (bbox[3] - bbox[1])), interpolation=cv2.INTER_LINEAR)
                    mask[mask > 0.5] = 1
                    mask[mask <= 0.5] = 0
                    mask_image[bbox[1]: bbox[3], bbox[0]: bbox[2]] = mask
                    whole_mask_image[bbox[1]: bbox[3], bbox[0]: bbox[2]] = mask
                    #cv2.imwrite(os.path.join(result_path, filename) + '_' + str(count) + '.png', mask_image)
                    #f.write('{:s} {:s} {:.8f}\n'.format(filename + '_' + str(count) + '.png', str(labelID), score))
                    count += 1
                patient_id = result_image_id.split("_")[0]
                slice_num = result_image_id.split("_")[1]
                if patient_id in nine_organs:
                    if slice_num not in nine_organs[patient_id]:
                        nine_organs[patient_id][slice_num] = {}
                    nine_organs[patient_id][slice_num][self.classes[labelID]] = mask_image
                else:
                    nine_organs[patient_id] = {}
                    nine_organs[patient_id][slice_num] = {}
                    nine_organs[patient_id][slice_num][self.classes[labelID]] = mask_image
                    # [122434_0]['Eye'] = mask
            # cv2.imwrite(os.path.join(result_path, filename) + '.png', mask_image)
            pred_masks[result_image_id] = whole_mask_image
            # writeLabelImage(mask_image, os.path.join(result_path, filename) + '.png')
        assert len(gt_masks) == len(pred_masks)
        # write nine label binary nii
        for patient_id in nine_organs.keys():
            print("now processing {} ...".format(patient_id))
            for cls in self.classes:
                _s = []
                print("procssing {} organ".format(cls))
                for slice_id in range(len(nine_organs[patient_id])):
                    if cls in nine_organs[patient_id][str(slice_id)]:
                        _slice = cv2.resize(nine_organs[patient_id][str(slice_id)][cls], None, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
                    else:
                        _slice = np.zeros((200, 200))
                    _s.append(_slice)
                _s = np.array(_s)
                # n*200*200
                print(_s.shape)
                dir_path = "/data/mx-maskrcnn/eval_files/"+patient_id
                filename = "/data/mx-maskrcnn/eval_files/"+patient_id+"/"+cls+".nii"
                if not (os.path.exists(dir_path)):
                    os.makedirs(dir_path)
                sitk.WriteImage(sitk.GetImageFromArray(_s), filename)
        """
        for k in gt_masks.keys():
            group_name = k.split("_")[0]
            label = gt_masks[k]
            pred = pred_masks[k]
            if group_name in case_hist:
                hist, slice_id = case_hist[group_name]
                slice_id = slice_id
            else:
                hist = np.zeros((len(self.class_id), len(self.class_id)))
                slice_id = []
            hist += get_hist(pred, label, len(self.class_id))
            slice_id.append(k.split("_")[1])
            case_hist[group_name] = (hist, slice_id)
        class_num = np.array([10,9,10,6,10,9,9,9,10,9]).astype(np.float32)
        class_dice = np.zeros((10))
        np.set_printoptions(suppress=True)
        for c in case_hist.keys():
            hist, _ids = case_hist[c]
            print(len(_ids))
            class_dice += per_class_dice(hist, len(self.class_id), self.classes)
        print("class_num", class_num)
        print("class_dice", class_dice)
        for _id, _cls in enumerate(self.classes):
            print("{} dice: {}".format(_cls, class_dice[_id] / class_num[_id]))
        """
    def evaluate_detections(self, detections):
        raise NotImplementedError

    def create_roidb_from_box_list(self, box_list, gt_roidb):
        """
        given ground truth, prepare roidb
        :param box_list: [image_index] ndarray of [box_index][x1, x2, y1, y2]
        :param gt_roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        assert len(box_list) == self.num_images, 'number of boxes matrix must match number of images'
        roidb = []
        for i in range(self.num_images):
            roi_rec = dict()
            roi_rec['image'] = gt_roidb[i]['image']
            roi_rec['height'] = gt_roidb[i]['height']
            roi_rec['width'] = gt_roidb[i]['width']

            boxes = box_list[i]
            if boxes.shape[1] == 5:
                boxes = boxes[:, :4]
            num_boxes = boxes.shape[0]
            overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)
            if gt_roidb is not None and gt_roidb[i]['boxes'].size > 0:
                gt_boxes = gt_roidb[i]['boxes']
                gt_classes = gt_roidb[i]['gt_classes']
                # n boxes and k gt_boxes => n * k overlap
                gt_overlaps = bbox_overlaps(boxes.astype(np.float), gt_boxes.astype(np.float))
                # for each box in n boxes, select only maximum overlap (must be greater than zero)
                argmaxes = gt_overlaps.argmax(axis=1)
                maxes = gt_overlaps.max(axis=1)
                I = np.where(maxes > 0)[0]
                overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            roi_rec.update({'boxes': boxes,
                            'gt_classes': np.zeros((num_boxes,), dtype=np.int32),
                            'gt_overlaps': overlaps,
                            'max_classes': overlaps.argmax(axis=1),
                            'max_overlaps': overlaps.max(axis=1),
                            'ins_seg': gt_roidb[i]['ins_seg'],
                            'ins_id': gt_roidb[i]['ins_id'],
                            'flipped': False})

            # background roi => background class
            zero_indexes = np.where(roi_rec['max_overlaps'] == 0)[0]
            assert all(roi_rec['max_classes'][zero_indexes] == 0)
            # foreground roi => foreground class
            nonzero_indexes = np.where(roi_rec['max_overlaps'] > 0)[0]
            assert all(roi_rec['max_classes'][nonzero_indexes] != 0)

            roidb.append(roi_rec)

        return roidb

    def rpn_roidb(self, gt_roidb, append_gt=False):
        """
        get rpn roidb and ground truth roidb
        :param gt_roidb: ground truth roidb
        :param append_gt: append ground truth
        :return: roidb of rpn
        """
        if append_gt:
            print 'appending ground truth annotations'
            rpn_roidb = self.load_rpn_roidb(gt_roidb)
            roidb = IMDB.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self.load_rpn_roidb(gt_roidb)
        return roidb