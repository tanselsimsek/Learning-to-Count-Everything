import copy
from PIL import Image
import os
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from os.path import exists
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
import torchvision.ops.boxes as bops
from collections import Counter
from utils_ltce import *
from model import *
import random

def count_class(results):

    dict_ = defaultdict(int)
    for i in results:
        dict_[i['name']] += 1
    num = max(dict_.values())

    return num

def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

def intersects(box1, box2):

    box1 = torch.tensor([box1], dtype=torch.float)
    box2 = torch.tensor([box2], dtype=torch.float)
    iou = bops.box_iou(box1, box2).numpy()[0][0]
    return iou

def YOLO_boxes(results_yolo, annotations, threshold=4):
    """
    E.g. annotations = annotations['6.jpg']['box_examples_coordinates']
    This function takes as input the results of YOLO and returns a set of boxes which will then be used for the
    feature extraction part. In order to do this we will review a few things:
    1. Yolo needs to detect at least 3 or more objects. If less than this number of objects has been detected we remove this observation
    2. We keep only information regarding the most common object (which is typically the object we are trying to identify)
    3. We remove the boxes that are overlapping (we don't want to provide FamNet with two similar boxes)
    4. Keep only the boxes that are over a certain threshold level
        a. if threshold is int take top threshold
        b. if threshold is float take values above threshold
    5. We provide as output the boxes that fulfill all of the previous requirements
    """

    # Remove all classes that don't belong to the most common class
    list_ = []
    for i in results_yolo:
        list_.append(i['name'])
    
    if len(list_) == 0:
        return None

    name = most_frequent(list_)
    results_yolo = [i for i in results_yolo if i['name'] == name]

    # More than 3 objects detected by YOLO
    if len(results_yolo) < 3:
        return None

    # Check for overlaps 
    non_overlap = list(range(1, len(results_yolo)+1))
    overlap = []
    for example_box in annotations:
        list_original = [example_box[0][0], example_box[0][1], 
                        example_box[2][0], example_box[2][1]]
        for i, yolo_res in enumerate(results_yolo):
            list_yolo = [yolo_res['xmin'], yolo_res['ymin'],
                        yolo_res['xmax'], yolo_res['ymax']]
            if intersects(list_yolo, list_original) >= 0.10:
                overlap.append(i+1) 

    non_overlap = list(set(non_overlap).difference(set(overlap)))
    results_yolo = [results_yolo[i-1] for i in non_overlap]

    # Keep only the most probable boxes
    list_ = []
    for i in results_yolo:
        list_.append(i['confidence'])
    order = list(np.argsort(list_)[::-1])
    if isinstance(threshold, int):
        results_yolo = [results_yolo[i] for i in order[:threshold]]
    elif isinstance(threshold, float):
        order_float = list(np.sort(list_)[::-1])
        pos = len([i for i in order_float if i >= threshold])
        results_yolo = [results_yolo[i] for i in order[:pos]]

    # Return boxes as in the annotation file
    final_list = []
    for i in results_yolo:
        final_list.append([i['xmin'], i['ymin'], i['xmax'], i['ymax']])

    return final_list


def test(data, num_img, backbone_model, regressor, yolo_model, yolo_flag, yolo_threshold, annotations, 
         plot_flag=False, im_dir='data/images_384_VarV2', use_gpu=False, model_path='model.pth',
         adapt=True, gradient_steps=100, learning_rate=1e-7):

    weight_mincount = 1e-9
    weight_perturbation = 1e-4

    if use_gpu: backbone_model.cuda()
    backbone_model.eval()
    
    if use_gpu:
        regressor.load_state_dict(torch.load(model_path))
    else:
        regressor.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

    if use_gpu: regressor.cuda()
    regressor.eval()

    cnt = 0
    SAE = 0  # sum of absolute errors
    SSE = 0  # sum of square errors

    print("Testing")
    n_imgs = num_img
    im_ids = data[:n_imgs]

    pbar = tqdm(im_ids)
    for im_id in pbar:
        anno = annotations[im_id]
        bboxes = anno['box_examples_coordinates']
        dots = np.array(anno['points'])
        image_path = '{}/{}'.format(im_dir, im_id)

        rects = list()
        for bbox in bboxes:
            x1, y1 = bbox[0][0], bbox[0][1]
            x2, y2 = bbox[2][0], bbox[2][1]
            rects.append([y1, x1, y2, x2])

        if yolo_flag:
                
            detections = yolo_model(image_path)
            results_yolo = detections.pandas().xyxy[0].to_dict(orient="records")

            try:
                yolo_obj_cnt = count_class(results_yolo)
            except:
                print("Yolo Failed")
                yolo_obj_cnt=0

            for result in results_yolo:
                con = result['confidence']
                cs = result['class']
                x1 = int(result['xmin'])
                y1 = int(result['ymin'])
                x2 = int(result['xmax'])
                y2 = int(result['ymax'])

            yolo_res = YOLO_boxes(results_yolo, bboxes, threshold=yolo_threshold)

            frame_1 = cv2.imread(image_path)
            frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2RGB)
            
            frame_2 = cv2.imread(image_path)
            frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2RGB)

            for i in bboxes:
                x1 = i[0][0]
                y1 = i[0][1]
                x2 = i[2][0]
                y2 = i[2][1]
                # Do whatever you want
                f1 = cv2.rectangle(frame_1, (x1, y1), (x2, y2),color=(255,0,0),thickness=3)
            
            if plot_flag:
                plt.imshow(f1)
                plt.show()
                plt.close()

            print('')

            if yolo_res:
                for i in yolo_res:
                    x1 = int(i[0])
                    y1 = int(i[1])
                    x2 = int(i[2])
                    y2 = int(i[3])
                    # Do whatever you want
                    f2 = cv2.rectangle(frame_2, (x1, y1), (x2, y2),color=(0,255,0),thickness=3)
            
                if plot_flag:
                    plt.imshow(f2)
                    plt.show()
                    plt.close()

            if yolo_res:
                rects += yolo_res
        else:
            yolo_obj_cnt = 0

        image = Image.open('{}/{}'.format(im_dir, im_id))
        image_path = '{}/{}'.format(im_dir, im_id)
        image.load()
        image.show()
        sample = {'image': image, 'lines_boxes': rects}
        sample = Transform(sample)
        image, boxes = sample['image'], sample['boxes']

        if use_gpu:
            image = image.cuda()
            boxes = boxes.cuda()

        with torch.no_grad(): features = extract_features(backbone_model, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)

        if not adapt:
            with torch.no_grad(): output = regressor(features)
        else:
            features.required_grad = True
            adapted_regressor = copy.deepcopy(regressor)
            adapted_regressor.train()
            optimizer = optim.Adam(adapted_regressor.parameters(), lr=learning_rate)
            for step in range(0, gradient_steps):
                optimizer.zero_grad()
                output = adapted_regressor(features)
                lCount = weight_mincount * MincountLoss(output, boxes)
                lPerturbation = weight_perturbation * PerturbationLoss(output, boxes, sigma=8)
                Loss = lCount + lPerturbation
                # loss can become zero in some cases, where loss is a 0 valued scalar and not a tensor
                # So Perform gradient descent only for non zero cases
                if torch.is_tensor(Loss):
                    Loss.backward()
                    optimizer.step() 
            features.required_grad = False
            output = adapted_regressor(features)


        gt_cnt = dots.shape[0]
        pred_cnt = output.sum().item()
        cnt = cnt + 1
        err = abs(gt_cnt - pred_cnt)
        SAE += err
        SSE += err**2

        pbar.set_description('{:<8}: actual-predicted: {:6d}, {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f}, YOLO: {:6.1f}'.\
                            format(im_id, gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), SAE/cnt, (SSE/cnt)**0.5, yolo_obj_cnt))
        print("")

    print('On test, MAE: {:6.2f}, RMSE: {:6.2f}'.format(SAE/cnt, (SSE/cnt)**0.5))



def train(data, backbone_model, regressor, optimizer, criterion, yolo_model, yolo_flag,
          yolo_threshold,n_img,shuffle_flag, annotations, plot_flag=False, im_dir='data/images_384_VarV2', 
          best_mae=1e7, best_rmse=1e7, gt_dir='gt_density_map_adaptive_384_VarV2'):

    print("Training on FSC147 train set data")
    im_ids = data[:n_img]
    if shuffle_flag:
        random.shuffle(im_ids)
    train_mae = 0
    train_rmse = 0
    train_loss = 0
    pbar = tqdm(im_ids)
    cnt = 0
    for im_id in pbar:
        cnt += 1
        anno = annotations[im_id]
        bboxes = anno['box_examples_coordinates']
        dots = np.array(anno['points'])

        image_path = '{}/{}'.format(im_dir, im_id)
        image = Image.open('{}/{}'.format(im_dir, im_id))
        image.load()
        if plot_flag:
            plt.imshow(image)
            plt.show()
            plt.close()
        density_path = gt_dir + '/' + im_id.split(".jpg")[0] + ".npy"
        density = np.load(density_path).astype('float32')   

        rects = list()

        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2]) 

        if yolo_flag:
            detections = yolo_model(image_path)
            results_yolo = detections.pandas().xyxy[0].to_dict(orient="records")

            try:
                yolo_obj_cnt = count_class(results_yolo)
            except:
                yolo_obj_cnt = 0

            for result in results_yolo:
                con = result['confidence']
                cs = result['class']
                x1 = int(result['xmin'])
                y1 = int(result['ymin'])
                x2 = int(result['xmax'])
                y2 = int(result['ymax'])

            yolo_res = YOLO_boxes(results_yolo, bboxes, threshold=yolo_threshold)

            frame_1 = cv2.imread(image_path)
            frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2RGB)
            
            frame_2 = cv2.imread(image_path)
            frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2RGB)

            for i in bboxes:
                x1 = i[0][0]
                y1 = i[0][1]
                x2 = i[2][0]
                y2 = i[2][1]
                # Do whatever you want
                f1 = cv2.rectangle(frame_1, (x1, y1), (x2, y2),color=(255,0,0),thickness=3)
            
            if plot_flag:
                plt.imshow(f1)
                plt.show()
                plt.close()

            if yolo_res:
                for i in yolo_res:
                    x1 = int(i[0])
                    y1 = int(i[1])
                    x2 = int(i[2])
                    y2 = int(i[3])
                    # Do whatever you want
                    f2 = cv2.rectangle(frame_2, (x1, y1), (x2, y2),color=(0,255,0),thickness=3)
            
                if plot_flag:
                    plt.imshow(f2)
                    plt.show()
                    plt.close()

            if yolo_res:
                rects += yolo_res
        else:
            yolo_obj_cnt

        sample = {'image':image,'lines_boxes':rects,'gt_density':density}
        sample = TransformTrain(sample)
        #image, boxes,gt_density = sample['image'].cuda(), sample['boxes'].cuda(),sample['gt_density'].cuda()
        image, boxes,gt_density = sample['image'], sample['boxes'],sample['gt_density']

        with torch.no_grad():
            features = extract_features(backbone_model, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)
        features.requires_grad = True
        optimizer.zero_grad()
        output = regressor(features)

        #if image size isn't divisible by 8, gt size is slightly different from output size
        if output.shape[2] != gt_density.shape[2] or output.shape[3] != gt_density.shape[3]:
            orig_count = gt_density.sum().detach().item()
            gt_density = F.interpolate(gt_density, size=(output.shape[2],output.shape[3]),mode='bilinear')
            new_count = gt_density.sum().detach().item()
            if new_count > 0: gt_density = gt_density * (orig_count / new_count)
        loss = criterion(output, gt_density)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred_cnt = torch.sum(output).item()
        gt_cnt = torch.sum(gt_density).item()
        cnt_err = abs(pred_cnt - gt_cnt)
        train_mae += cnt_err
        train_rmse += cnt_err ** 2

        pbar.set_description('actual:{:6.1f} -- predicted:{:6.1f} -- YOLO:{:6.1f} -- FAMNet error:{:6.1f} -- YOLO error:{:6.1f} -- Current MAE:{:5.2f} -- RMSE:{:5.2f} -- Best VAL MAE:{:5.2f} -- RMSE: {:5.2f}'.format( gt_cnt, pred_cnt, yolo_obj_cnt,abs(pred_cnt - gt_cnt),abs(yolo_obj_cnt - gt_cnt), train_mae/cnt, (train_rmse/cnt)**0.5,best_mae,best_rmse))
        print("")
    train_loss = train_loss / len(im_ids)
    train_mae = (train_mae / len(im_ids))
    train_rmse = (train_rmse / len(im_ids))**0.5
    return train_loss,train_mae,train_rmse


def eval(data, backbone_model, regressor, yolo_model, yolo_flag, yolo_threshold, 
         n_img, annotations, plot_flag=False, im_dir='data/images_384_VarV2'):
    cnt = 0
    SAE = 0 # sum of absolute errors
    SSE = 0 # sum of square errors

    print("Evaluation")
    im_ids = data[:n_img]
    pbar = tqdm(im_ids)
    for im_id in pbar:
        anno = annotations[im_id]
        bboxes = anno['box_examples_coordinates']
        dots = np.array(anno['points'])
        image_path = '{}/{}'.format(im_dir, im_id)

        rects = list()
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])
        
        if yolo_flag:
                
            detections = yolo_model(image_path)
            results_yolo = detections.pandas().xyxy[0].to_dict(orient="records")

            try:
                yolo_obj_cnt = count_class(results_yolo)
            except:
                yolo_obj_cnt = 0

            for result in results_yolo:
                con = result['confidence']
                cs = result['class']
                x1 = int(result['xmin'])
                y1 = int(result['ymin'])
                x2 = int(result['xmax'])
                y2 = int(result['ymax'])

            yolo_res = YOLO_boxes(results_yolo, bboxes, threshold=yolo_threshold)

            frame_1 = cv2.imread(image_path)
            frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2RGB)
            
            frame_2 = cv2.imread(image_path)
            frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2RGB)

            for i in bboxes:
                x1 = i[0][0]
                y1 = i[0][1]
                x2 = i[2][0]
                y2 = i[2][1]
                # Do whatever you want
                f1 = cv2.rectangle(frame_1, (x1, y1), (x2, y2),color=(255,0,0),thickness=3)
            
            if plot_flag:
                plt.imshow(f1)
                plt.show()
                plt.close()

            print('')

            if yolo_res:
                for i in yolo_res:
                    x1 = int(i[0])
                    y1 = int(i[1])
                    x2 = int(i[2])
                    y2 = int(i[3])
                    # Do whatever you want
                    f2 = cv2.rectangle(frame_2, (x1, y1), (x2, y2),color=(0,255,0),thickness=3)
            
                if plot_flag:
                    plt.imshow(f2)
                    plt.show()
                    plt.close()

            if yolo_res:
                rects += yolo_res
        else:
            yolo_obj_cnt = 0

        image = Image.open('{}/{}'.format(im_dir, im_id))
        image.load()
        sample = {'image':image,'lines_boxes':rects}
        sample = Transform(sample)
        sample['image'].shape
        #image, boxes = sample['image'].cuda(), sample['boxes'].cuda()
        image, boxes = sample['image'], sample['boxes']

        with torch.no_grad():
            output = regressor(extract_features(backbone_model, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales))

        gt_cnt = dots.shape[0]
        pred_cnt = output.sum().item()
        cnt = cnt + 1
        err = abs(gt_cnt - pred_cnt)
        SAE += err
        SSE += err**2

        pbar.set_description('{:<8}: actual-predicted: {:6d}, {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f}, YOLO: {:6.1f}'.format(im_id, gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), SAE/cnt, (SSE/cnt)**0.5, yolo_obj_cnt))
        print("")

    print('On evaluation data, MAE: {:6.2f}, RMSE: {:6.2f}'.format(SAE/cnt, (SSE/cnt)**0.5))
    return SAE/cnt, (SSE/cnt)**0.5


def run_train_phase(epochs, backbone_model, regressor, yolo_model, optimizer, criterion, data_train, shuffle, data_val, 
                    num_img_train, num_img_val, yolo_flag, yolo_threshold, plot_flag, annotations,
                    save='model.pth', im_dir='data/images_384_VarV2', gt_dir='gt_density_map_adaptive_384_VarV2'):

    best_mae, best_rmse = 1e7, 1e7
    stats = list()
    for epoch in range(0,epochs):
        regressor.train()
        train_loss,train_mae,train_rmse = train(data=data_train, backbone_model=backbone_model, yolo_model=yolo_model, yolo_flag = yolo_flag, 
                                                optimizer=optimizer, criterion=criterion, regressor=regressor, yolo_threshold = yolo_threshold,n_img = num_img_train, annotations=annotations,
                                                shuffle_flag=shuffle,plot_flag=plot_flag, im_dir=im_dir, best_mae=best_mae, best_rmse=best_rmse,
                                                gt_dir=gt_dir)
        regressor.eval()
        val_mae,val_rmse = eval(data=data_val, backbone_model=backbone_model, regressor=regressor, 
                                annotations=annotations, yolo_model=yolo_model, 
                                yolo_flag=yolo_flag, yolo_threshold=yolo_threshold, 
                                n_img=num_img_val, plot_flag=plot_flag, im_dir=im_dir)
        stats.append((train_loss, train_mae, train_rmse, val_mae, val_rmse))
        stats_file = "stats"+ ".txt"
        with open(stats_file, 'w') as f:
            for s in stats:
                f.write("%s\n" % ','.join([str(x) for x in s]))    
        if best_mae >= val_mae:
            best_mae = val_mae
            best_rmse = val_rmse
            if save:
                torch.save(regressor.state_dict(), save)

        print("Epoch {}, Avg. Epoch Loss: {} Train MAE: {} Train RMSE: {} Val MAE: {} Val RMSE: {} Best Val MAE: {} Best Val RMSE: {} ".format(
                epoch+1,  stats[-1][0], stats[-1][1], stats[-1][2], stats[-1][3], stats[-1][4], best_mae, best_rmse))