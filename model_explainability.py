import torch.nn as nn
from model import  Resnet50FPN,Wide_Resnet50_2,VGG16FPN,CountRegressor,weights_normal_init
from utils_ltce import MAPS, Scales, Transform,TransformTrain,extract_features, visualize_output_and_save
from PIL import Image
import os
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from os.path import exists,join
import random
import torch.optim as optim
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
import torchvision.ops.boxes as bops

import torch
import torch.nn.functional as F
from PIL import Image
import os
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import torchvision
from torchvision import models
from torchvision import transforms
from captum.attr import GradientShap
from  interpretability_captum import visualize_image_attr

import warnings
warnings.filterwarnings("ignore")


data_path = '/Users/alessandroquattrociocchi/Git/AML/Final_Project/data/'
output_dir = "./logsSave"
test_split = "val" #choices=["train", "test", "val"]
gpu = 0
learning_rate = 1e-5
data_path = data_path
anno_file = data_path + 'annotation_FSC147_384.json'
data_split_file = data_path + 'Train_Test_Val_FSC_147.json'
im_dir = data_path + 'images_384_VarV2'
gt_dir = data_path + 'gt_density_map_adaptive_384_VarV2'
pre_trained_backbone = 'resnet' #choices=[resnet,wide_resnet,vgg16]


## YOLO
import cv2
from collections import defaultdict

def YOLO_objects_count(image_path):

    detections = model(image_path)
    frame = cv2.imread(image_path)

    #frame = cv2.imread(image_path)
    #detections = model(frame[..., ::-1])
    results = detections.pandas().xyxy[0].to_dict(orient="records")
    for result in results:
        con = result['confidence']
        cs = result['class']
        x1 = int(result['xmin'])
        y1 = int(result['ymin'])
        x2 = int(result['xmax'])
        y2 = int(result['ymax'])
        return results


def count_class(results):

    dict_ = defaultdict(int)
    for i in results:
        dict_[i['name']] += 1
    num = max(dict_.values())

    return num


if pre_trained_backbone == 'vgg16':
    backbone = VGG16FPN()
elif pre_trained_backbone == 'resnet':
    backbone = Resnet50FPN()
elif pre_trained_backbone == 'wide_resnet':
    backbone = Wide_Resnet50_2()

if not exists(output_dir):
    os.mkdir(output_dir)

#setting id for each GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

if not exists(output_dir):
    os.mkdir(output_dir)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

#criterion = nn.MSELoss().cuda()
criterion = nn.MSELoss()

#backbone.cuda()
#backbone.eval()

model_path='/Users/alessandroquattrociocchi/Git/AML/Final_Project/data/pretrainedModels/FamNet_Save1.pth'
regressor = CountRegressor(6, pool='mean')
regressor.eval()
regressor.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
optimizer = optim.Adam(regressor.parameters(), lr = learning_rate)

with open(anno_file) as f:
    annotations = json.load(f)

with open(data_split_file) as f:
    data_split = json.load(f)

from collections import Counter

model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5l6', pretrained=True)

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




def interpretability(image_path,grad_iter=1):

    img = Image.open(image_path)

    transform = transforms.Compose([
    #transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor()
    ])

    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transformed_img = transform(img)
    input = transform_normalize(transformed_img)
    input = input.unsqueeze(0)

    model = models.resnet50(pretrained=True)
    model = model.eval()

    output = model(input)
    output= F.softmax(output,dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    pred_label_idx.squeeze_()

    default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                    [(0, '#ffffff'),
                                                    (0.25, '#000000'),
                                                    (1, '#000000')], N=256)


    gradient_shap = GradientShap(model)

    # Defining baseline distribution of images
    rand_img_dist = torch.cat([input * 0, input * 1])

    attributions_gs = gradient_shap.attribute(input,
                                            n_samples=50,
                                            stdevs=0.0001,
                                            baselines=rand_img_dist,
                                            target=pred_label_idx)
    return attributions_gs,default_cmap


def model_explainability(im_id,data=annotations,pre_trained_backbone='resnet',
                        yolo_flag = True, yolo_threshold = 3,
                        regressor_attention = True,gradient_attention = True,
                        im_dir=im_dir,data_path=data_path):
    print('interpreting...')
    anno = data[im_id]
    bboxes = anno['box_examples_coordinates']
    dots = np.array(anno['points'])

    image_path = '{}/{}'.format(im_dir, im_id)
    image = Image.open('{}/{}'.format(im_dir, im_id))
    image.load()
    image_to_viz = image
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
        detections = model_yolo(image_path)
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

        print('')
        #print('YOLO:', yolo_res)
        if yolo_res:
            print('Updating rects')
            for i in yolo_res:
                x1 = int(i[0])
                y1 = int(i[1])
                x2 = int(i[2])
                y2 = int(i[3])
                # Do whatever you want
                f2 = cv2.rectangle(f1, (x1, y1), (x2, y2),color=(0,255,0),thickness=3)

        if yolo_res:
            print('Updating rects')
            rects += yolo_res


    sample = {'image':image,'lines_boxes':rects,'gt_density':density}
    sample = TransformTrain(sample)
    #image, boxes,gt_density = sample['image'].cuda(), sample['boxes'].cuda(),sample['gt_density'].cuda()
    image, boxes,gt_density = sample['image'], sample['boxes'],sample['gt_density']

    if pre_trained_backbone == 'vgg16':
        backbone = VGG16FPN()
    elif pre_trained_backbone == 'resnet':
        backbone = Resnet50FPN()
    elif pre_trained_backbone == 'wide_resnet':
        backbone = Wide_Resnet50_2()

    with torch.no_grad():
        features = extract_features(backbone, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)
    features.requires_grad = True
    optimizer.zero_grad()
    output = regressor(features)

    if regressor_attention:
        density_matrix = output[0][0].detach().cpu().numpy()
    if gradient_attention:
        attr_grad,default_cmap = interpretability(image_path)
        normalized_attr_grad = visualize_image_attr(np.transpose(attr_grad.squeeze().cpu().detach().numpy(), (1,2,0)),
                                cmap=default_cmap,show_colorbar=True,
                                use_pyplot=False)

    pred_cnt = output.sum().item()
    gt_cnt = dots.shape[0]

    plt.rcParams["figure.figsize"] = [15, 7]
    plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots(2,2)

    ax[0,0].imshow(image_to_viz)
    ax[0,0].set_title("Original")
    ax[0,0].set_axis_off()

    ax[0,1].imshow(f2)
    ax[0,1].set_title("FamNet and YOLO Boxes")
    ax[0,1].set_axis_off()


    ax[1,0].imshow(density_matrix, cmap='hot', interpolation='nearest')
    ax[1,0].set_title("FamNet Regressor Attention")
    ax[1,0].set_axis_off()

    ax[1,1].imshow(normalized_attr_grad,cmap=default_cmap)
    ax[1,1].set_title("Gradient Attention")
    ax[1,1].set_axis_off()

    fig.set_title('Predicted: ', pred_cnt, '\n Ground Truth: ', gt_cnt)
    fig.tight_layout()
    plt.savefig(data_path+"img_test.png")
    plt.show()

if __name__ == "__main__":
    model_explainability(im_id='210.jpg')
