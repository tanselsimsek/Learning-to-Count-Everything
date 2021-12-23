# LEARNING TO COUNT EVERYTHING...BETTER

<p align="center">
<img src=https://github.com/qtt-alessandro/test_colab-/blob/main/sapienza_logo.jpg width="100"/>
 </p>
  
  <p align="center">
  <b>Advanced Machine Learning Final Project a.y 2021-2022<br />
La Sapienza University of Rome <br />
MSC IN DATA SCIENCE<b>  <br />
Daniel Jimenez, Juan Mata Naranjo, Alessandro Quattrociocchi, Tansel Simsek<b> <br />
</p>
  
  

## Abstract
Current models to count objects on imagesare often based on pre-trained models and density estima-tion, however they are still not close to optimal. Our pro-posal we first aim at reducing the gap between trainingand test error by introducing regularization techniquessuch Batch Normalization, Dropout and Data Augmenta-tion. In addition, to enhance the behaviour of the model,we proposed to use diverse ImageNet pre-trained mod-els (i.e. VGG16) as an alternative for ResNet50. As afinal novelty, we implemented an ensemble method bycombining ResNet with YOLO to produce a model thatoutperforms the current state-of-the-art work

  

  
  
  
  
<p align="center">
<img src="https://github.com/AMLSapienza/Final_Project/blob/main/data/img_show.png" width="500"/ >
</p>
   
### Folder Tree
```bash
 └── Final_Project
    ├── README.md
    ├── data
    │   ├── ImageClasses_FSC147.txt
    │   ├── Train_Test_Val_FSC_147.json
    │   ├── annotation_FSC147_384.json
    │   └── pretrainedModels
    │       └── FamNet_Save1.pth
    ├── funcs.py
    ├── interpretability_captum.py
    ├── main.ipynb
    ├── model.py
    ├── model_explainability.py
    └── utils_ltce.py
```

### References
- [Learning To Count Everything(Original Paper)](https://openaccess.thecvf.com/content/CVPR2021/papers/Ranjan_Learning_To_Count_Everything_CVPR_2021_paper.pdf)


