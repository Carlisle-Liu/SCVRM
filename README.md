<h1 align="center"> Self-Calibrating Vicinal Risk Minimisation for Model Calibration </h1>

<sub><sub>Interim Implementation (Will Be Updated Into Final Version)</sub></sub>
[<a href="https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Self-Calibrating_Vicinal_Risk_Minimisation_for_Model_Calibration_CVPR_2024_paper.html" target="_blank">Paper</a>] [<a href="#bibtex">BibTex</a>]


**[Abstract]** *Model calibration measuring the alignment between the prediction accuracy and model confidence is an important metric reflecting model trustworthiness. Existing dense binary classification methods without proper regularisation of model confidence are prone to being over-confident. To calibrate Deep Neural Networks (DNNs) we propose a Self-Calibrating Vicinal Risk Minimisation (SCVRM) that explores the vicinity space of labeled data where vicinal images that are farther away from labeled images adopt the groundtruth label with decreasing label confidence. We prove that in the logistic regression problem SCVRM can be seen as a Vicinal Risk Minimisation plus a regularisation term that penalises the over-confident predictions. In practical implementation SCVRM is approximated using Monte Carlo sampling that samples additional augmented training images and labels from the vicinal distributions. Experimental results demonstrate that SCVRM can significantly enhance model calibration for different dense classification tasks on both in-distribution and out-of-distribution data. Code is available at https://github.com/Carlisle-Liu/SCVRM.*


## Environment

- python 3.8.12
- cuda 11.3
- pytorch 1.11.0
- torchvision 0.12.0


## Prepare the Data
Download the training dataset: <a target="_blank" href="https://www.kaggle.com/datasets/balraj98/duts-saliency-detection-dataset">DUTS-TR</a> and the six SOD testing datasets: <a target="_blank" href="https://www.kaggle.com/datasets/balraj98/duts-saliency-detection-dataset">DUTS-TE</a> (same link as DUTS-TR), <a target="_blank" href="http://saliencydetection.net/dut-omron/">DUT-OMRON</a>, <a target="_blank" href="http://cbi.gatech.edu/salobj/">PASCAL-S</a>, <a target="_blank" href="https://www.elderlab.yorku.ca/resources/salient-objects-dataset-sod/">SOD</a>, <a target="_blank" href="https://i.cs.hku.hk/~yzyu/research/deep_saliency.html">HKU-IS</a> and <a target="_blank" href="https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html">ECSSD</a>. The 500 Out-of-Distribution texture images are selected from the <a target="_blank" href="https://www.robots.ox.ac.uk/~vgg/data/dtd/">Describable Texture Dataset (DTD)</a>. To extract the texture image dataset, DTD_Texture_500, run the following code after placing the DTD dataset under the "Dataset" directory:

```
python Dataset/Process_Texture_Dataset.py
```


## Train, Test and Evaluate
To train, test and evaluate the model consecutively, run the following line of code:
```
CUDA_VISIBLE_DEVICES=GOU_ID python main_SCVRM.py
```


## Pretrained Model
Pretrained model weight can be downloaded from the [<a target="_blank" href="https://drive.google.com/drive/folders/1KIJ6k-cnlCbu8Rxtrln-xbG5Qta444aw?usp=share_link">Google Drive</a>].


**The structure of dataset directory is illustrated as below:**
```
├── Dataset
│   ├── DUTS-TR-Train.txt
│   ├── DUTS-TR-Validation.txt
│   ├── Train
│   │   ├── DUTS-TR
│   │   │   ├── Image
│   │   │   ├── GT
│   ├── Test
│   │   ├── DUTS-TE
│   │   ├── DUT-OMRON
│   │   ├── PASCAL-S
│   │   ├── SOD
│   │   ├── HKU-IS
│   │   ├── ECSSD
│   │   ├── DTD_Texture_500
```
The subdirectory structure of the testing dataset follows that of the training dataset.







## <a name="bibtex">Citing SCVRM</a>

```BibTex
@InProceedings{Liu_2024_CVPR,
    author    = {Liu, Jiawei and Ye, Changkun and Cui, Ruikai and Barnes, Nick},
    title     = {Self-Calibrating Vicinal Risk Minimisation for Model Calibration},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {3335-3345}
}
```