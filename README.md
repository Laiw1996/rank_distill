
# Ranking Distillation

A PyTorch implementation of ranking distillation (The paper is under reviewing)

# Requirements
* Python 2 or 3
* [PyTorch v4.0+ (build from source)](https://github.com/pytorch/pytorch)
* Numpy
* SciPy

# Usage
1. Install required packages.
2. run <code>python train_caser.py</code> to get the well-trained teacher model (you can also skip this step, as there is one in the checkpoint folder).
2. run <code>python teach_caser.py</code>

# Configurations

#### Model Args (in train_caser.py)

- <code>L</code>: length of sequence, set to 5 as default

- <code>T</code>: number of targets, set to 1 as default

- <code>d</code>: number of latent dimensions, set to 100 for teacher model and 50 for student

- for <code>nv</code>, <code>nh</code>, <code>ac_conv</code>, <code>ac_fc</code>, <code>drop_rate</code>, they are set according to the paper:

   *Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding, Jiaxi Tang and Ke Wang , WSDM '18*


#### Model Args (in teach_caser.py)

- <code>teacher_model_path</code>: path to the teacher model
- <code>teacher_topk_path</code>: path to the file with teacher's top-K ranking for each training query
- <code>teach_alpha</code>:  alpha for balance ranking loss and distillation loss
- <code>K</code>: teacher's top-K exemplary ranking size
- <code>lamda</code>: sharpness for static weighting
- <code>mu</code>: sharpness for dynamic weighting
- <code>dynamic_samples</code>: number of dynamic samples
- <code>dynamic_start_epoch</code>: number of epochs when the dynamic weighting starts



# Acknowledgement

This project is still under construction, and the full version will be released once the paper accepted.