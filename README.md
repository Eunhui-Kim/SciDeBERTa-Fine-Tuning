
# SciDeBERTa Fine-Tuning
This code is for testing SciDeBERTa PLM model usig multi-tasking framework, dygiepp.
The paper is <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9791256">SciDeBERTa:  Learning DeBERTa for Science and Technology Documents and Fine-tuning Information Extraction Tasks</a>

# About Continual Learning of SciDeBERTa Fine-Tuning
For continual learning of PLM, you could use two kinds of methods.
1. you could use hugging face library. - <a href="https://www.topbots.com/pretrain-transformers-models-in-pytorch/">Pretrain Transformers Models in PyTorch Using Hugging Face Transformers</a>
2. If you want to use deberta, you could use <a href="https://github.com/microsoft/DeBERTa">deberta code</a> which is open. They also open MLM(masked langauge model) example, too.

# introduction
This repository is related with the paper SciDeBERTa.
The main idea of optimizing fine-tuning is

 1. Overfitting mitigation: In order to alleviate the overfitting problem that occurs when the fine-tuning training dataset is small (less than 10,000 examples) when training the fine-tuning task, the upper layer of the training model is reconstructed as a general xavier Gaussian rather than a pre-learning parameter. Initialization is performed. In the case of the SciERC dataset tested in the paper, the learning performance was improved through re-initialization.


 2. Optimizer selection: This program includes the latest and best performing optimization algorithms RAdam and AdamP. For reference, as the optimization algorithm provided by AllenNLP, the algorithm that has been confirmed to have the best performance is AdamW. This is one of the algorithms that improved the performance since Adam. After Adam, in addition to AdamW, RAdam and AdamP show improved performance similar to AdamW. In particular, in the case of RAdam, it shows improved performance without configuring a separate warmup scheduler. Through the experiment, it was possible to confirm the improved performance of RAdam compared to AdamW on the SciERC dataset.
 
# our experimental Results for SciDeBERTa
This figure shows the SciDeBERTa model performance in the metric of F1 score in SciERC & Genia DataSet.
![performance](https://github.com/Eunhui-Kim/SciDeBERTa-Fine-Tuning/blob/main/Test%20Performance%20of%20SciDeBERTa.png)

 
# pre-requisite
Note that this code is tested only in the environment decribed below. Mismatched versions can also occasionally make correct execution.
 - Ubuntu kernel ver. 4.15.0-117-generic #118~16.04.1
 - Cuda 10.2
 - torch.__version__ '1.12.0+cu102'
 - python 3.8.1
 - allennlp.__version__ '1.1.0'
 
# Setting
 1) clone dygiepp github in the local path
```    
     git clone https://github.com/dwadden/dygiepp
```   
 2) create conda environment such as 
```    
     conda create -n create dygiepp python==3.8
```   
 3) activate conda environment 
```      
     conda activate dygiepp
```   
 4) install <a href="https://pytorch.org/get-started/locally/">torch</a> 
```     
     pip3 install torch torchvision torchaudio
```   
 5) install requirements in the dygiepp local path 
```     
     pip install -r requirements.txt
```   
# Testing BERT model with SciERC dataset 
  -  For Check Setting, you could first download SciERC dataset and test by using default BERT model.  
   To train a model for named entity recognition, relation extraction, and coreference resolution on the SciERC dataset:

  -  Download the data. From the top-level folder for this repo, 
```
      sh ./scripts/data/get_scierc.sh
```
   This will download the scierc dataset into a folder ./data/scierc
  - Train the model. 
``` 
      sh scripts/train.sh scierc
```

# Testing SciDeBERTa model 
  - You can download SciDeBERTa model from huggingface https://huggingface.co/model/scideberta
  - As described in the table and the paper SciDeBERTa, you could use two kinds of model scideberta-abs and scideberta-cs.
    - (scideberta-abs denotes scideberta model which is continually learned from deberta by s2orc abstracts datasets.)
    - (scideberta-cs denotes scideberta model which is continually learned from deberta by s2orc computer science datasets.)
  - To test scideberta model, you can use configuration files such as
    - training_config/template.deberta.libsonnet
    - training_config/scierc_radam_st_scideberta_re9-net-1.jsonnet
    >> This example is the case which you download the configuration files into the local path.
    >> configuration files : pytorch_model.bin, config.json, tokenizer.json, vocab.json, merges.txt
    
# Testing SciDeBERTa model with optimizing fine-tuning option
  1) BackUp the following installed files A(initializers.py) and B(optimizer.py), 
  
    A. $HOME/.local/lib/python3.8/site-packages/allennlp/nn/initializers.py 
    
    B. $HOME/.local/lib/python3.8/site-packages/allennlp/training/optimizer.py
    
  2) Substitute following files.
  
    A. Substitute A(initilizers.py) file with patch_code/initializers.py 
    
      >> you can select initializers.py as initializers_re1011_bert.py, initializers_re1011_roberta.py, and initializers_re1011_deberta.py according to the model  Bert, Roberta, and Deberta, respectively. 
         
    B. Substitute B(optimizers.py) file with patch_code/optimizers.py
    
  3) Option Setting in training config
 -  A. Task Setting : 
    -  To run the "ner" only task, open scierc.jsonnet in training_config 
       and set loss_weights to ner: 1.0 and relation, coref, and events to 0.0.
       Set target_task to “ner” such as target_task: “ner”. (refer to the file, ner_finetune_re10-11.json)
    -  Multi-task option setting is possible in the same way. Depending on the loss weight, the weight ratio of each task is different, 
       so the performance is different. In particular, it helps to improve ner performance when running ner and coref at the same time.
       (refer to training_config/scierc_radam_st_scideberta_re9-ner-1.jsonnet)
 -  B. Optimizer Setting: 
    -  To use optimizer radam, open sicerc.jsonnet in training_config and set optimizer to radam.
    -  (refer to the file,  scierc_ner_radam.jsonnet)
 -  C. RUN
 ```
      sh scripts/train.sh scierc    
 ```
 
 # Citation
 ``` latex
 @article{jeong2022scideberta,
  title={SciDeBERTa: Learning DeBERTa for Science and Technology Documents and Fine-tuning Information Extraction Tasks},
  author={Jeong, Yuna and Kim, Eunhui},
  journal={IEEE Access},
  year={2022},
  publisher={IEEE}
}
```   
