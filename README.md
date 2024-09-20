# Dual Prompt Tuning 
PyTorch implement of DPT: [《Prompt Tuning based Contrastive Learning for Hierarchical Text Classification》](https://aclanthology.org/2024.findings-acl.723/) (ACL 2024 Findings)


## Requirements
- Python >= 3.6
- torch >= 1.6.0
- transformers >= 4.11.0
- datasets
- numpy
- pandas
- scikit_learn
- openprompt
- tqdm

Install the dependencies with `pip install -r requirements.txt`.

## Data Prepare
### 1. Web-of-Science (WoS)  

- Download link: [Web-of-Science](https://data.mendeley.com/datasets/9rw3vkcfy4/6)  (**public**)
- preprocess
    ```
    cd data/WebOfScience
    python preprocess_wos.py
    python build_wos.py
    ```
    The preprocessed script codes are modified based on HPT's script (Thanks!!).


### 2. Blurb Genre Collection (BGC) 
- Download link: [Blurb Genre Collection](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html).  (**public**)
- Preprocess
    ```shell
    cd data/BGC
    python preprocess_bgc.py
    python build_bgc.py
    ```

### 3. RCV1-V2
- Download link: [RCV1-V2](https://github.com/ductri/reuters_loader), by signing an agreement. 
- Preprocess:
    Place rcv1.tar.xz and lyrl2004_tokens_train.dat (can be downloaded here) inside data/rcv1.
    ```
    cd data/rcv1
    python preprocess_rcv1.py
    python build_rcv1.py
    ```


### 4. NYTimes (NYT)
We download [The New York Times Annotated Corpus](https://catalog.ldc.upenn.edu/LDC2008T19) here. However now the download website shows that the resource does not exist: **This corpus is no longer available.**



## Pretrained-Model Modify
To avoid the side impact of the overlapping interaction between label names and text words, we assign an
unique fabricated symbol for each label, like "L0", and add these symbols to the vocabulary list of pretrained BERT model.

- Method: Directly modify the vocabulary vocab.txt by replacing the token of `[unused]` to identify each label and retain the label as a whole token.

Examples are provided in the `config_examples/` folder for reference and modification.

## Model Train
```
bash scripts/train_bgc.sh
```

## Model Infer
```
python src/inference.py
```


## References
- [1] Zihan Wang, Peiyi Wang, Tianyu Liu, Binghuai Lin, Yunbo Cao, Zhifang Sui, and Houfeng Wang. 2022b. HPT: Hierarchy-aware prompt tuning for hierarchical text classification. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing pages 3740–3751, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.


## Citation
If this was useful to you, please cite the paper:
```
@inproceedings{xiong-etal-2024-dual,
    title = "Dual Prompt Tuning based Contrastive Learning for Hierarchical Text Classification",
    author = "Xiong, Sishi  and
      Zhao, Yu  and
      Zhang, Jie  and
      Mengxiang, Li  and
      He, Zhongjiang  and
      Li, Xuelong  and
      Song, Shuangyong",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.723",
    pages = "12146--12158",
}
```