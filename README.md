# Domain Generalization in Computer Vision - Group-42

## Team Members 
- Anuj Rajan Lalla (B22AI061)
- Vaibhav Gupta (B22CS058)
- Akarsh Katiyar (B22CS006)
- Ambati Rahul Reddy (B22CS088)
- Mayank Bansal (B22CS070)
    
## Project introduction 
In this project , we explored the impact of several pruning methods on domain generalization 

## List of Algorithms:
- ERM
- EQRM
- ERM++
- IRM
- URM
- HOG Based ERM

## List of pruning methods:
- Unstructured pruning
- Structured pruning
- Random pruning
- Sensitivity based pruning

We performed an exhaustive study on the PACS dataset . 

[The links to the Baseline models can be found here](https://drive.google.com/drive/folders/1QNcMcL89xYzU55Zqywqr9HOh6bNNZ6xF?usp=sharing) 

## Commands to reproduce
To download the dataset
```sh
python3 -m domainbed.scripts.download  --data_dir=./domainbed/data
```

To compute the HoG features for HoG Based ERM ,
```sh
python -m domainbed.scripts.convert_pacs_to_hog
```

To run HoG Based ERM
```sh
python -m domainbed.scripts.train  --dataset HOGPACS --algorithm PrecomputedHOGMLP  --data_dir=./domainbed/data/ --test_env 0 --hparams "{}" --output_dir plain/HOGPACS/0
```

For other algorithms
```sh
python3 -m domainbed.scripts.train --data_dir=./domainbed/data/ --algorithm EQRM --dataset PACS --test_env 0 --output_dir plain/EQRM/0
```
## Results

![Results_1](./results_1.jpg )  
**Figure 1:** Results for ERM , IRM, ERM++

![Results_2](./results_2.jpg )  
**Figure 2:** Results for URM, EQRM

## Acknowledgments

This project builds upon [DomainBed](https://github.com/facebookresearch/DomainBed) for domain generalization in computer vision. 

```bibtex
@misc{domainbed,
  title = {DomainBed: A PyTorch Suite for Domain Generalization},
  author = {Gulrajani, Ishaan and Lopez-Paz, David},
  howpublished = {\url{https://github.com/facebookresearch/DomainBed}},
  note = {Accessed: \today},
  year = {2021}
}
