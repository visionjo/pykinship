pykin
==============================

Python tools for automatic kinship recognition in images and videos.

Download data and learn more about it here https://web.northeastern.edu/smilelab/fiw/.

**Version: 0.1.0**

Created:    16 January 2020

Author: Joseph Robinson

Email: robinson.jo@husky.neu.edu


------------
## Overview
This API serves as the main code-base for kinship effort with FIW database. In addition, below is detailed description of database (i.e., data and label) structure.

## Families In the Wild (FIW) Data and Labels
This documentation describes FIW DB and (working) development kit. This is work in prgress (i.e., still to come are FIW-CNN models, updated benchmarks, more in README (this), and more).

Check out FIW [project page](https://web.northeastern.edu/smilelab/fiw/index.html)

### Download
Download [here](https://web.northeastern.edu/smilelab/fiw/download.html)

### Details of the data
[Papers on FIW](https://web.northeastern.edu/smilelab/fiw/publications.html) decribe the data collection processes and details; supplemental to this is the [FIW Data Card]("DatasheetForFiw/main.pdf") below. A more complete list of references can be found [here](https://web.northeastern.edu/smilelab/fiw/publications.html)

## Reference

```
 @article{robinson2018fiw,
   title={},
   author={},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   year={2018},
   publisher={IEEE}
 }
 
 @InProceedings{kinFG2017,
  author       = "Wang, Shuyang and Robinson, Joseph P and Fu, Yun",
  title        = "Kinship Verification on Families In The Wild with Marginalized Denoising Metric Learning",
  booktitle    = "Automatic Face and Gesture Recognition (FG), 2017 12th IEEE International Conference and Workshops on",
  year         = "2017",
}

@InProceedings{robinson2016families,
  author       = "Robinson, Joseph P. and Shao, Ming and Wu, Yue and Fu, Yun",
  title        = "Families In the Wild (FIW): Large-Scale Kinship Image Database and Benchmarks",
  booktitle    = "Proceedings of the 2016 ACM on Multimedia Conference",
  pages        = "242--246",
  publisher    = "ACM",
  year         = "2016"
}

```

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   ├── experiments      <- Scripts to reproduce experiments
    │   │   
    │   ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │   │                         the creator's initials, and a short `-` delimited description, e.g.
    │   │                         `1.0-jqp-initial-data-exploration`.
    │   │    
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │           └── visualize.py    
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org

## To Do
### General
- [ ] Benchmark demos
- [ ] RFIW stats
  - [ ] Data stats
  - [ ] Numbers added to project for reference (e.g., in rfiw-tools)
  - [ ] Update data card accordingly
- [ ] Benchmark results
  - [ ] Verification results
  - [ ] Tri-Subject results
  - [ ] Search and retrieval results
- [ ] Generate sample submissions
- [ ] Data Augmentation

### Experiments (TO DO)
- [x] Verification
- [x] Tri-Subject
- [ ] Search and Retrieval


## License

By downloading the image data you agree to the following terms:
1. You will use the data only for non-commercial research and educational purposes.
1. You will NOT distribute the above images.
1. Northeastern University makes no representations or warranties regarding the data, including but not limited to warranties of non-infringement or fitness for a particular purpose.
1. You accept full responsibility for your use of the data and shall defend and indemnify Northeastern University, including its employees, officers and agents, against any and all claims arising from your use of the data, including but not limited to your use of any copies of copyrighted images that you may create from the data.

See Download links (and Terms and Conditions) [here](https://web.northeastern.edu/smilelab/fiw/download.html).


## Authors
* **Joseph Robinson** - [Github](https://github.com/visionjo) - [web](http://www.jrobsvision.com)
* **Zaid Khan** - [Github](https://github.com/codezakh)


## Bugs and Issues
Please bring up any questions, comments, bugs, PRs, etc.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

