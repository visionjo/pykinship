# FIW-MM
mirror of [FIW-MM dataset - a large-scale kinship recognition dataset](https://web.northeastern.edu/smilelab/fiw/)

THIS IS WORK IN PROGRESS.
Our goal is to provide reproducible code for most all the steps in the building, preprocessing, and experimenting on FIW-MM [1].

----
FIW-MM contains over X utterances for Y subjects, extracted from videos uploaded to YouTube.

The dataset is gender balanced (?), with ??% of the speakers male.

The speakers span a wide range of different ethnic groups, accents, professions and ages.

There are no overlapping families between development and test sets.

|                      |**train**|**test**|
| -------------------- |:-------:| ------:|
| *# of families*      |    --   |   -    |
| *# of speakers*      |         |        |
| *# of videos*        |         |        |
| *# of utterances*    |         |        |


Nationality Distribution: The nationalities of the speakers in the dataset were obtained by crawling Wikipedia and can be found (@zaid, correct) [here]().

You can also view the distribution in the following graph:

@TODO

.. image:: ./data/v1/distribution.png


The train/val/test split used in [1] below for kinship recognition can be found [here](./data/v1/recognition_split.txt).

Models:
 - Pretrained models from dataset authors for VGGFIW - Kinship Identification and Verification [1] can be found [here](https://github.com/).


Notice:
> We are preparing an extended dataset (FIW-MM-2), containing up to double the number of families and many more speakers and videos.    
  FIW was originally released in 2016 as an image-based DB [2]. Then, in 2018, FIW was extended by great amounts of data and label fixes [3].

-------

Publications:

[1] Joseph P. Robinson, Zaid Khan, Ming Shao, Yun Fu - [Families In Wild Multimedia (FIW-MM): A Multi-Modal Database for Recognizing Kinship](<soon>) - ACM on Multimedia Conference, 2020.

[2] Joseph P. Robinson, Ming Shao, Yue Wu, Hongfu Liu, Timothy Gillis, Yun Fu - [Visual Kinship Recognition of Families in the Wild](./docs/papers/tpami-final.pdf) - IEEE Transactions on pattern analysis and machine intelligence (TPAMI), 2018.

[3] Joseph P. Robinson, Ming Shao, Yue Wu, Yun Fu - [Families in the Wild (FIW): Large-scale Kinship Image Database and Benchmarks](./docs/papers/acm-mm-short-final.pdf) - ACM on Multimedia Conference, 2016.

Several other works of ours are listed in the publication section of the FIW website [[link](https://web.northeastern.edu/smilelab/fiw/publications.html)].
