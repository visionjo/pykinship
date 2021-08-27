# <h1><a id="user-content-fiw-data-development-kit" class="anchor" aria-hidden="true" href="#fiw-data-development-kit"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>FIW Data Development Kit</h1>

## Introduction
Python tools for automatic kinship recognition in images and videos.Python tools for automatic kinship recognition in images and videos.

* This is the documentation of the visual kinship recognition toolbox and tools specific to the [FIW](https://web.northeastern.edu/smilelab/fiw/) dataset (i.e., FIW data development kit). If you want the Places-CNN models instead of the training data, please refer to the [FIW-models](https://github.com/jvison/pykin).

## Brief overview of contents:

- [Project Detail](#version)
- [Data Desciption](#Details-of-the-data)
- [Referencees](#tag2020)
- [License](#license)
- [To Do](#to-do)
- [License](#license)
- [Getting Involved](#getting-involved)


**In summary, the following items are available:**
    
* Version and contact information, download links, along with a brief description of the different download options.

* Overview of the API, its purpose, contents, and premiere features.

* Image data details for FIW-Standard and RFIW-Challenge.

    1. Image list and annotations
    2. Submission format
    3. Evaluation routines

* Overview of the FIW data development kit.

* List of action items (tentative; open to requests and PRs)


This repo serves as the main set of tools for kinship recognition effort, including the FIW database. Besides, the next section is detailed description of database (i.e., data and label) structure.

Please contact Joseph Robinson [robinson.jo@northeastern.edu](robinson.jo@northeastern.edu) for questions, comments, or bug reports.



Download data and learn more about it here [https://web.northeastern.edu/smilelab/fiw/](https://web.northeastern.edu/smilelab/fiw/).


# <a name="version">Version</a>
**0.1.0**
Created:    16 January 2020

## Download
FIW can be obtained from two primary locatiions: the main dataset (i.e., raw data, experimental splits, and more) [downloads page](https://web.northeastern.edu/smilelab/fiw/download.html), along with task-specific data splits on codalab (i.e., [Task 1](https://competitions.codalab.org/competitions/21843), [Task 2](https://competitions.codalab.org/competitions/22117), and [Task 3](https://competitions.codalab.org/competitions/22152)), which were at one time used for data challenge (i.e., [2020 RFIW](https://arxiv.org/pdf/2002.06303.pdf) in conjunction with the [IEEE FG Conference](https://fg2020.org/)). Oncce download, we suggest to decompress the files in the data to their own folder. 



## Families In the Wild (FIW) Data and Labels
This documentation describes FIW DB and (working) development kit. This is work in prgress (i.e., still to come are FIW-CNN models, updated benchmarks, more in README (this), and more).

Check out FIW [project page](https://web.northeastern.edu/smilelab/fiw/index.html)



## DB Contents and Structure
### Counts

![Task-1 Statistics](docs/task-1-counts.png)

![Task-2 Statistics](docs/task-2-counts.png)

![Task-3 Statistics](docs/task-3-counts.png)

#

### Download
Download [here](https://web.northeastern.edu/smilelab/fiw/download.html)


## <a name="Details-of-the-data">Details of the data</a>
[Papers on FIW](https://web.northeastern.edu/smilelab/fiw/publications.html) decribe the data collection processes and details; supplemental to this is the [FIW Data Card]("DatasheetForFiw/main.pdf") below. Note that the Latex source file for the datasheet could be borrowed as a tempalate for another dataset of similar structure. Check out [repo](https://github.com/visionjo/DatasheetForFIW/tree/master), as well as [DatasheetForFiw/main.pdf](DatasheetForFiw/main.pdf)

A more complete list of references can be found [here](https://web.northeastern.edu/smilelab/fiw/publications.html).

### Structure
* *FIW_PIDs.csv:* Photo lookup table. Each row is an image instance, containing the following fields:
  * *PID:* Photo ID
  * *Name:* Surname.firstName (root reference for given family)
  * *URL:* Photo URL on web
  * *Metadata:* Text caption for photo
  
* *FIW_FIDs.csv:* FID (family)/ Surname lookup table.
  * *FID:* Unique ID key assigned to each family.
  * *Surname:* Family Name corresponding to FID key.
  
* *FIW_RIDs.csv:* Relationship lookup table with keys [1-9] assigned to relationship types.

* *FIDs/*
  * *FID####/* Contains labels and cropped facial images for members of family (1-1000)
    * *MID#/:* Face images of family member with ID key <N>, i.e., MID #.
    * *F####.csv:* File containing member information of each family:
      * *relationships matrix:* representation of relationships.
      * *name:* First name of family member.
      * *gender:* gender of family member.
      
      
For example:

```
FID0001.csv
    
	MID     1     2     3     Name    Gender
	 1      0     4     5     name1     F
	 2      1     0     1     name2     F
	 3      5     4     0     name3     M
	
```

Here we have 3 family members, as listed under the MID column (far-left). Each MID reads acorss its row.


We can see that MID1 is related to MID2 by 4->1 (Parent->Sibling), which of course can be viewed as the inverse, i.e., MID2->MID1 is 1->4. It can also be seen that MID1 and MID3 are Spouses of one another, i.e., 5->5. And so on, and so forth.


<div>
<div class="paragraph">If you found our data and resources useful please cite our works.
</div>



## <a name="tag2020">2020</a>

<div class="ref">Joseph P. Robinson, Yu Yin, Zaid Khan, Mind Shao, Siyu Xia, Michael Stopa, Samson Timoner, Matthew A. Turk, Rama Chellappa, and Yun Fu  
<a href=https://arxiv.org/abs/2002.06303>Recognizing Families In the Wild (RFIW): The 4th Edition</a> 
<i>IEEE International Conference on Automatic Face & Gesture Recognition</i>
<div class="links"><a onclick="if (document.getElementById(&quot;BIBrfiw2020&quot;).style.display==&quot;none&quot;) document.getElementById(&quot;BIBrfiw2020&quot;).style.display=&quot;block&quot;; else document.getElementById(&quot;BIBrfiw2020&quot;).style.display=&quot;none&quot;;"><font color="red">Bibtex</font> </a>| <a onclick="if (document.getElementById(&quot;ABSfiwpamiSI2018&quot;).style.display==&quot;none&quot;) document.getElementById(&quot;ABSfiwpamiSI2018&quot;).style.display=&quot;block&quot;; else document.getElementById(&quot;ABSfiwpamiSI2018&quot;).style.display=&quot;none&quot;;"><font color="red">Abstract</font> </a>| <a href=https://arxiv.org/pdf/2002.06303.pdf>PDF</a></div>

<div class="BibtexExpand" id="BIBrfiw2020" style="display: none;">

<pre class="bibtex">@article{robinson2020recognizing,
            title={Recognizing Families In the Wild (RFIW): The 4th Edition},
            author={Robinson, Joseph P and Yin, Yu and Khan, Zaid and Shao, Ming and Xia, Siyu and
                    Stopa, Michael and Timoner, Samson and Turk, Matthew A and Chellappa, Rama and Fu, Yun},
            journal={arXiv preprint arXiv:2002.06303},
            year={2020}
            }
          </pre>

</div>

<div class="AbstractExpand" id="ABSfiwpamiSI2018" style="display: none;">Recognizing Families In the Wild (RFIW): an annual large-scale, multi-track automatic kinship recognition evaluation that supports various visual kin-based problems on scales much higher than ever before. Organized in conjunction with the 15th IEEE International Conference on Automatic Face and Gesture Recognition (FG) as a Challenge, RFIW provides a platform for publishing original work and the gathering of experts for a discussion of the next steps. This paper summarizes the supported tasks (i.e., kinship verification, tri-subject verification, and search & retrieval of missing children) in the evaluation protocols, which include the practical motivation, technical background, data splits, metrics, and benchmark results. Furthermore, top submissions (i.e., leader-board stats) are listed and reviewed as a high-level analysis on the state of the problem. In the end, the purpose of this paper is to describe the 2020 RFIW challenge, end-to-end, along with forecasts in promising future directions.</div>

</div>

## <a name="tag2018">2018</a>

<div class="ref">Joseph P. Robinson, Ming Shao, Yue Wu, Hongfu Liu, Timothy Gillis, and Yun Fu <a href=https://web.northeastern.edu/smilelab/fiw/papers/tpami-final.pdf>Visual Kinship Recognition of Families in the Wild</a> 
<i>IEEE International Conference on Automatic Face & Gesture Recognition</i>
<div class="links"><a onclick="if (document.getElementById(&quot;BIBfiwpamiSI2018&quot;).style.display==&quot;none&quot;) document.getElementById(&quot;BIBfiwpamiSI2018&quot;).style.display=&quot;block&quot;; else document.getElementById(&quot;BIBfiwpamiSI2018&quot;).style.display=&quot;none&quot;;"><font color="red">Bibtex</font> </a>| <a onclick="if (document.getElementById(&quot;ABSfiwpamiSI2018&quot;).style.display==&quot;none&quot;) document.getElementById(&quot;ABSfiwpamiSI2018&quot;).style.display=&quot;block&quot;; else document.getElementById(&quot;ABSfiwpamiSI2018&quot;).style.display=&quot;none&quot;;"><font color="red">Abstract</font> </a>| <a href=https://arxiv.org/pdf/2002.06303.pdf>PDF</a></div>

<div class="BibtexExpand" id="BIBfiwpamiSI2018" style="display: none;">

<pre class="bibtex">@article{robinson2018visulkinship,
	title={Visual Kinship Recognition of Families in the Wild},
	author={Robinson, Joseph P and Shao, Ming and Wu, Yue and Liu, Hongfu and Gillis, Timothy and Fu, Yun},
	journal={IEEE Transactions on pattern analysis and machine intelligence (TPAMI) Special Issue: Computational Face},
	year={2020}
	}
  </pre>
</div>

<div class="AbstractExpand" id="ABSfiwpamiSI2018" style="display: none;">
We present the largest database for visual kinship recognition, _Families In the Wild_ (FIW), with over 13,000 family photos of 1,000 family trees with 4-to-38 members. It took only a small team to build FIW with efficient labeling tools and work-flow. To extend FIW, we further improved upon this process with a novel semi-automatic labeling scheme that used annotated faces and unlabeled text metadata to discover labels, which were then used, along with existing FIW data, for the proposed clustering algorithm that generated label proposals for all newly added data-- both processes are shared and compared in depth, showing great savings in time and human input required. Essentially, the clustering algorithm proposed is semi-supervised and uses labeled data to produce more accurate clusters. We statistically compare FIW to related datasets, which unarguably shows enormous gains in overall size and amount of information encapsulated in the labels. We benchmark two tasks, kinship verification and family classification, at scales incomparably larger than ever before. Pre-trained CNN models fine-tuned on FIW outscores other conventional methods and achieved state-of-the-art on the renowned KinWild datasets. We also measure human performance on kinship recognition and compare to a fine-tuned CNN.
</div>

</div>




## <a name="tag2016">2016</a>

<div class="ref">Joseph P. Robinson, Ming Shao, Yue Wu, Yun Fu<a href=https://web.northeastern.edu/smilelab/fiw/papers/tpami-final.pdf>Families in the Wild (FIW): Large-scale Kinship Image Database and Benchmarks</a> 
<i>IEEE International Conference on Automatic Face & Gesture Recognition</i>
<div class="links"><a onclick="if (document.getElementById(&quot; BIBrobinson2016families&quot;).style.display==&quot;none&quot;) document.getElementById(&quot; BIBrobinson2016families&quot;).style.display=&quot;block&quot;; else document.getElementById(&quot; BIBrobinson2016families&quot;).style.display=&quot;none&quot;;"><font color="red">Bibtex</font> </a>| <a onclick="if (document.getElementById(&quot; BIBrobinson2016families&quot;).style.display==&quot;none&quot;) document.getElementById(&quot; BIBrobinson2016families&quot;).style.display=&quot;block&quot;; else document.getElementById(&quot; BIBrobinson2016families&quot;).style.display=&quot;none&quot;;"><font color="red">Abstract</font> </a>| <a href=https://web.northeastern.edu/smilelab/fiw/papers/acm-mm-short-final.pdf>PDF</a></div>

<div class="BibtexExpand" id="BIBrobinson2016families" style="display: none;">

<pre class="bibtex">
@article{robinson2016families,
	title="Visual Kinship Recognition of Families in the Wild",
	author="Robinson, Joseph P and Shao, Ming and Wu, Yue and Liu, Hongfu and Gillis, Timothy and Fu, Yun",
	journal="ACM on Multimedia Conference",
	year="2016"
}
  </pre>
</div>

<div class="AbstractExpand" id="ABSfiwpamiSI2018" style="display: none;">
We present the largest kinship recognition dataset to date, Families in the Wild (FIW). Motivated by the lack of a single, unified dataset for kinship recognition, we aim to provide a dataset that captivates the interest of the research community. With only a small team, we were able to collect, organize, and label over 10,000 family photos of 1,000 families with our annotation tool designed to mark complex hierarchical relationships and local label information in a quick and efficient manner. We include several benchmarks for two image-based tasks, kinship verification and family recognition. For this, we incorporate several visual features and metric learning methods as baselines. Also, we demonstrate that a pre-trained Convolutional Neural Network (CNN) as an off-the-shelf feature extractor outperforms the other fea- ture types. Then, results were further boosted by fine-tuning two deep CNNs on FIW data: (1) for kinship verification, a triplet loss function was learned on top of the network of pre-train weights; (2) for family recognition, a family-specific softmax classifier was added to the network.
</div>

</div>


# <a name="license">License</a>

By downloading the image data you agree to the following terms:
1. You will use the data only for non-commercial research and educational purposes.
2. You will NOT distribute the above images.
3. Northeastern University makes no representations or warranties regarding the data, including but not limited to warranties of non-infringement or fitness for a particular purpose.
4. You accept full responsibility for your use of the data and shall defend and indemnify Northeastern University, including its employees, officers and agents, against any and all claims arising from your use of the data, including but not limited to your use of any copies of copyrighted images that you may create from the data.

See Download links (and Terms and Conditions) [here](https://web.northeastern.edu/smilelab/fiw/download.html).


# Authors
* **Joseph Robinson** - [Github](https://github.com/visionjo) - [web](http://www.jrobsvision.com)
* **Zaid Khan** - [Github](https://github.com/codezakh)


# <a name="getting-involved">Getting Involved</a>
## Bugs and Issues
Please bring up any questions, comments, bugs, PRs, etc.

--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

