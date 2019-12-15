# HASOC2019

Code and analysis for submission to [HASOC2019](https://hasoc2019.github.io/index.html) titled [3Idiots at HASOC 2019: Fine-tuning Transformer Neural Networks for Hate Speech Identiﬁcation in Indo-European Languages](http://ceur-ws.org/Vol-2517/T3-4.pdf)

Competition task description and evaluation metrics https://hasoc2019.github.io/call_for_participation.html

Slides available at: [HASOC2019_presentation.pdf](./HASOC2019_presentation.pdf) - [HASOC2019_presentation.pptx](./HASOC2019_presentation.pptx)

Please cite our work as follows:

```
@inproceedings{Mishra2019HASOC,
title={{3Idiots at HASOC 2019: Fine-tuning Transformer Neural Networks for Hate Speech Identification in Indo-European Languages}},
author={Mishra, Shubhanshu and Mishra, Sudhanshu},
booktitle={Proceedings of the 11th annual meeting of the Forum for Information Retrieval Evaluation},
series = {FIRE 2019 Working Notes},
publisher = {CUER Workshop Proceedings},
year={2019},
Month={December},
pages={208--213}
}
```

Our code is part of the large Social Media Information Extraction project called [SocialMediaIE](https://socialmediaie.github.io/)

## Instructions

* Format of training data and distribution of labels provided in [notebooks/Training%20Data%20exploration.ipynb](https://github.com/socialmediaie/HASOC2019/blob/master/notebooks/Training%20Data%20exploration.ipynb)
* Try pytorch-transformers to use Bert, XLNet, and XLM to get document embedding and then do classification, with same scroring as before. - https://github.com/huggingface/pytorch-transformers
* English Model training - [notebooks/HASOC_English.ipynb](https://github.com/socialmediaie/HASOC2019/blob/master/notebooks/HASOC_English.ipynb)
* German Model training - [notebooks/HASOC_German.ipynb](https://github.com/socialmediaie/HASOC2019/blob/master/notebooks/HASOC_German.ipynb)
* Hindi Model training - [notebooks/HASOC_Hindi.ipynb](https://github.com/socialmediaie/HASOC2019/blob/master/notebooks/HASOC_Hindi.ipynb)
* Creation of run submissions - [notebooks/Create_run_submissions.ipynb](https://github.com/socialmediaie/HASOC2019/blob/master/notebooks/Create_run_submissions.ipynb)
* Paper tables and figures - [notebooks/System%20paper%20figures%20and%20tables.ipynb](notebooks/System%20paper%20figures%20and%20tables.ipynb)

## Contributors

* [Shubhanshu Mishra](https://github.com/napsternxg/)
* [Sudhanshu Mishra](https://github.com/ghostktjMactavish)


## Acknowledgements

* Our code relies on [pytorch-transformers library](https://github.com/huggingface/transformers)
