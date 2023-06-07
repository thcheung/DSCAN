# DSCAN
The official code of paper, "Dual-Stream Cross-Attentive Network for Claim Veracity Assessment with Social and External Evidence".

## Abstract
Accessing the veracity of claims on social media is an important and challenging task in natural language processing and social media analytics. Most existing methods for rumour verification are based on social-context features, including user, propagation, and engagement features. However, the performance of assessing the veracity of out-of-context claims is still limited, especially for emerging events on social media. Motivated by recently proposed automatic fact-check systems using an Internet search, we propose to integrate external web retrieval results as complementary signals, together with the social response within social media, for rumour verification. To leverage the cross-evidence relationship between the social evidence and external evidence for feature representation, we propose a Dual-Stream Cross-Attentive Network (DSCAN), to fuse social response and external evidence, through a dual attention mechanism. The proposed DSCAN consists of two streams of intra-evidence attentive modules, one is used for social reply and the other for external evidence, followed by a cross-evidence attentive module. The attentive modules leverage both intra-evidence and cross-evidence correlations, to enhance the representational capability for veracity assessment. To facilitate the research of external evidence-based veracity assessment on social media, we have extended two real-world datasets, namely PHEME and RumourEval, by collecting relevant evidence with three different web search engines. Our experimental results show that the proposed DSCAN significantly boosts the performance of rumour verification on social media, evaluated on the two extended datasets. Codes and extended datasets will be publicly available for future research.

## Datasets

The datasets can be downloaded below.

- [PHEME Dataset](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078)

- [RumourEval Dataset](https://figshare.com/articles/dataset/RumourEval_2019_data/8845580)
