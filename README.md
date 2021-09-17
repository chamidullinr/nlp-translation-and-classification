# NLP Translation and Classification

The repository contains a method for classifying and cleaning text using NLP transformers.

## Overview
The input data are web-scraped product names gathered from various e-shops.
The products are either monitors or printers.
Each product in the dataset has a scraped name containing information
about the product brand, and product model name, but also unwanted noise - irrelevant information about the item.
Additionally, only some records are relevant, meaning that they belong to the correct category: monitor or printer,
while other records belong to unwanted categories like accessories or TVs.

The goal of the tasks is to preprocess web-scraped data by removing noisy records and cleaning product names.
Preliminary experiments showed that classic machine learning methods
like tf-idf vectorization and classification struggled to achieve good results.
Instead NLP transformers were employed:
* First, DistilBERT was utilized for removing irrelevant records. 
  The available data are monitors with annotated labels 
  where the records are classified into three classes: "Monitor", "TV", and "Noise".
* After, T5 was applied for cleaning product names by translating scraped name 
  into clean name containing only product brand and product model name. 
  For instance, for the given input "monitor led aoc 24g2e 24" ips 1080 ..."
  the desired output is "aoc | 24g2e".
  The available data are monitors and printers with annotated targets.

The datasets are split into training, validation and test sets without overlapping records.

The results and details about training and evaluation procedure can be found in the Jupyter Notebooks, 
see Content section below.

## Content
The repository contains Jupyter Notebooks for training and evaluating NNs:
* [01_data_exploration.ipynb](01_data_exploration.ipynb) -
  The notebook contains an exploration of the datasets for sequence classification and translation.
  It includes visualization of distributions of targets, and overview of available metadata.
* [02a_classification_fine_tuning.ipynb](02a_classification_fine_tuning.ipynb) -
  The notebook fine-tunes a DistilBERT classifier using training and validation sets,
  and saves the trained checkpoint.
* [02b_classification_evaluation.ipynb](02b_classification_evaluation.ipynb) -
  The notebook evaluates classification scores on the test set.
  It includes: a classification report with precision, recall and F1 scores; and a confusion matrix. 
* [03a_translation_fine_tuning.ipynb](03a_translation_fine_tuning.ipynb) -
  The notebook fine-tunes a T5 translation network using training and validation sets,
  and saves the trained checkpoint.
* [03b_translation_evaluation.ipynb](03b_translation_evaluation.ipynb) -
  The notebook evaluates translation metrics on the test set.
  The metrics are: Text Accuracy (exact match of target and predicted sequences);
  Levenshtein Score (normalized reversed Levenshtein Distance where 1 is the best and 0 is the worst);
  and Jaccard Index.
* [04_benchmarking.ipynb](04_benchmarking.ipynb) -
  The notebook evaluates GPU memory and time needed for running inference on DistilBERT and T5 models
  using various values of batch size and sequence length.


## Getting Started
### Package Dependencies
The method were developed using `Python=3.7` with `transformers=4.8` framework
that uses `PyTorch=1.9` machine learning framework on a backend.
Additionally, the repository requires packages:
`numpy`, `pandas`, `matplotlib` and `datasets`.

To install required packages with PyTorch for CPU run:
```bash
pip install -r requirements.txt
```

For PyTorch with GPU run:
```bash
pip install -r requirements_gpu.txt
```

The requirement files do not contain `jupyterlab` nor any other IDE.
To install `jupyterlab` run
```bash
pip install jupyterlab
```

## Contact
**Rail Chamidullin** - chamidullinr@gmail.com  - [Github account](https://github.com/chamidullinr)
