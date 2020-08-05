<p align="center"><a href="url"><img src="https://images.unsplash.com/photo-1500099817043-86d46000d58f?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=634&q=80" height="500" width="300" ></a></p>
<p align="center">Photo by <a href="https://unsplash.com/@greystorm?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText" target="_blank">Ian Espinosa</a> on <a href="https://unsplash.com/s/photos/sad?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText" target="_blank">Unsplash</a></p>

# Classifying Mental Health: The Reddit Proxy

Using the Pushshift API, I gathered approximately 135,000 Reddit posts to classify text into 4 possible groups: ADHD, anxiety, depression, and non-clinical.

A [Flask app](https://mental-health-classifier.herokuapp.com/) using the final model is also available to classify user-inputted text.

## Replication Instructions

1. Clone repo
2. Install package requirements in ```requirements.txt```
3. Process data with [cleaning](Notebooks/Cleaning.ipynb) notebook
4. Model ([modeling](Notebooks/Modeling.ipynb) notebook) <br>
(Optional) Run [webscraper](Notebooks/Scraping.ipynb) notebook

A test sample of data is provided to run the notebooks. Should you want to gather new data, simply refer to the ```Scraping``` notebook.

## Directory Descriptions

```Data```
- ```raw``` - csv test sample of raw data
- ```processed``` - cleaned data used in the modeling notebook. More detailed descriptions of each can be found in the [cleaning](Notebooks/Cleaning.ipynb) notebook

```Notebooks```
- [Scraping](Notebooks/Scraping.ipynb) - used for querying the Reddit Pushshift API
- [Cleaning](Notebooks/Cleaning.ipynb) - text processing, feature engineering, EDA
- [Modeling](Notebooks/Modeling.ipynb) - model building and evaluation

```Models```
- Final models fit from original data

```Reports```
- ```presentation``` - pdf and ppt format of final project presentation
- ```figures``` - images from original output used as references in notebooks

```src```
- Contains code for functions used in the notebooks. Each directory corresponds to one notebook, as well as a separate ```visualizations``` directory

```flask```
- Contains templates, static files, models, and python scripts used to build the flask app.

## Conclusions

- Test F1 = 0.84
- Best performing model utilized Tf-idf and logistic regression
- Model strongest at predicting non-clinical observations

### Methods Used
- Classification algorithms (logistic regression, KNN, random forest, linear SVM)
- Document embeddings
- Word embeddings
- Tf-idf
- Cross validation
- Flask


