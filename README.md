# README

My solution for the Proofpoint hands-on technical exercise.

## Running the code

Create a conda environment using the provided `environment.yml` file:

```shell
conda env create -f environment.yml
conda activate dathena-exo
```

Execute the `script.py` file to run my solution:

```shell
python3 script.py
```

## Submission contents

My submission contains 4 files:

- `script.py`: the script to execute my final solution.
- `research.ipynb`: a notebook containing my detailed approach to solving the exercise. Contains visualizations and
  alternative approaches considered. Please note that some of the visualizations are interactive and will require
  the notebook to be re-run in order to display.
- `helpers.py`: functions which are called in the two other files.
- `environment.yml`: contains the list of dependencies of the code.

## Document Extraction

I chose to scrape recipes in English, French and Spanish. Since I could not find a multilingual recipe website, I scrape
50 recipes from a different website for each language. For each recipe, three sections are scraped:

- the recipe's title
- the list of ingredients
- the directions to make the recipe

To get a good mix of different types of recipes (main dish, dessert, etc.), I scrape the recipes from the front page
or 'popular recipes' page of the website.

## Language Detection

Each recipe's language is detected with perfect accuracy by using langdetect on the recipe's directions. Using the
directions is more robust since it has the largest size and the title and ingredients sometimes contain foreign words.

I then chose to translate each recipe to English using googletrans. This has the downside of taking a bit of time,
although not too much (~3 minutes for 150 recipes) since the recipes are often quite short. But it then greatly
simplifies the preprocessing of the text and allows us to cluster all the documents at the same time without having to
treat each language case by case. By translating everything to English, it is also very easy to add recipes in a new
language to the dataset.

## Text Processing

Each section of the recipe (plus a section containing all the others) is preprocessed following these steps:

- Case folding
- Punctuation removal
- Tokenization
- Lemmatization: reduces vocabulary size
- Removal of single-character tokens: artifacts from punctuation removal
- Removal of numbers: portions vary between recipes so numerical values can't be used to differentiate entrées (which
  are smaller) from main dishes
- Removal of stopwords and too frequent/unfrequent words: lower and upper frequency thresholds were hand-tuned to keep
  only relevant words

I also tried to add bigrams to the text (which would for example allow to differentiate between crème_fraîche which is
salty and cream which is sweet). But in the end it produced worse results so I did not use them. A possible improvement
which might make the bigrams beneficial could be to only keep noun+noun bigrams.

## Cluster documents into logical groups

To vectorize the preprocessed text, I use TF-IDF. I also tried loading word vectors for fasttext, but my laptop doesn't
have enough RAM to handle them. Using a word embedding could be a possible strong improvement. Since it encodes semantic
similarity between 'chicken', 'beef' and 'lamb' for example, it could potentially be able to group recipes for meat
dishes.

First, to gain insight into the structure of the data, I project it using PCA or t-SNE and visualize. See the **TF-IDF
projection** section in `research.ipynb` for more detail. From this I infer that the 'ingredients' section is most
promising for separating the recipes and that there seems to be 2 clusters in the data: 'sweet' and 'salty'.

Next, I attempt 3 different methods to group the recipes:

- K-means clustering
- DBSCAN as it should be able to better detect non-convex clusters while ignoring the 'outlier' recipes
- Latent Dirichlet Allocation (LDA)

I attempt K-means clustering with different numbers of clusters and on different sections. The best results are given by
running with 2 clusters on the TF-IDF vectors of the 'ingredients' and groups the recipes well
between 'sweet' and 'salty'. Please see the **K-means clustering** section in `research.ipynb` for visualization of the
results.

In the `research.ipynb` notebook, I also try clustering with DBSCAN, but this gives slightly worse results than K-means.
Perhaps some more fine-tuning of the epsilon and min_samples parameters could lead to better results.

Finally, I give LDA a try and tune the num_topics, alpha and eta parameters with grid search, trying to optimize both
the topics' coherence and diversity. However the results are disappointing as the topics found are unbalanced and
uncoherent. It is possible that trying wider ranges of parameters in the grid search, or optimizing a different metric
could lead to a better model.

## Results and discussion

To conclude, I have built a pipeline which scrapes 150 recipes in 3 languages, perfectly detects their languages,
reduces the size of the text by a factor of 2.5 and then groups the recipes very accurately between salty and sweet.

The advantages of my solution are that it is very easy to add recipes in new languages and that the clustering is fast
and quite lightweight as there is no need to load heavy word vectors.

A few limitations (translation time, only 2 clusters) of my solution and possible remedies (bigrams, more tuning of
models, word embeddings) have already been mentioned above. Another limitation is that it
is only able to group recipes between 'salty' and 'sweet', not in a more
fine-grained way between 'entrée', 'main dish', 'dessert', 'snack', etc. A possible solution to this could be to also
scrape the introductory text of the recipe, which might contain more useful data about the type of the dish.
