# Scope of Problem
The idea here is that we are provided with data on <u>all</u> 144 compounds in T cells (TODO: add subtypes) and NK cells, but only 10% in Myeloid cells and B cells. We want to be able to predict drug effects in Myeloid cells and B cells. In another words, we want **cross-cell-type transfer**.

### Determining Test Splits
We need to determine our train-test splitting strategy to accurately evaluate model performance. I first employed `StratifiedKSplits` on drugs. This avoids the situation in which all data points for one drug accidentally fall into the test set (and none in the train set). In other words, each train and test set contains drugs at equal proportion as the global data set. For example:

|       | Masitinib  | R428 |Resminostat    |
--------| -------- | ------- |------- |
| Train (80%)|  16   |8|56
| Test (20%)| 4 |2|14
| Global|   20 |    10|70


The intention was so that the model would never see a drug that it has never seen before. 

Concretely with our pseudobulked dataset, say the stratified splits are:

|       | Masitinib  | R428|Resminostat    |
--------| -------- | ------- |------- |
| Train (80%)|  3   |5|2
| Test (20%)| 1 |1|1
| Global|   4 |    6|3

That means that within Masitinib,
```
   NK cells            -> train
   T cells CD4+        -> train
   T cells CD8+        -> train
   T regulatory cells  -> TEST     ← row being predicted
```

When the model predicts the held-out the (Masitinib, T regulatory cells) row, the training set contains (Masitinib, NK), (Masitinib, CD8+) and (Masitinib, CD4+), as well as every other drug tested in T regulatory cells. The model has seen what Mastinib <i>usually</i> does to cells, and how T regulatory cells <i>usually</i> respond to drug perturbations. That is essentially what the median baseline encodes. This is hard to beat!
```
cell_type
NK cells              146
T cells CD4+          146
T regulatory cells    146
T cells CD8+          142
B cells                17
Myeloid cells          17
```
This also means that B cells and Myeloid cells take up an extremely small portion of any test splits, and any errors would be diluted by correctly predicting drug effects in the other cell types. 
|       | Model  | Real test|   
--------| -------- | ------- |
|Train| 100+ drugs in T/NK <br>~10% in B/Myeloid| 100+ drugs in T/NK<br> ~10% drugs in B/Myeloid|
|Test| Mostly T/NK <br>(many data points in train set) |All B/Myeloid<br>(few data points in train set)|
|Task| Interpolation within cell type|Transfer to new cell type|

Therefore, this is not a good way of assessing model performance since we would be optimizing for the wrong thing :/

What if we use LOGO (TODO: define LOGO) on cell type? This works, but makes the problem a LOT harder. In the real test case, we <i>should have</i> seen some B/Myeloid cell data, instead of predicting in a completely novel cell type. 

A better way to approach this would be to create test-split sets where the test set primarily comprises of B/Myeloid cells x drug. However, this would give us a very small test set. We can instead artificially stimulate this holdig out drugs (~10%) from T/NK cells.   
We can employ KFold split <i>within</i> each cell type to generate 90-10 train-test splits, then combine corresponding <i>i</i>-th splits across all cell type to generate final train-test folds. KFold ensures that all drugs within each cell-type is held out at least once. 

Transfer within T/NK cell types is likely much easier than transfer to B/Myeloid cells, as they share a closer developmental lineage. We would need some additional assessment metric (eg. a test set that is only in B/Myeloid) or some weighted scoring system (eg. penalize wrong predictions in B/Myeloid more heavily).

## Establishing baselines
Now that we have defined our train-test splits, we will first establish the baseline values that we need to beat. The Open Problem scores predictions using Mean Rowwise Root Mean Squared Error, so this is the error metric I will stick with throughout this project. MRRMSE is computed as follows:
$$\text{MRRMSE} = \frac{1}{R}(\sum_{j=1}^n(y_{ij}-\hat{y}_{ij})^2)^{1/2}$$
where $R$ is the number of scored rows (ie. predictions), $y_{ij}$ and $\hat{y}_{ij}$ are the true and predicted values for a specific gene in that sample, and $n$ is the number of columns (ie. genes).

The simplest thing to do would be to just return the average value across all samples in the training set, per gene, like this:
```python
class MeanBaseline:
    def fit(self, X, y):
        self.means = y.mean()   # average gene expression PER GENE

    def predict(self, X):
        df = pd.DataFrame([self.means] * X.shape[0])
        return df
```
This gave us an average MRRMSE of 1.3327 across folds. Not very good.  
What if we use the median instead? 
```python
class MedianBaseline:
    def fit(self, X, y):
        self.median = y.median()    # median gene expression PER GENE

    def predict(self, X):
        df = pd.DataFrame([self.median] * X.shape[0])
        return df
```
Hmm slightly better, 1.2639. What about conditional mean? Here we give the option to return the conditional mean of a specified category, like cell type or drug. For example, if cell type was specified as the condition, the function returns the average expression of each gene across all drugs in that cell type, for all test samples that are also in that cell type.
```python
    def __init__(self, condition='sm_name'):
        self.condition = condition

    def fit(self, X, y):
        if self.condition not in X.columns:
            raise ValueError("condition must be in training set")
        self.conditional_mean = {} # initialize lookup table category -> mean

        # Get unique categories in `condition`
        u_cat = X[self.condition].unique()

        # For each category, get the mean of that category across samples
        for cat in u_cat:
            is_cat = X[self.condition].values == cat
            cat_mean = y.values[is_cat].mean(axis=0, keepdims=True) #(1, N)
            self.conditional_mean[cat] = cat_mean

    def predict(self, X):
        if self.condition not in X.columns:
            raise ValueError("condition must be in test set")

        default_shape = list(self.conditional_mean.values())[0].size
        y_hat = []

        for cat in X[self.condition].values:
            if cat not in self.conditional_mean.keys():
                y_hat.append(np.zeros(default_shape))
            else: 
                y_hat.append(list(self.conditional_mean[cat]))
        df = pd.DataFrame(np.vstack(y_hat))
        return df

```
Here, our average error with drug as the condition is 1.3358, and with cell type, 1.3508. Let's go try some linear models :3
