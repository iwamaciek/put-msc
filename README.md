# put-msc

This is a repository containing the code and notebooks prepared during the work on my Masters Thesis, "An efficient and scalable algorithm for learning rule ensembles with application to medical data" under the supervision of dr hab. inż. Wojciech Kotłowski, prof. PP.

The ENDER algorithm is described in the paper "Ender: a statistical framework for boosting decision rules."[1] by Krzysztof Dembczyński, Wojciech Kotłowski, and Roman Słowiński.

The original code was written by Maciej Kurzawa for his Master's Thesis, "Efficient and interpretable ensembles of decision rules". The code was modified to execute faster when compiled with Cython.

The k-Means equivalent points lower bound, as seen in the paper "Optimal Sparse Regression Trees" [2], was also added as a heuristic to lower the length of the rules. This can be viewed as increasing the interpretability. It also improved the predictive performance.

With the help of mgr Weronika Kraczkowska of the Poznań University of Medical Sciences, a special dataset about endometriosis was used. It contains survey data collected in paper form at the Gynecological Obstetric Clinical Hospital of Poznań University of
Medical Sciences between September 2024 and July 2025. The study was supervised by dr. hab. n. med. Maciej Brązert.

### Repository structure
- Endometriosis/ - contains the files used to prepare the endometriosis dataset.
    - ankiety_raw_new.csv - the raw version of the newest survey data
    - ankiety.csv - the raw version of the older, smaller set of surveys
    - endometriosis_with_na.csv - the preprocessed version of the data, with missing values
    - endometriosis.csv - the preprocessed version of the data, no missing values.
- OriginalCode/ - contains the original code written by Maciej Kurzawa
- Rewrite/ - contains the modified code.

### Versions of the algorithm
There are six versions of the algorithm in the Rewrite folder. They were used during runtime and performance testing, to see the effect of using Cython.
- EnderClassifierBase.py - the original code with added typing
- EnderClassifier.py - uses static typing, meant to be compiled with Cython
- EnderClassifierModified.py - version optimized for Cython usage
- EnderClassifierModifiedPara.py - same as before, but can utilize parallelism when searching accross attributes.
- EnderClassifierBoundedFast.py - optimized for Cython, uses the k-Means equivalent points lower bound.
- EnderClassifierBoundedFastPara.py - same as before, but can utilize parallelism when searching accross attributes.

### Usage
The code first has to be compiled with Cython. The recommended method is to use:
```
cd Rewrite/
python setup.py build_ext --inplace
```
Which will compile the code and place the output files in the Rewrite folder.
On unix systems, use setup.py. On Windows, use setup_windows.py