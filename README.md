# clustering-analysis

### A Python machine learning project that aims to compare the performances of various K-mean and hierarchical clustering algorithms that utilize different distance metrics, building off of previous work that explored implementation of uncommonly used metrics such as Manhattan and Hamming distances. 

* Libraries used include sk-learn, pyclustering and matplotlib
* Uses Iris dataset for simplicity and ease of calculations, as well as enabling more accurate comparisons
* Compares accuracies of Euclidean vs. Manhattan distance metrics as well as different types of clustering (Density-based (DBSCAN), Filter)
* Optimized implementation of each algorithm by adjusting initiation parameters from results of confusion matrices and general accuracy

### Visualizations (Scatterplot and Confusion Matrix)
![Plot](https://github.com/candace-sun/clustering-analysis/blob/main/Figure%202023-06-06%20120903.png)
![Confusion Matrix](https://github.com/candace-sun/clustering-analysis/blob/main/Figure%202023-06-06%20120911.png) 

### Notes:
* Can produce inconsistent results due to random selection in algorithms
* Best performance: K-means Euclidean distance clustering at 89.3% accuracy
