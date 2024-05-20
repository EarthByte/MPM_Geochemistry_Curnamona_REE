# Prospectivity mapping of rare earth elements through geochemical data analysis
::Rare earth elements (REEs), a significant subset of critical minerals, are crucial for global sustainability. Separating geochemical anomalies associated with specific types of mineralization from the background reflecting geological processes has long been a significant subject in exploration geochemistry. The processing of high-dimensional, non-linear geochemical data necessitates a systematic framework to address common issues, including missing values, the closure effect, the selection of appropriate multivariate analysis methods, and anomaly detection techniques in order to detect geochemical anomalies associated with mineral occurrences. In this study, we establish a novel machine learning-based framework that incorporates a random forest-based data imputation method, an effective multivariate statistical analysis technology, and a deviation network-based anomaly recognition approach to address these challenges. We utilize lithogeochemical data to map potential greenfield regions of REE mineralization in the northern Curnamona Province. The comprehensive workflow for processing geochemical data proposed in this study can effectively address common challenges in the geochemical exploration of critical minerals. The identified geochemical anomalies can provide important clues for subsequent exploration. ::
## 1. Missing Value Imputation 
   - The `Comparison of imputation methods.ipynb` file provides various imputation methods that can be used to handle missing values in the geochemical subset.
   - By comparing the effects of different imputation methods, the optimal imputation strategy can be selected to improve data quality.

## 2. Multivariate Statistical Analysis
   - The `ILR_RPCA.R` file is used to process geochemical data, reducing the impact of the closure effect and outliers on principal component analysis.
   - By removing redundant information and extracting principal components representing different geochemical information and geological processes, subsequent data analysis can be optimized.

## 3. Anomaly Pattern Recognition
   - The code within ‘Geo_DevNet’ folder is designed to identify geochemical anomaly patterns associated with mineralization under imbalanced data conditions by leveraging prior knowledge of known REE deposits.
   - This method can effectively distinguish anomalies caused by mineralization from those caused by noise or anthropogenic factors, improving the accuracy of anomaly recognition.

## Datasets
For user convenience, each folder corresponding to a functional module contains the necessary datasets required to run the code. Users can directly use the provided datasets for testing and analysis.

