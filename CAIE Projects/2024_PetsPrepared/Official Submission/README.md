## Personal Details
Full Name: Tan Min Zheng

Email Address: #######

## Folder Overview:

######/

├── saved_model
│ └── saved_model_xgb.json # Saved model file (or other best model file)
├── src
│ ├── dataprep
│ │ └── data
│ │ ├── breed_labels.csv # Breed labels data
│ │ ├── cleaned_data.csv # Cleaned data
│ │ ├── color_labels.csv # Color labels data
│ │ ├── eda_data.csv # EDA modified data
│ │ ├── pets_prepared.csv # Main prepared dataset
│ │ ├── state_labels.csv # State labels data
│ ├── data_cleaning.py # Data cleaning script
│ ├── data_processing.py # Data processing script
│ ├── eda_data_modification.py # EDA data modification script
│ └── model
│ └── pipeline.py # Model training and evaluation script
├── eda.ipynb # EDA Jupyter Notebook
├── eda.pdf # EDA Report in PDF
├── parameter.env # Environment parameters file
├── README.md # Project description and documentation
├── requirements.txt # Python packages requirements
└── run.sh # Shell script to run the data preparation and modeling pipeline

## Software Details:

Python Version: Python 3.10.12

Environment: 
- Windows Command Prompt with python installed in a ubuntu subsystem
- Coded all on Visual Studio Code and ran in it also
- All done on a Windows OS

List of Libraries:
- pandas==2.2.1
- numpy==1.26.4
- matplotlib==3.8.3
- scikit-learn==0.23.2
- xgboost==2.0.3
- plotly==5.22.0
- wordcloud==1.9.3
- nltk==3.8.1
- seaborn==0.13.2
- joblib==0.17.0


## Summary of EDA Findings

### Data Overview

The dataset contains approximately **15,000 pets and 49 features**.

The features include information about the pets' **health, breed, type, rescuer ID, social media posts, cost of adoption and names**.
Engineered bins exist in the dataset, but they were removed for a more detailed analysis.

### Missing and Duplicate Data
- Columns with missing values were identified, `Names, BreedName and Descriptions`.
- No duplicate PetIDs were found in the dataset.


### Data Cleaning
- Bin columns were dropped to avoid over-generalization and ensure a detailed analysis based on raw data.
- Missing `BreedNames` were noted but considered non-critical due to the presence of other breed-related columns. Eventually they were correlated to not having values in `Breed1`, of which were replaced by `Breed2`.
- `Names` that had *"No Name"* or text to that effect did not correlate with the animal having no name in the `NameorNO` field. By using n-grams, we could see what combinations of words can be used to remove such field, although limited in effect. Additionally, some names has special characted, which may affect
- Eventually, columns that were referenced from original data or repetitive was dropped due to possibility of inconsistency, although in the background, checks have been made

### Key Findings
1) More dogs are adopted than cats, though cats generally had a faster adoption rate compared to dogs. Both types are adopted on a majority during the time period of 8-30 days.
2) For age, the younger the animal, the geater the likelihood of adoption, of which ages in 0-2 will be most favorable, followed by a steep decline.
3) For breeds, dogs are more favored to be mixed breeds at 43.8% of total adoptions while cats are more favoured to be pure breeds at 37.3% of adoptions.  
4) Gender did not seem to have a significant impact on the adoption rateas there is no varaince between male, female or mixed
5) Same for color, it followed the trendline for % of adoption rate by type For both types, majority of adoptions for all 3 colours are at 8-30 days
6) For maturity, medium sized animal are mostly favoured at 28.3% for cats and 39.6% for dogs for all adoptions
7) For health, The main factor that would follow a positive trend is dewormed, of which most adopted animals are dewormed, however vaccinated and sterilized are not, with majoirty coming in at `No` for both types.
8) For quanity, posts with 1 animals shown are mostly adopted, those with more than 1 shown are extremely unlikely to be adopted
9) For fees, most animals adopted are those with 0 fees at ~10000, with those requiring costs decreasing significantly as soon as it costs money.
10) As for state, majority of cats and dogs are adopted in Selangor, at 27.3% for cats and 33.5% for dogs. As they take up a significant proportion of the dataset, it could be heavily biased towards Selangor residents.
11) As for adoption by name, those with names are adopted more, but most common names given are generic like "puppy," "puppies," "kitten," and "kitty". This could possibly require more indepth cleaning to happen to get a clearer result.
12) As for media content, most animals adopted have 0 videos posted about them, with 1 or more being extremely low. As for photos, majority of animals adopted have 1 photo about them, with 2, 3, coming close. It experiences a sudden dip in adoption count at 4 and back up at 5. 
Adoption Distribution by Media Content:


### Summary and Recommendations
1) Additional feature engineering is needed for VideoAmt, PhotoAmt, State, Fee, and Color to be able to show more details
2) Both cats and dogs are typically adopted within 8-30 days.
3) The feature Fee had the highest impact on adoption rates, as free animals were preferred over those with an associated fee.
4) These findings suggest that further refining and engineering of specific features could enhance the predictive modeling of pet adoption rates.


## Instruction of executing pipeline & modifying paramters
1) Install Required libraries
    - Run the Command `pip3 install -r requirements.txt` in your terminal
2) Change the paths in `parameter.env` to the relevant absolute paths for the `INPUT_FILE and PROCESSED_FILE`
3) To execute the pipeline, simply run `bash run.sh` after `cd 222983R_CaseStudy`

## Description of logical steps/flow of the pipeline
1) Config & Setup, where it reads parameters from `parameter.env` & loads data
2) Data Preparation, where I split my data into train and test sets. Separated targe variable also
3) Model training, where i train the XGBoost, RandomForest and Logistic Regression models
4) Model Evaluations, wher i print out the classification reports
5) Model Comparison, whre i load the bast model trained and compare it with the current training results. It will save the RF or LR  models if it exceeds the historical accuracy

## Explanation of your choice of models for each ML Task
1. XGBoost (Extreme Gradient Boosting)
- High Performance: XGBoost is known for its high performance in terms of both speed and accuracy. It is often one of the top choices in machine learning competitions.
- Feature Importance: It provides clear insights into feature importance, helping in understanding which factors most influence adoption rates.
- Regularization: XGBoost includes L1 and L2 regularization, which can prevent overfitting and improve the generalization of the model.
2. RandomForest
- Ensemble Learning: RandomForest is an ensemble method that builds multiple decision trees and merges them to get a more accurate and stable prediction.
- Non-Linear Relationships: It can model complex non-linear relationships well, which is often necessary for real-world data.
- Feature Importance: Like XGBoost, RandomForest also provides a way to understand feature importance.
- Robust to Overfitting: By averaging multiple trees, RandomForest reduces the risk of overfitting compared to individual decision trees.
3. Logistic Regression
- Simplicity and Interpretability: Logistic Regression is simple and easy to interpret, making it a good baseline model. The coefficients indicate the direction and magnitude of the relationship between features and the target variable.
- Probabilistic Output: It provides probabilistic outputs, which can be useful for understanding the confidence of predictions.
- Efficiency: Logistic Regression is computationally efficient and works well on smaller datasets.
- Baseline Performance: It serves as a good baseline model to compare more complex models against.
### Summary of Advantages
- XGBoost: High accuracy, regularization, and feature importance.
- RandomForest: Ensemble method, models non-linear relationships, robust to overfitting, and feature importance.
- Logistic Regression: Simplicity, interpetability and baseline performance

### Use of multiple models
1) Different models may perform better on different types of data.So by using multiple models, i can compare and identify which is better
2)  Robustness: Having multiple models provides robustness for my predictions, it also helps in seeing if my models are overfitting or not


## Evaluation of the models developed
1) Logistic Regression
- Accuracy of 27% generally, least accurate of the 3
- unable to predict any of the classes correctly other than Class 4 for the longest wait times.

As it is a baseline model, it does show that the data itself is unable to predict the rate of adoption accurately 

2) Random Forest
- accuracy of 40%, able to predict class 1, 2, and 3 of much most adopted animals are found within this. predicts class 4 with an accuracy of 50%
- The RandomForest classifier performs better than the Logistic Regression model, as evidenced by the higher overall accuracy and better precision, recall, and f1-scores for most classes.
Other considerations for deploying models developed
- Class 3 has poor recall and f1-score, suggesting difficulty in correctly predicting this class.


3) XGBoost
- accuracy of 38%, however able to predict all classes at least once correctly. 
- The model performs relatively well for class 4, with higher precision, recall, and f1-score, indicating better prediction capability for this class.
- However it underperforms for class 0 and class 3

## Other considerations
Some considerations that could be made is to not overgeneralize the data, ideally we can use each columns raw data. Additionally, better data cleanign could have been done.