## Personal Details
Full Name: Tan Min Zheng
Admin Number: 222983R
Email Address(School): 222983R@mymail.nyp.edu.sg
Email Address(Private): businesszheng@yahoo.com

## Folder Overview

tan_min_zheng_222983R/

├── .venv
├── conf
│ └── base
|      ├── catalog.yml # Assigning all variables in the pipeline to locations to store them
│      └── parameters_ML.yml # Declaration of all ML parameters
│ └── local
|      ├── .gitkeep
│      └── credentials.yml # Unused File, but important for environment location
│ └── README.md # Not the actual readme, a kedro support file that explains this filder
├── saved_model
│ └── regressor_fine.pickle # Folder that saves all fine tuned model parameters w/ versioning
| └── regressor.pickle # Folder that saves all base model parameters w/ versioning
├── src
│ ├── dataprep
│ │ └── processed_data
│ │ │    ├── data_processing
│ │ │    ├── ml_pipeline_data
│ │ │    └── saved_outputs
│ │ └── raw_data
│ └── model
│      ├── __pycache__ # Kedro's Cached Memory Folder
│      ├── pipelines
│      │    ├── data_processing
│      │    │   ├── __pycache__ # Kedro's Cached Memory Folder
│      │    │   ├── __init__.py # Initialize pipeline creation
│      │    │   ├── nodes.py # Data processing nodes
│      │    │   └── pipeline.py # Data processing pipeline
│      │    └── ml_models
│      │         ├── __pycache__ # Kedro's Cached Memory Folder
│      │         ├── __init__.py # Initialize pipeline creation
│      │         ├── nodes.py # Data processing nodes
│      │         └── pipeline.py # Data processing pipeline
│      ├── __init__.py # Declaring project name and version
│      ├── __main__.py # Kedro Error Checking
│      ├── pipeline_registry.py # Register Pipeline
│      └── settings.py # Kedro COnfig settings
├── .gitignore # Kedro support file that ignores certain codes
├── .telemetry # Kedro variable for analytics purposes
├── eda.ipynb # EDA Jupyter Notebook
├── eda.pdf # EDA Report in PDF
├── pyproject.toml # Declaration of Kedro pipeline and location of important files
├── README.md # Project description and documentation
├── requirements.txt # Python packages requirements
└── run.sh # Shell script to run the data preparation and modeling pipeline

## Software Details:
Environment:
- Jupyter Notebook created & ran in Visual Studio Code
- Jupyter Notebook ran in Ananconda, Python Interpreter
- Ran in a virtural environment [.venv]
- Code is ran on Window Subsystem for Linux

Python Environments:
- Jupyter Notebook: Anaconda (Python 3.11.5)
- Environment: Python 3.10.12

## List of Libraries:
### Data Processing
numpy==2.0.0
pandas==2.2.2
seaborn
### Machine Learning
scikit-learn==1.5.1
imbalanced-learn==0.12.3
xgboost==2.1.0
### Kedro
kedro[all]
ipython>=8.10
jupyterlab>=3.0
kedro-telemetry>=0.3.1
notebook==7.2.1
pyarrow==16.1.0
fastparquet==2024.5.0

# EDA Summary
## Pre-Merge
In my EDA, I first look at each dataset on the individual level before merging. 

| DataFrame      | Shape        | Duplicate Rows | Rows with Null Values | Cols with Null Values | Name of Null Cols                                                                                                                                 |
|----------------|--------------|----------------|-----------------------|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| customers_df   | (99441, 5)   | 0              | 0                     | 0                     | []                                                                                                                                                |
| location_df    | (1000163, 5) | 261831         | 0                     | 0                     | []                                                                                                                                                |
| items_df       | (112650, 7)  | 0              | 0                     | 0                     | []                                                                                                                                                |
| payment_df     | (103886, 5)  | 0              | 0                     | 0                     | []                                                                                                                                                |
| review_df      | (99224, 7)   | 0              | 89385                 | 2                     | ['review_comment_title', 'review_comment_message']                                                                                                |
| order_df       | (99441, 8)   | 0              | 2980                  | 3                     | ['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date']                                                            |
| product_df     | (32951, 9)   | 0              | 611                   | 8                     | ['product_category_name', 'product_name_lenght', 'product_description_lenght', 'product_photos_qty', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm'] |
| seller_df      | (3095, 4)    | 0              | 0                     | 0                     | []                                                                                                                                                |
| translation_df | (71, 2)      | 0              | 0                     | 0                     | []                                                                                                                                                |

As we can see from this report, it shows that some datasets have duplicates and null values. We will remove this immediately in data processing.

## Errorneous Values
To just get a glimpse of whether errors exist, we look at the location_df, which after passing thru the code returns this:
| state_code | unique_cities                                                                 | no_of_cities |
|------------|-------------------------------------------------------------------------------|--------------|
| SP         | [sao paulo, são paulo, sao bernardo do campo, ...]                            | 1048         |
| RN         | [são paulo, natal, parnamirim, sao jose de mip...]                            | 214          |
| AC         | [sao paulo, rio de janeiro, sena madureira, ri...]                            | 34           |


Though there are many errors, we can thankfully just drop these and use the states as the representation of buyer/seller location. This reduces the possibility of errors and dimensionality of data. 

## Merging
We merge based on the following schema found linked with the dataset on kaggle:
![Schema of Dataset](https://i.imgur.com/HRhd2Y0.png "Schema of OLIST")

The code is as follows:
```
merge_info = [
    (customers_df, 'customer_id', 'customer_id'),
    (payment_df, 'order_id', 'order_id'),
    (review_df, 'order_id', 'order_id'),
    (items_df, 'order_id', 'order_id'),
    (product_df, 'product_id', 'product_id'),
    (seller_df, 'seller_id', 'seller_id'),
]
```

We avoid merging translations and locations due to their errors and we ahve not processed them at this stage

## Identifying Repeat customers
After merging, its noticeable that there is no clear indication of repeat buyers
Eventually, I came up with 3 possible scenarios to represent repeat buyers

- **Solution A: Customers per unique order**
- **Soltuion B: Customers per unique item bought**
- **Solution C: Customers by repurchase from same seller**

Out of all these 3 solutions, **Solution A** was the most prospective as the difference between those that ordered >1, the mean and the standard deviation above the mean of the customers showed the smallest possible difference amongst all 3. (By about 50000)

## Summary of EDA Findings
### From analysis of trends
1) Certain factors can affect the seasonal demand of goods from customers, as shown by the 2 peaks in the distribution of orders bought over time. During that period it seems that is where majority of people has reordered. 

2) In the whole dataset, almost all orders (95%) is delivered, giving us a good starting point to just keep all delivered orders.

3) Generally people were contented with the deliveries by giving high review scores. However, the fact that in those re-ordered sections do indicate that low reviews were given means that not all sellers has the same and consistent service

4) Between the type of product bought, there is not a big difference in the top 5 items bought. The total of the top 5 categories consist of bed bath table, sports & leisure, furniture & decor, health & beauty and computer accessories

5) As for payments, the amount and spread of each type is about the same in terms of its monetary value, meaning that as the cost of payment increases, so does they type of mayment method they use.

6) As for price and freight value, both datasets show a positive relationship with each other, albeit at a very gentle rate of increase possibly due to outliers. However, multiple orders made generally are below 100 frieght value and 1000 order value. 

7) As for date of delivery, the spread of early and late orders are similar, with majority being early. THis does affect the order rate, but in terms of repeated orders, doesnt show a good representation

8) The relative amount of sellers and buyers in each state doesnt vary much, leading us to belive that it is not important in E-commerce

### From the data itself
1) the number of categories, number of items bought and number of sellers dont matter from the correlation matrix.

2) Days & Review score, Frieght Value & Price, Payment method and number of payments have some correlation between each other.

### Dataset Changes
1) 50k rows removed, from 119143 entries to 67902
2) Columns removed, from 25 to 10
3) Eventually we will remove 3 more columns (number of categories, number of items bought and number of sellers), leading to 7 columns left, with `repurchase` as the variable indicating repeat buyers in the notation of 0 for non-repeat, 1 for repeat




# Pipeline
## Explanation of pipeline
Kedro is an open-source Python framework designed for creating reproducible, maintainable, and modular data science and machine learning workflows. Its primary strength lies in its pipeline capabilities, which help structure and organize code and data processing tasks in a way that enhances collaboration, scalability, and reliability. In my scenario, it allows me to sequence my processes, ensure accurate input and outputs, and also return any errors found within the dataset. Furthermore, i can run specific sections of the code to test my model. 

In this case, my pipeline consists of 2 sections, the data cleaning pipeline and machine learning pipeline
In general, each pipeline is defined by the nodes.py and pipeline.py in each folder
- The `nodes.py` file contains the functions used in the pipeline to help keep the code modular and neat. 
- The `pipeline.py` file defines the node, sequence, inputs and outputs of the code

## Instruction of execution of pipeline & modifying parameters
1) Enter the file's current directory in WSL (It should look like `cd /mnt/c/Users/RyanT/tan_min_zheng_222983R`)
2) Install Required libraries
    - Run the Command `pip install -r requirements.txt` in your terminal in WSL in the venv
3) To execute the pipeline, simply run `bash run.sh` and select the fuctions of the pipeline you want to use
    - Note: Type `chmod +x run.sh` to initialize the file if it doesnt run


## Explanation of each individual section
### Data Cleaning Pipeline
Data cleaning consists of the following steps:

1) Feature Selection in the individual dataset level, where we immediately remove redundant information
    - It returns each individual cleaned dataset and passes the variables to `merge_data()`

2) Merging to account for all datasets by the schema found in the olist datasets 
![Schema of Dataset](https://i.imgur.com/HRhd2Y0.png "Schema of OLIST")
    - it returns the `merged_df` and passes to `preprocess_all`

3) Processing the merged dataset to clean and engineer new features to summarize certain variables based on the following steps:

    1) Creating the `repurchase` variable indicating customers that have bought 

    2) Removing the multiple entries from payments by condensing the information into simple columns

    3) Taking only delivered orders to completely handle majority of errors

    4) Getting whether orders are delivered early or late as well as by their value

    5) Final check by ensuring that the sum of payments is equal to the price and freight value of the order

    6) Dropping all unused or processed columns

This in turn gives us the dataset with the following details:
|**Original Dataset:**  |**Final Dataset**	|**Rows Removed**	|**Columns Removed**|
|(119143, 25)	        |(67902, 10)	    |51241	            |15                 |

4) Normalisation.
We normalize 3 features, payment, freight and days. This is as they all have large values and can be reduced.

We use `StandardScaler()` for `days` and `MinMaxScaler()` for `payment value and freight value`.

This is as days is revolved around the value 0, which indicated whether the delivery was early or late, which just coincidentally `StandardScaler()` does also, trying to match the mean values to 0.

`MinMaxScaler()` for payment value and freight value is used as they both have a upper boundary and lower boundary from the EDA done.

The rest of the features do not require feature engineering as they are all already in numerical format. Furthermore, most of them do not have large, variable values, hence there will not be a need for such modification

Then this final dataset is saved to a parquet file in the form of a pandas.DataFrame before it gets used for processing. The variable name is `model_input_table`

### Machine Learning Pipeline:
The machine learning pipeline takes the in `model_input_table` and returns a correlation matrix at the end
The pipeline consists of 3 general sections:
1) Data splitting & scaling
2) Model training
3) Model evaluation

Specifically these are what each of the individual nodes do.
#### Data Splitting/Scaling:
It takes `model_input_table` and processes it into the train, test and validate components. It all then gets saved under `processed_data/saved_outputs` 

A scaling tool is then used to balance the dataset
Knowing that we have multiple features and also a extremely biased dataset, we need to conduct splitting and scaling carefully. Wrong methods of scaling and splitting will result is a biased model towards the majority class(in this case is single order buyers). 

Undersampling would mean to decrease the sample size of the majority class to the minority. This would result in a fair representation of the minority class, but as a result underrespresent the majority class.

From here, it is expected that the model would achieve better results for the minority than the majority. However the majority class would suffer a decrease in it's accuracy.

##### Explanation of the 3 sampling techniques
- SMOTE (Synthetic Minority Over-sampling Technique)
SMOTE is a technique used to handle imbalanced datasets by generating synthetic samples for the minority class. It creates new instances by interpolating between existing minority class instances. It Increases the number of minority class samples without duplication. It also helps improve the performance of classifiers by providing more balanced training data. However, it can create noise by generating synthetic samples that might not represent real data. It also may lead to overfitting if not carefully applied.

- NearMiss
NearMiss is an under-sampling technique that selects a subset of majority class samples by choosing those that are closest to the minority class samples. It focuses on keeping instances that are near the decision boundary. It Reduces the size of the dataset, making training faster. It also keeps relevant majority class samples that are informative for the classification task. However, it also leads to loss of potentially important information from the majority class. Furthermore, it may still suffer from class imbalance if the reduction is not sufficient.

- Random Undersampling
Random Undersampling reduces the number of majority class instances by randomly selecting and removing samples. This technique aims to balance the class distribution. It is simple and easy to implement and reduces the dataset size, making computation more efficient. However, Randomly removing samples can lead to loss of important information and cause underfitting as the reduced dataset might not capture the overall data distribution well.


#### Model Training
##### Models used:
###### Random forest
RandomForest is an ensemble method that builds multiple decision trees and merges them to get a more accurate and stable prediction. It can model complex non-linear relationships well, which is often necessary for real-world data like ours, where ther are multiple features needed to be analysed RandomForest also provides a way to understand feature importance thru the gini coefficient of each node, indicating the threshold of the split. Furthermore, By averaging multiple trees, RandomForest reduces the risk of overfitting compared to individual decision trees.
###### XGBoost
XGBoost is known for its high performance in terms of both speed and accuracy. It is often one of the top choices in machine learning competitions. It provides clear insights into feature importance, helping in understanding which factors most influence adoption rates. It also includes L1 and L2 regularization, which can prevent overfitting and improve the generalization of the model.

##### Use of multiple models
1) Different models may perform better on different types of data. So by using multiple models, i can compare and identify which is better
2) Having multiple models provides robustness for my predictions, it also helps in seeing if my models are overfitting or not

#### Model Evaluation
In our model evaluations, we evaluate based on the recall score of the model, seeing which of the 2 (fine_tuned vs untuned) on the validation dataset, which is 20% of the train dataset. Then we compare and return the model with the best recall score and we pass the test dataset, which involves 20% of the untrained data unscaled, to get the classification report. Showing the precision, recall, F1 score and accuracy

- Precision: The proportion of true positive predictions among all positive predictions.
- Recall: The proportion of true positive predictions among all actual positive cases.
- F1: The harmonic mean of precision and recall, balancing both metrics.
- Accuracy: The proportion of all correct predictions (both true positives and true negatives) among all cases.

# Evaluations
1) SMOTE + RF 
Parameters:
- RF_criterion: 'log_loss' # Between gini, entropy, log_loss
- RF_n_estimator: 30 # 0 - inf
- RF_max_depth: 10
- RF_max_features: 5

Base Model: 
| Repurchase | Precision | Recall | F1   |
|------------|-----------|--------|------|
| 0          | 0.95      | 0.86   | 0.90 |
| 1          | 0.07      | 0.18   | 0.10 |

Accuracy 0.82

Fine-Tuned Model:
| Repurchase | Precision | Recall | F1   |
|------------|-----------|--------|------|
| 0          | 0.95      | 0.52   | 0.69 |
| 1          | 0.06      | 0.50   | 0.10 |

Accuracy 0.52

As we can see, the fine tuned model does worse accuracy wise, but is able to recall much better. This can be due to SMOTE just duplicating values and causing the recall to be way better as it had more data to learn from

2) SMOTE + XGB
Parameters:
- GB_booster: 'gbtree' # Between 'gbtree' or 'dart'
- GB_lambda: 2 # 0 - inf
- GB_learningrate: 0.02 # 0 - 1
- GB_n_estimator: 30 # 0 - inf
- GB_max_depth: 10 # 0 - inf

Base Model: 
| Repurchase | Precision | Recall | F1   |
|------------|-----------|--------|------|
| 0          | 0.95      | 0.74   | 0.83 |
| 1          | 0.06      | 0.26   | 0.09 |

Accuracy 0.72

Fine-Tuned Model:
| Repurchase | Precision | Recall | F1   |
|------------|-----------|--------|------|
| 0          | 0.95      | 0.74   | 0.83 |
| 1          | 0.06      | 0.27   | 0.10 |

Accuracy 0.715

As we can see, the fine tuned model does as bad as normal. This can be due to SMOTE just duplicating values and XGB not being able to handle it.

2) RandomUndersampling + XGB
Parameters:
- GB_booster: 'gbtree' # Between 'gbtree' or 'dart'
- GB_lambda: 3 # 0 - inf
- GB_learningrate: 0.04 # 0 - 1
- GB_n_estimator: 30 # 0 - inf
- GB_max_depth: 10 # 0 - inf

Base Model: 
| Repurchase | Precision | Recall | F1   |
|------------|-----------|--------|------|
| 0          | 0.95      | 0.51   | 0.66 |
| 1          | 0.06      | 0.51   | 0.10 |

Accuracy 0.51

Fine-Tuned Model:
| Repurchase | Precision | Recall | F1   |
|------------|-----------|--------|------|
| 0          | 0.95      | 0.51   | 0.67 |
| 1          | 0.06      | 0.52   | 0.11 |

Accuracy 0.51

Here, the accuracy of the fine tuned and base model are the same, meaning that majority of the work comes from the sampling module.

