import pandas as pd

# Just used to change numeric to string values & some feature engineering
# Load data
data = pd.read_csv('./data/cleaned_data.csv')

# Reference Datasets
df_breed = pd.read_csv('./data/breed_labels.csv')
df_color = pd.read_csv('./data/color_labels.csv')
df_state = pd.read_csv('./data/state_labels.csv')

# Type Mapping
type_mapping = {1: 'Dog', 2: 'Cat'} 
data['Type'] = data['Type'].map(type_mapping)

# Adoption speed Mapping
spd_mapping = {0: 'Same Day', 1: '1-7 Days', 2: '8-30 Days',
                 3: '31-90 Days', 4: 'More than 100 Days'}
data['AdoptionSpeed'] = data['AdoptionSpeed'].map(spd_mapping)

# Age Binning intervals
bins = [0, 2, 6, 12, 24, 60, 255]
data['AgeBins'] = pd.cut(data['Age'], bins=bins, 
                        labels=['[0, 2)', '[2, 6)', '[6, 12)',
                                '[12, 24)', '[24, 60)', '[60, 255]'])

# Function to check if an animal is purebred or mixed breed
def breed_type(row):
    if row['Breed1'] == 307 or row['Breed2'] != 0:  
        return 'Mixed Breed' # Mixed breed
    else:
        return 'Pure Breed' # Pure Breed

# Create New Column
data['BreedType'] = data.apply(breed_type, axis=1)

# Gender mapping
gender_mapping = {1: 'Male', 2: 'Female', 3: 'Mixed'}
data['Gender'] = data['Gender'].map(gender_mapping)

# Color

# Function to check colors of an animal
def color_no(row):
    a = 0
    if row['Color1'] != 0:  
        a += 1
    elif row['Color2'] != 0:  
        a += 1
    elif row['Color3'] != 0: 
        a += 1 
    return a

# Create New Column
data['ColorAmt'] = data.apply(color_no, axis=1)

# Maturity Size
maturity_mapping = {1: 'Small', 2 : 'Medium', 3: 'Large', 4: 'Extra Large'}
data['MaturitySize'] = data['MaturitySize'].map(maturity_mapping)

# Fur Length
fur_mapping = {1: 'Short', 2 : 'Medium', 3: 'Long'}
data['FurLength'] = data['FurLength'].map(fur_mapping)

data.to_csv("./data/eda_data.csv",index=False)