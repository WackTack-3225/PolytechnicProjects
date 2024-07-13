import pandas as pd
import re

# Load data
data = pd.read_csv('./data/pets_prepared.csv')

# Reference Datasets
df_breed = pd.read_csv('./data/breed_labels.csv')
df_color = pd.read_csv('./data/color_labels.csv')
df_state = pd.read_csv('./data/state_labels.csv')

# Drop specific columns
data = data.drop(['BreedName', 'BreedPure', 'TypeName', 'GenderName',
                  'AdoptedName', 'MaturitySizeName', 'ColorAmt', 'ColorName', 
                  'FurLengthName', 'VaccinatedName', 'DewormedName', 
                  'SterilizedName', 'HealthName', 'StateName'], axis=1)

# Drop Bins
bin_columns = [x for x in data.columns if 'Bin' in x]
data.drop(columns=bin_columns, inplace=True)

# Breed Cleaning

# Handing Duplicate Breeds
data.loc[data['Breed1'] == data['Breed2'], 'Breed2'] = 0 

# Take Breed 2 if Breed 1 is 0 and fill in the correct fields
data.loc[data['Breed1'] == 0, 'Breed1'] = data['Breed2']
data.loc[data['Breed1'] == data['Breed2'], 'Breed2'] = 0

# Merge and drop to replace type
merged_df = pd.merge(data,
                     df_breed,
                     left_on='Breed1',
                     right_on='BreedID',
                     how='left',
                     suffixes=('','_ref'),
                     indicator=True)
merged_df['Type'] = merged_df['Type_ref']

# Return data to original
data = merged_df.drop(['BreedID','Type_ref','BreedName','_merge'], axis=1)

# Make it so that if breed 1 & 2 are the same change breed 2 to 0
data.loc[data['Breed1'] == data['Breed2'], 'Breed2'] = 0

# Name Cleaning

# Create New Lowercased Name column
data['NameLower'] = data['Name'].str.lower()
data.drop(['Name'], axis=1, inplace = True)
data['NameLower'].fillna('', inplace=True)

# Text-based Cleaning based on individual words and bi-grams
data.loc[data['NameLower'].str.contains('no name', na=False), 
         'NameorNO']  = 'N'
data.loc[data['NameLower'].str.contains('temporary name', na=False), 
         'NameorNO']  = 'N'
data.loc[data['NameLower'].str.contains('be named', na=False), 
         'NameorNO']  = 'N'
data.loc[data['NameLower'].str.contains('name them', na=False), 
         'NameorNO']  = 'N'
data.loc[data['NameLower'].str.contains('unamed yet', na=False), 
         'NameorNO']  = 'N'
data.loc[data['NameLower'].str.contains('not name', na=False), 
         'NameorNO']  = 'N'
data.loc[data['NameLower'].str.contains('nameless', na=False), 
         'NameorNO']  = 'N'
data.loc[data['NameLower'].str.contains('name less', na=False), 
         'NameorNO']  = 'N'
data.loc[data['NameLower'].str.contains('unnamed', na=False), 
         'NameorNO']  = 'N'

# Handling non-english text (i.e. special characters)
def keep_english_and_numbers(text):
    if isinstance(text, str):
        # remove non-ASCII characters
        cleaned_text = re.sub(r'[^\x00-\x7F]+', '', text)  

        # remove special characters
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned_text)  
        return cleaned_text.strip()
    return text 

data['NameLower'] = data['NameLower'].apply(keep_english_and_numbers)
data.loc[data['NameLower'] == '', 'NameorNO'] = 'N'

data.to_csv("./data/cleaned_data.csv",index=False)

