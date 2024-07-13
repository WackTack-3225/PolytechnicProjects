import pandas as pd 
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
import configparser

config = configparser.ConfigParser()
config.read('../../parameter.env')
output_file = config.get('PARAMETERS', 'PROCESSED_FILE')
input_file = config.get('PARAMETERS', 'INPUT_FILE')

print("Reading file from " + input_file)
df = pd.read_csv(input_file)

print("Data Processing Stage1: feature engineering")
# Function to check if an animal is purebred or mixed breed
def breed_type(row):
    if row['Breed1'] == 307 or row['Breed2'] != 0:  
        return 1 # Mixed breed
    else:
        return 0 # Pure Breed

# Create New Column
df['BreedType'] = df.apply(breed_type, axis=1)

# Get colors of an animal
def color_no(row):
    a = 0
    if row['Color1'] != 0:  
        a += 1
    elif row['Color2'] != 0:  
        a += 1
    elif row['Color3'] != 0: 
        a += 1 
    return a
df['ColorAmt'] = df.apply(color_no, axis=1)

df['Free'] = df['Fee'].apply(lambda x: 1 if x == 0 else 0)
df['NameorNO'] = df['NameorNO'].apply(lambda x: 1 if x == 'Y' else 0)

cols_to_use = ['Type', 'Age', 'Gender', 'MaturitySize', 'FurLength', 
               'Vaccinated', 'Dewormed','Sterilized', 'Health', 'Quantity', 
               'Fee', 'State', 'VideoAmt', 'PhotoAmt', 'AdoptionSpeed', 
               'NameorNO', 'BreedType','ColorAmt','Free']
pets_data = df[[col for col in cols_to_use if col in df.columns]]
cat_cols = ['Type', 'Gender', 'MaturitySize', 'FurLength', 'Vaccinated', 
            'Dewormed', 'Sterilized', 'Health', 'State','NameorNO', 
            'BreedType','ColorAmt','Free']
float_cols = ['Age','Quantity', 'VideoAmt','PhotoAmt']

print("Data Processing Stage2: One Hot Encoder")
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
ohe.fit(pets_data[cat_cols])
ohe_pets_data = pd.DataFrame(ohe.transform(pets_data[cat_cols]), columns=ohe.get_feature_names(cat_cols),index=pets_data.index)
pets_data = pd.concat([pets_data, ohe_pets_data], axis=1)

print("Data Processing Stage3: MinMaxScaler")
mm_scalar = MinMaxScaler()
for col in float_cols:
    mm_scalar.fit(pets_data[col].values.reshape(-1, 1))
    pets_data[col] = mm_scalar.transform(pets_data[col].values.reshape(-1,1))

print("Data Processing completed. Saving file to "+ output_file)
pets_data.to_csv(output_file)

