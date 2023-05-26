import pandas as pd
import bisect
import saveLoadData
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

class Preprocess:

    saveLoad = saveLoadData.SaveLoadData()

    # Constructor
    def __init__(self, x, y=None):
        self.X = x
        self.Y = y

    def handle_outliers(self, columns, data):
        # Handle outliers
        Q1 = data[columns].quantile(0.25)
        Q3 = data[columns].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_indices = []
        for col in columns:
            outliers = data[
                (data[col] < lower_bound[col]) | (data[col] > upper_bound[col])].index
            outlier_indices.extend(outliers)
        data.drop(outlier_indices, inplace=True)

        return data

    def handleGenres(self):
        # Declare dictionary
        oneHotEncode_dict = {}

        # Apply one-hot encoding to genres column
        df_encoded = self.X['genres'].str.get_dummies('|')

        # Save the genres of the encoded dataframe as a list in oneHotEncode_dict
        oneHotEncode_dict['genres'] = df_encoded.columns.tolist()

        # Join the one-hot encoded dataframe back to the original dataframe X
        self.X = pd.concat([pd.DataFrame(self.X).reset_index(drop=True), pd.DataFrame(df_encoded).reset_index(drop=True)], axis=1)

        # Drop columnName from X
        self.X.drop('genres', axis=1, inplace=True)

        self.oneHotEncode_dict = oneHotEncode_dict

    def encode(self, columns):
        # Take copy of original X
        self.X = self.X.copy()
        # Declare dictionary
        encoder_dict = {}
        for x in columns:
            # Create object of LabelEncoder
            le = LabelEncoder()
            # Learn encoder model on ele of X
            le.fit(self.X[x].values)
            # Implement encode on ele of X
            self.X[x] = le.transform(list(self.X[x].values))
            # save encoder model in dictionary
            encoder_dict[x] = le

        self.encoder_dict = encoder_dict

    def scale_data(self):
        # Create object of MinMaxScaler
        scaler = MinMaxScaler()
        # Reshape age column to a 2D array with a single feature
        age_col = self.X['age'].values.reshape(-1, 1)
        # Implement scaling on age column
        self.X['age'] = scaler.fit_transform(age_col)
        self.scaler = scaler

    def trainDataCleaning(self):
        # Handle genres column
        self.handleGenres()
        # Save oneHotEncode of genres
        self.saveLoad.saveModel(self.oneHotEncode_dict, 'genresEncode')
        # ===================================================================================================================================
        # Concatenate X & Y in train_data
        train_data = pd.concat([pd.DataFrame(self.X).reset_index(drop=True), pd.DataFrame(self.Y).reset_index(drop=True)], axis=1)
        # print(train_data.columns)
        listOutlier = ['age']
        # Handle outlier
        train_data = self.handle_outliers(listOutlier, train_data)
        # ===================================================================================================================================
        # No duplicate
        # print(train_data.head())
        # print(train_data.isnull().sum())
        # drop rows with null values 134 in year column
        # Replace null values with mode for the 'year' column
        # mode_year = train_data['year'].mode()[0]
        # self.saveLoad.saveModel(mode_year, 'modeYear')
        # train_data['year'].fillna(mode_year, inplace=True)
        train_data.dropna(inplace=True)

        # Split the dataframe train_data back into X and Y
        self.X = train_data.iloc[:, :-1]
        self.Y = pd.DataFrame(train_data.iloc[:, -1])
        # ===================================================================================================================================
        # No constant column or unique column
        # Encode gender & title & zip-code
        encodeList = ['title', 'gender', 'zip-code']
        self.encode(encodeList)
        # Save encoder Models
        self.saveLoad.saveModel(self.encoder_dict, 'EncodeValues')
        # ===================================================================================================================================
        # Scale Numerical Data
        self.scale_data()
        # Save scale model
        self.saveLoad.saveModel(self.scaler, 'scalingValues')

        return self.X, self.Y

    def testDataCleaning(self):
        # Load oneHotEncode of genres
        oneHotEncode_dict = self.saveLoad.loadModel('genresEncode')
        for columnName in oneHotEncode_dict.keys():
            # get the encoded column names for the current column
            encoded_cols = oneHotEncode_dict.get(columnName)
            # 'Apply one-hot encoding to columnName column'
            df_encoded = self.X[columnName].str.get_dummies(', ')
            df_encoded = df_encoded.reindex(columns=encoded_cols, fill_value=0)
            # Concatenate encoded columns to original dataframe X
            self.X = pd.concat([self.X, df_encoded], axis=1)
            # Drop columnName
            self.X.drop(columnName, axis=1, inplace=True)
        # ===================================================================================================================================
        # Load mode of year column
        mode_year = self.saveLoad.loadModel('modeYear')
        # Fill null values of year column with mode
        self.X['year'] = self.X['year'].fillna(mode_year)
        # ===================================================================================================================================
        # Load labelEncode_dict
        labelEncode_dict = self.saveLoad.loadModel('EncodeValues')
        for feature in labelEncode_dict.keys():
            encode = labelEncode_dict[feature]
            # Handle unseen values with 'other' value
            self.X[feature] = self.X[feature].map(lambda s: 'other' if s not in encode.classes_ else s)
            le_classes = encode.classes_.tolist()
            bisect.insort_left(le_classes, 'other')
            encode.classes_ = le_classes
            # Implement encode on feature
            self.X[feature] = encode.transform(self.X[feature])
        # ===================================================================================================================================
        # Load scaling model
        scalar = self.saveLoad.loadModel('scalingValues')
        age_col = self.X['age'].values.reshape(-1, 1)
        # Apply scaling
        self.X['age'] = scalar.transform(age_col)

        return self.X
