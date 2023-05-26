import Preprocessing
import Regression
import pandas as pd

class TestPredict:

    def __init__(self, xTest=None):
        self.X = xTest

    def readData(self):
        # create empty lists for each column
        movie_ids = []
        titles = []
        genres = []

        # Read data from csv file
        with open('', 'r', encoding='ISO-8859-1') as f:
            # read the file line by line
            for line in f:
                # split the row on the delimiter
                row_split = line.strip().split(';')

                # append the values to their corresponding lists
                movie_ids.append(row_split[0])
                if '(' in row_split[2] or ' ' in row_split[2]:
                    titles.append(row_split[1] + ':' + row_split[2])
                    genres.append("")
                else:
                    titles.append(row_split[1])
                    genres.append(row_split[2])

        moviesData = pd.DataFrame({'movieId': movie_ids, 'title': titles, 'genres': genres})
        ratingsData = pd.read_csv('', sep=';', encoding='ISO-8859-1')
        usersData = pd.read_csv('', sep=';', encoding='ISO-8859-1')

        moviesData.drop(index=0, inplace=True)
        moviesData['movieId'] = moviesData['movieId'].astype('int64')

        # Merge dataframes
        mergedData = pd.merge(ratingsData, moviesData, on='movieId')
        mergedData = pd.merge(mergedData, usersData, on='userId')

        # Extract the year from the title column using regex
        mergedData['year'] = mergedData['title'].str.extract(r"\((\d{4})\)$")

        # Remove the year from the title column using regex
        mergedData['title'] = mergedData['title'].str.replace(r"\(\d{4}\)$", "").str.strip()

        # Set input in self.X
        self.X = mergedData.iloc[:,:]

    def preprocess(self):
        # Create object of Preprocess
        preproces = Preprocessing.Preprocess(self.X)
        # Clean data
        self.X = preproces.testDataCleaning()

        return self.X

    def predict(self, xTest):
        # Create object of regression and set constructor to xTest
        regression = Regression.regression(xTest)
        # Get prediction of xtest
        polyp, desp, elasricp, lassop, boostp = regression.test()

        return polyp, desp, elasricp, lassop, boostp