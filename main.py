import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import Preprocessing
import Regression
import saveLoadData
import test


def getInputOutput(mergedData):

    mergedData = mergedData.copy()

    # Get output data
    Y = pd.DataFrame(mergedData['rating'])

    mergedData.drop('rating', axis=1, inplace=True)

    # Get input data
    X = mergedData.iloc[:,:]

    return X, Y

def Fcategorical(input, output):
    # Flatten the output array to convert it to a 1D array
    output_flat = output.to_numpy().ravel()
    # Create a SelectKBest object to select features with the k highest F-values
    k_best = SelectKBest(f_regression, k=6)
    # Fit the object to extract the top k features
    X_new = k_best.fit_transform(input, output_flat)
    # Get the names of the selected features
    mask = k_best.get_support()  # Get a boolean mask of the selected features
    top_feature_categorical = input.columns[mask]  # Index the feature names using the mask
    input_data = pd.DataFrame(X_new, columns=top_feature_categorical)

    return input_data

def calculateMSE(prediction, actualValue):
    # Display MSE value for feature
    print('Mean Square Error: ' + str(mean_squared_error(actualValue, prediction)))

def calculateR2Score(prediction, actualValue):
    # Display R2 Score
    print(f'R2 Score : {r2_score(actualValue, prediction)}')

def getSimilarMovies(movie_data, movie_name):
    pivot_matrix = movie_data.pivot_table(index='movieId', columns='userId', values='rating')
    ratings_matrix = pivot_matrix.fillna(0)
    movie_similarity = cosine_similarity(ratings_matrix)

    movie_id = movie_data[movie_data['title'] == movie_name]['movieId'].iloc[0]

    similar_movies = pd.DataFrame(movie_similarity[movie_id - 1], index=ratings_matrix.index, columns=['similarity'])
    similar_movies = similar_movies.sort_values(by='similarity', ascending=False)[1:11]

    recommended_movies = pd.merge(similar_movies, movie_data, on='movieId')
    recommended_movies = recommended_movies[['movieId', 'title']].drop_duplicates().reset_index(drop=True)
    plt_similarity_dendrogram(movie_similarity, ratings_matrix)
    result = []
    for index, row in recommended_movies.iterrows():
        movie_id = row['movieId']
        movie_title = row['title']
        movie_ratings = pivot_matrix.loc[movie_id, :]
        average_rating = movie_ratings.mean(skipna=True)
        result.append({"Movie": movie_title, "Average Rating": average_rating})
    return result

def get_age_matched_movies(user_id, merged_data):

    # Get the age of the input user
    input_user_age = merged_data.loc[merged_data["userId"] == user_id, "age"].iloc[0]

    # Get the IDs of all users with the same age as the input user
    same_age_user_ids = merged_data.loc[merged_data["age"] == input_user_age, "userId"].unique()

    # Get the movies watched by these users
    movies_watched = merged_data.loc[merged_data["userId"].isin(same_age_user_ids), "title"].unique()

    # Get the movies watched by the input user
    input_user_movies = merged_data.loc[merged_data["userId"] == user_id, "title"].unique()

    # Find the movies watched by the age-matched users that were not watched by the input user
    age_matched_movies = set(movies_watched) - set(input_user_movies)

    # Remove duplicates and return the list of movies
    return list(set(age_matched_movies))

def plt_predictions(predictions, act_values, title):
    fig, md = plt.subplots()
    md.scatter(act_values, predictions)
    md.plot([act_values.min(), act_values.max()], [act_values.min(), act_values.max()], 'k--', lw=3)
    md.set_xlabel('Actual values')
    md.set_ylabel('Predicted Values')
    md.set_title(title)
    plt.show()


def plt_ratings_distribution(predictions):
    fig, rd = plt.subplots()
    rd.hist(predictions, bins=15)
    rd.set_xlabel('Predicted Ratings')
    rd.set_ylabel('Count')
    rd.set_title('Predicted Ratings Distribution')
    plt.show()



def plt_similarity_matrix(recommended_movies):
    # Create a list of movie titles and their corresponding average ratings
    movies = [movie['Movie'] for movie in recommended_movies]
    ratings = [movie['Average Rating'] for movie in recommended_movies]
    plt.bar(movies, ratings)
    plt.xlabel('Movie Title')
    plt.ylabel('Average Rating')
    plt.title('Recommended Movies with Average Ratings')
    plt.xticks(rotation=90)
    plt.show()


def plt_similarity_dendrogram(movie_similarity,ratings_matrix):

    dendrogram = hierarchy.dendrogram(hierarchy.linkage(movie_similarity, method='ward'), labels=ratings_matrix.index)
    plt.title('Movie Similarity Dendrogram')
    plt.xlabel('Movie Title')
    plt.ylabel('Distance')
    plt.show()

if __name__=='__main__':

    saveLoad = saveLoadData.SaveLoadData()

    # create empty lists for each column
    movie_ids = []
    titles = []
    genres = []

    # Read data from csv file
    with open('movies.csv', 'r', encoding='ISO-8859-1') as f:
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
    ratingsData = pd.read_csv('ratings.csv', sep=';', encoding='ISO-8859-1')
    usersData = pd.read_csv('users.csv', sep=';', encoding='ISO-8859-1')

    moviesData.drop(index=0, inplace=True)
    moviesData['movieId'] = moviesData['movieId'].astype('int64')

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

    # Merge dataframes
    mergedData = pd.merge(ratingsData, moviesData, on='movieId')
    mergedData = pd.merge(mergedData, usersData, on='userId')

    # Extract the year from the title column using regex
    mergedData['year'] = mergedData['title'].str.extract(r"\((\d{4})\)$")

    # Remove the year from the title column using regex
    mergedData['title'] = mergedData['title'].str.replace(r"\(\d{4}\)$", "").str.strip()

    # Get X & Y from loaded data
    X, Y = getInputOutput(mergedData)

    # print(mergedData.head(12))

    # Splitting the X,Y into the Training set(80%) and Test set(20%)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # Create object of Preprocess
    preprocess = Preprocessing.Preprocess(x_train, y_train)

    # Clean Training Data
    x_train, y_train = preprocess.trainDataCleaning()

    # Select effective feature
    categorical_data = pd.DataFrame(x_train.drop('age',axis = 1))

    catF = Fcategorical(categorical_data,y_train)

    catF = pd.concat([pd.DataFrame(catF.reset_index(drop=True)), pd.DataFrame(x_train['age'].reset_index(drop=True))], axis=1)

    # print(x_train.columns)
    saveLoad.saveModel(catF.columns, 'selectedFeatures')
    x_train = catF
    # print(x_train.columns)
    # ==========================================================================================================================
    regression = Regression.regression(x_train, y_train)
    y_poly, y_decision, enetPrediction, y_lasso, y_grad = regression.train()
    print("======================================================================================")
    print("=====================================Train============================================")
    print("======================================================================================")
    # print MSE
    print("Polynomial ==> ")
    calculateMSE(np.asarray(y_poly), np.asarray(y_train))
    calculateR2Score(y_poly, y_train)
    print("======================================================================================")
    print("decision Regression ==> ")
    calculateMSE(np.asarray(y_decision), np.asarray(y_train))
    calculateR2Score(y_decision, y_train)
    # print("======================================================================================")
    # print("RandomForest Regression ==> ")
    # calculateMSE(np.asarray(rfPrediction), np.asarray(y_train))
    # calculateR2Score(rfPrediction, y_train)
    print("======================================================================================")
    print("Elastic Regression ==> ")
    calculateMSE(np.asarray(enetPrediction), np.asarray(y_train))
    calculateR2Score(enetPrediction, y_train)
    print("======================================================================================")
    print("Lasso Regression ==> ")
    calculateMSE(np.asarray(y_lasso), np.asarray(y_train))
    calculateR2Score(y_lasso, y_train)
    # print("======================================================================================")
    # print("Ridge Regression ==> ")
    # calculateMSE(np.asarray(y_ridge), np.asarray(y_train))
    # calculateR2Score(y_ridge, y_train)
    print("======================================================================================")
    print("Gradient Boosting Regression  ==> ")
    calculateMSE(np.asarray(y_grad), np.asarray(y_train))
    calculateR2Score(y_grad, y_train)

    print("======================================================================================")
    print("=====================================Test=============================================")
    print("======================================================================================")


    test = test.TestPredict(x_test)
    x_test = test.preprocess()
    selectedFeatures = saveLoad.loadModel('selectedFeatures')
    xTest = x_test[selectedFeatures]
    polyp, desp, elasricp, lassop, boostp = test.predict(xTest)

    print("Polynomial ==> ")
    calculateMSE(np.asarray(polyp), np.asarray(y_test))
    calculateR2Score(polyp, y_test)
    print("======================================================================================")
    print("decision Regression ==> ")
    calculateMSE(np.asarray(desp), np.asarray(y_test))
    calculateR2Score(desp, y_test)
    # print("======================================================================================")
    # print("RandomForest Regression ==> ")
    # calculateMSE(np.asarray(randp), np.asarray(y_test))
    # calculateR2Score(randp, y_test)
    print("======================================================================================")
    print("Elastic Regression ==> ")
    calculateMSE(np.asarray(elasricp), np.asarray(y_test))
    calculateR2Score(elasricp, y_test)
    print("======================================================================================")
    print("Lasso Regression ==> ")
    calculateMSE(np.asarray(lassop), np.asarray(y_test))
    calculateR2Score(lassop, y_test)
    # print("======================================================================================")
    # print("Ridge Regression ==> ")
    # calculateMSE(np.asarray(ridgep), np.asarray(y_test))
    # calculateR2Score(ridgep, y_test)
    print("======================================================================================")
    print("Gradient Boosting Regression  ==> ")
    calculateMSE(np.asarray(boostp), np.asarray(y_test))
    calculateR2Score(boostp, y_test)

    # print(get_age_matched_movies(2, mergedData))

    # Plot a scatter plot of predicted vs. actual values for the best performing model
    plt_predictions(polyp, y_test, 'Polynomial Regression: Actual vs. Predicted Ratings')
    plt_predictions(desp, y_test, 'decision Regression: Actual vs. Predicted Ratings')
    plt_predictions(elasricp, y_test, 'Elastic Regression: Actual vs. Predicted Ratings')
    plt_predictions(lassop, y_test, 'Lasso Regression: Actual vs. Predicted Ratings')
    plt_predictions(boostp, y_test, 'Gradient Boosting Regression: Actual vs. Predicted Ratings')
    # --------------------------------------------------------------------------------------
    # Plot a bar chart of the predicted ratings distribution for the best performing model
    plt_ratings_distribution(polyp)
    plt_ratings_distribution(desp)
    plt_ratings_distribution(elasricp)
    plt_ratings_distribution(lassop)
    plt_ratings_distribution(boostp)
    # --------------------------------------------------------------------------------------

    movie_title = 'Jumanji'
    recommended_movies = getSimilarMovies(mergedData, movie_title)

    plt_similarity_matrix(recommended_movies)
