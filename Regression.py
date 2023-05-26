from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import saveLoadData

class regression:

    saveLoad  = saveLoadData.SaveLoadData()

    def __init__(self, x, y=None):
        self.X = x
        self.Y = y

    def train(self):
        poly = PolynomialFeatures(degree=3)
        # Transform the data to include polynomial features
        x_poly = poly.fit_transform(self.X)
        poly_model = LinearRegression()
        poly_model.fit(x_poly, self.Y)
        # predicting on training data-set
        y_poly = poly_model.predict(x_poly)
        # Save polynomial model
        self.saveLoad.saveModel(poly_model, 'LinearModel')
        self.saveLoad.saveModel(poly, 'PolynomialModel')

        # # Create a Random Forest regressor with 100 trees
        # rf = RandomForestRegressor(n_estimators=100)
        # # Fit the regressor to the training data
        # rf.fit(self.X, self.Y.to_numpy().ravel())
        # self.saveLoad.saveModel(rf, 'RandomForestRegressorModel')
        # # Make predictions on the test set
        # rfPrediction = rf.predict(self.X)

        # Fit the elastic net regression model
        alpha = 0.1  # regularization strength
        l1_ratio = 0.9  # balance between L1 and L2 regularization
        enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        enet.fit(self.X, self.Y)
        self.saveLoad.saveModel(enet, 'elastic')
        enetPrediction = enet.predict(self.X)

        # Create a decision tree regressor with maximum depth of 3
        regressor = DecisionTreeRegressor(max_depth=3)
        # Fit the regressor to the training data
        regressor.fit(self.X, self.Y)
        # Save DecisionTreeRegressorModel
        self.saveLoad.saveModel(regressor, 'DecisionTreeRegressorModel')
        # Make predictions on the test set
        y_decision = regressor.predict(self.X)

        # create a Lasso regression model
        lasso = linear_model.Lasso(alpha=0.1)
        # fit the model on the training data
        lasso.fit(self.X, self.Y)
        self.saveLoad.saveModel(lasso, 'LassoRegressionModel')
        # make predictions on the test data
        y_lasso = lasso.predict(self.X)

        # # Train the Ridge regression model
        # ridge = Ridge(alpha=1.0)
        # ridge.fit(self.X, self.Y)
        # self.saveLoad.saveModel(ridge, 'RidgeRegressionModel')
        # # Make predictions on the test data
        # y_ridge = ridge.predict(self.X)

        # Gradient Boosting Regression
        # Create the Gradient Boosting Regression model
        model = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, max_depth=3)
        # Train the model on the training data
        model.fit(self.X, self.Y.to_numpy().ravel())
        self.saveLoad.saveModel(model, 'GradientRegressionModel')
        # Make predictions on the testing data
        y_grad = model.predict(self.X)

        return y_poly, y_decision, enetPrediction, y_lasso, y_grad

    def test(self):
        polylTest = self.saveLoad.loadModel('LinearModel')
        polyTest = self.saveLoad.loadModel('PolynomialModel')
        transPoly = polyTest.transform(self.X)
        polyp = polylTest.predict(transPoly)

        # randTest =self.saveLoad.loadModel('RandomForestRegressorModel')
        # randp = randTest.predict(self.X)

        elasticTest = self.saveLoad.loadModel('elastic')
        elasricp = elasticTest.predict(self.X)

        desTest = self.saveLoad.loadModel('DecisionTreeRegressorModel')
        desp=desTest.predict(self.X)

        lassoTest = self.saveLoad.loadModel('LassoRegressionModel')
        lassop=lassoTest.predict(self.X)

        # ridgeTest = self.saveLoad.loadModel('RidgeRegressionModel')
        # ridgep= ridgeTest.predict(self.X)

        BoostingTest =self.saveLoad.loadModel('GradientRegressionModel')
        boostp = BoostingTest.predict(self.X)

        return polyp, desp, elasricp, lassop, boostp





