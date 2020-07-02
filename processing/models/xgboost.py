from processing.models.predictor import Predictor
from sklearn.ensemble import GradientBoostingRegressor
import misc.constants as cs

class SVR(Predictor):
    """
    The SVR predictor is based on Support Vector Regression.
    Parameters:
        - self.params["hist"], history length
        - self.params["kernel"], kernel to be used
        - self.params["C"], loss
        - self.params["epsilon"], wideness of the no-penalty tube
        - self.params["gamma"], kernel coefficient
        - self.params["shrinking"], wether or not to use the shrinkin heuristic
    """

    def fit(self):
        # get training data

        # https: // scikit - learn.org / stable / auto_examples / ensemble / plot_gradient_boosting_regression.html

        x, y, t = self._str2dataset("train")

        # define the model
        self.model = GradientBoostingRegressor(
            n_estimators=self.params["n_estimators"],
            criterion="mse",
            max_depth=self.params["max_depth"],
            min_samples_split=self.params["min_samples_split"],
            learning_rate=self.params["learning_rate"],
            loss=self.params["loss"],
            random_state=cs.seed
        )

        # fit the model
        self.model.fit(x, y)

    def predict(self, dataset):
        # get the data for which we make the predictions
        x, y, t = self._str2dataset(dataset)

        # predict
        y_pred = self.model.predict(x)
        y_true = y.values

        return self._format_results(y_true, y_pred, t)
