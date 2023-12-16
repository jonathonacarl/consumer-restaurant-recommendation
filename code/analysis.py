import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from processing import get_clean_data


def prepare_data():

    df = get_clean_data()
    min_count = df['userID'].value_counts().min()
    df = df.groupby('userID').head(min_count)

    restaurant_features = ['userID', 'placeID'] + \
        [c for c in df.columns if c.startswith(
            'R')] + [c for c in df.columns if 'rating' in c]
    user_features = ['userID'] + [c for c in df.columns if c.startswith('C')]

    df_user = df[user_features]
    df_user = df_user.drop_duplicates(subset='userID')
    df_restaurant = df[restaurant_features]
    restaurant_features.remove('userID')
    restaurant_features.remove('placeID')
    user_features.remove('userID')

    # Add a column indicating the set of restaurant features (e.g., 'restaurant 1', 'restaurant 2', 'restaurant 3')
    df_restaurant['restaurant_set'] = 'restaurant ' + \
        (df_restaurant.groupby('userID').cumcount() % 3 + 1).astype(str)

    # Pivot the DataFrame
    df_pivoted = df_restaurant.pivot_table(index=['userID'], columns=[
                                           'restaurant_set'], values=['userID', 'placeID'] + restaurant_features, aggfunc='first')

    # Flatten the MultiIndex columns
    df_pivoted.columns = [
        f"{col[1]} {col[0]}" for col in df_pivoted.columns.values]

    df_pivoted = pd.merge(df_pivoted, df_user, on='userID')

    imputer = KNNImputer(n_neighbors=2)
    columns_to_impute = df_pivoted.columns.difference(
        ['userID'] + [c for c in df_pivoted.columns if 'placeID' in c])
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_pivoted[columns_to_impute]), columns=columns_to_impute)

    df_pivoted[columns_to_impute] = df_imputed.set_index(df_pivoted.index)

    column_order = ['userID'] + user_features
    restaurant_columns = [
        c for c in df_pivoted.columns if 'R' in c or 'rating' in c]
    for num in sorted(set([col.split(" ")[1] for col in restaurant_columns]), key=lambda x: int(x)):
        column_order.append(f'restaurant {num} placeID')
        column_order.extend(
            [feature for feature in restaurant_columns if num in feature])

    df_pivoted = df_pivoted[column_order]

    return df_pivoted


def get_panel_data():
    df = get_clean_data()

    imputer = KNNImputer(n_neighbors=2)
    columns_to_impute = df.columns.difference(['userID'])
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df[columns_to_impute]), columns=columns_to_impute, index=df.index)

    return df_imputed, get_model_inputs(df_imputed)


def get_model_inputs(df, simple=False):

    x = df.drop(columns=[
                c for c in df.columns if 'userID' in c or 'placeID' in c or 'rating' in c])
    y = df[[c for c in df.columns if 'rating' in c]]

    if simple:
        x = x.drop(columns=[
            c for c in x.columns if '2' in c or '3' in c])
        y = y[[c for c in y.columns if '1' in c and '_' not in c]]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42)

    dfs = [x_train, x_val, x_test, y_train, y_val, y_test]
    for d in dfs:
        d.reset_index(inplace=True)
        d.drop(columns=['index'], inplace=True)
    return x_train, x_val, x_test, y_train, y_val, y_test


"""
All credit goes to Dennis Trimarchi for this confusion matrix function.
You can find the Github repository here: https://github.com/DTrimarchi10/confusion_matrix/tree/master
"""


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.

    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(
            value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(
        group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar,
                xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

    plt.show()


def standardize_data(x_train, x_val, x_test):
    scaler = StandardScaler()

    # normalize only on the training set to avoid overfitting
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    return x_train, x_val, x_test


def update_thresholds(y_hat, thresholds=[]):

    if len(thresholds) != 2:
        print("please provide 3 thresholds in order to update yhat suitably")
        return None

    first, second = thresholds[0], thresholds[1]
    y_hat = np.where(y_hat < first, 0,
                     np.where(y_hat < second, 1, 2))
    return y_hat


def get_metrics(y_train=None, y_train_pred=None, y_val=None, y_val_pred=None,
                y_test=None, y_test_pred=None, outcome_name="rating",
                conf_train=False, conf_val=False, conf_test=False,
                model_type="Linear Regression"):

    if y_train is not None and y_train_pred is not None:

        if conf_train:

            cm_train = confusion_matrix(
                y_true=y_train[outcome_name], y_pred=y_train_pred, labels=[0, 1, 2])

            make_confusion_matrix(cm_train, group_names=[
                                  '0', '1', '2'], title=f'{model_type} (Testing) Predictions', percent=False)

        train_accuracy = accuracy_score(y_train[outcome_name], y_train_pred)
        train_precision = precision_score(
            y_train[outcome_name], y_train_pred, average=None)
        train_recall = recall_score(
            y_train[outcome_name], y_train_pred, average=None)
        train_f1score = f1_score(
            y_train[outcome_name], y_train_pred, average=None)

        # Training Stats
        print(f"Training Accuracy: {train_accuracy}")
        print(f"Training Precision: {train_precision}")
        print(f"Training Recall: {train_recall}")
        print(f"Training F1 Score: {train_f1score}")

    if y_val is not None and y_val_pred is not None:

        if conf_val:

            cm_val = confusion_matrix(
                y_true=y_val[outcome_name], y_pred=y_val_pred, labels=[0, 1, 2])
            make_confusion_matrix(cm_val, group_names=[
                                  '0', '1', '2'], title=f'{model_type} (Testing) Predictions', percent=False)

        val_accuracy = accuracy_score(y_val[outcome_name], y_val_pred)
        val_precision = precision_score(
            y_val[outcome_name], y_val_pred, average=None)
        val_recall = recall_score(
            y_val[outcome_name], y_val_pred, average=None)
        val_f1score = f1_score(y_val[outcome_name], y_val_pred, average=None)

        # Validation Stats
        print(f"Validation Accuracy: {val_accuracy}")
        print(f"Validation Precision: {val_precision}")
        print(f"Validation Recall: {val_recall}")
        print(f"Validation F1 Score: {val_f1score}")

    if y_test is not None and y_test_pred is not None:

        if conf_test:

            cm_test = confusion_matrix(
                y_true=y_test[outcome_name], y_pred=y_test_pred, labels=[0, 1, 2])
            make_confusion_matrix(cm_test, group_names=[
                                  '0', '1', '2'], title=f'{model_type} (Testing) Predictions', percent=False)

        test_accuracy = accuracy_score(y_test[outcome_name], y_test_pred)
        test_precision = precision_score(
            y_test[outcome_name], y_test_pred, average=None)
        test_recall = recall_score(
            y_test[outcome_name], y_test_pred, average=None)
        test_f1score = f1_score(
            y_test[outcome_name], y_test_pred, average=None)

        # Testing Stats
        print(f"Testing Accuracy: {test_accuracy}")
        print(f"Testing Precision: {test_precision}")
        print(f"Testing Recall: {test_recall}")
        print(f"Testing F1 Score: {test_f1score}")


def train_decision_tree_regressor(x_train, y_train, x_val=None, y_val=None,
                                  outcome_name='rating', conf_train=False,
                                  conf_val=False, conf_test=False,
                                  model_type="Linear Regression"):

    tree_reg = DecisionTreeRegressor(random_state=42)

    param_grid = {
        'max_depth': [None],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    grid_search = GridSearchCV(
        tree_reg, param_grid, cv=5, scoring='neg_mean_squared_error')

    grid_search.fit(x_train, y_train[outcome_name])
    best_params = grid_search.best_params_
    print(f'optimal model parameters: {best_params}')
    model = grid_search.best_estimator_

    model.fit(x_train, y_train[outcome_name])
    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)

    # these thresholds ensure the predictions from our model maintain the true distribution of outcomes
    thresholds = [0.8, 1.35]

    y_train_pred = update_thresholds(y_train_pred, thresholds)

    y_val_pred = update_thresholds(y_val_pred, thresholds)
    get_metrics(y_train=y_train, y_train_pred=y_train_pred,
                y_val=y_val, y_val_pred=y_val_pred, outcome_name=outcome_name,
                conf_train=conf_train, conf_val=conf_val,
                conf_test=conf_test, model_type=model_type)

    return model, thresholds


def train_random_forest_regressor(x_train, y_train, x_val=None, y_val=None,
                                  outcome_name='rating', conf_train=False,
                                  conf_val=False, conf_test=False,
                                  model_type="Linear Regression"):

    param_grid = {
        'n_estimators': [i for i in range(50, 550, 50)],
        'max_features': ['log2', 'sqrt'],
        'max_depth': [None],
    }

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    grid_search.fit(x_train, y_train[outcome_name])
    best_params = grid_search.best_params_
    print(f'optimal model parameters: {best_params}')
    model = grid_search.best_estimator_
    model.fit(x_train, y_train[outcome_name])

    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)

    # these thresholds ensure the predictions from our model maintain the true distribution of outcomes
    thresholds = [0.8, 1.35]

    y_train_pred = update_thresholds(y_train_pred, thresholds)
    y_val_pred = update_thresholds(y_val_pred, thresholds)
    get_metrics(y_train=y_train, y_train_pred=y_train_pred,
                y_val=y_val, y_val_pred=y_val_pred, outcome_name=outcome_name,
                conf_train=conf_train, conf_val=conf_val,
                conf_test=conf_test, model_type=model_type)

    return model, thresholds


def train_one_versus_rest(x_train, y_train, x_val=None, y_val=None,
                          outcome_name='rating', conf_train=False,
                          conf_val=False, conf_test=False,
                          model_type="Linear Regression"):

    param_grid = {
        'estimator__n_estimators': [i for i in range(50, 550, 50)],
        'estimator__max_features': ['log2', 'sqrt'],
        'estimator__max_depth': [None],
    }

    model = OneVsRestClassifier(RandomForestRegressor(random_state=42))

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    grid_search.fit(x_train, y_train[outcome_name])
    best_params = grid_search.best_params_
    print(f'optimal model parameters: {best_params}')
    model = grid_search.best_estimator_

    model.fit(x_train, y_train[outcome_name])

    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)
    get_metrics(y_train=y_train, y_train_pred=y_train_pred,
                y_val=y_val, y_val_pred=y_val_pred, outcome_name=outcome_name,
                conf_train=conf_train, conf_val=conf_val,
                conf_test=conf_test, model_type=model_type)

    return model


def train_fixed_effects(x_train, y_train, x_val=None, y_val=None,
                        outcome_name='rating', conf_train=False,
                        conf_val=False, conf_test=False,
                        model_type="Linear Regression"):

    # estimates obtained via Stata reghdfe
    ests = {
        'rating': np.array([
            -1.368701,    # cactivity
            1.043331,     # cambience
            0.4957121,    # cbudget
            0.7307377,    # cchildren
            0,            # ccolor (omitted)
            0,            # ccuisine (omitted)
            0.8277519,    # cdress
            0,            # cdrink (omitted)
            0,            # cinterest (omitted)
            -4.4828,      # cmaritalstatus
            -3.566204,    # cpayment
            0,            # cpersonality (omitted)
            0,            # creligion (omitted)
            0,            # csmoker (omitted)
            -0.1141879,   # ctransport
            0,            # raccessibility (omitted)
            0,            # ralcohol (omitted)
            0,            # rambience (omitted)
            0,            # rarea (omitted)
            0.0017656,    # rcuisine
            0,            # rdress (omitted)
            0,            # rfranchise (omitted)
            0,            # rotherservices (omitted)
            0,            # rparking (omitted)
            -0.7914075,   # rpayment
            0,            # rprice (omitted)
            0,            # rsmoking (omitted)
            -0.0528389,   # averagehoursopen
            0,            # birth_year (omitted)
            0,            # height (omitted)
            0,            # weight (omitted)
            0,            # numdaysopen (omitted)
        ]),
        'service_rating': np.array([
            -0.5658588,   # cactivity
            0.5597006,    # cambience
            0.5194619,    # cbudget
            1.784924,     # cchildren
            0,            # ccolor (omitted)
            0,            # ccuisine (omitted)
            0.4107101,    # cdress
            0,            # cdrink (omitted)
            0,            # cinterest (omitted)
            -3.437101,    # cmaritalstatus
            -4.062977,    # cpayment
            0,            # cpersonality (omitted)
            0,            # creligion (omitted)
            0,            # csmoker (omitted)
            -0.3255292,   # ctransport
            0,            # raccessibility (omitted)
            0,            # ralcohol (omitted)
            0,            # rambience (omitted)
            0,            # rarea (omitted)
            0.0032809,    # rcuisine
            0,            # rdress (omitted)
            0,            # rfranchise (omitted)
            0,            # rotherservices (omitted)
            0,            # rparking (omitted)
            -1.314151,    # rpayment
            0,            # rprice (omitted)
            0,            # rsmoking (omitted)
            0.0258762,    # averagehoursopen
            0,            # birth_year (omitted)
            0,            # height (omitted)
            0,            # weight (omitted)
            0             # numdaysopen (omitted)
        ]),
        'food_rating': np.array([
            -0.2452921,   # cactivity
            0.4277757,    # cambience
            0.3620904,    # cbudget
            1.037878,     # cchildren
            0,            # ccolor (omitted)
            0,            # ccuisine (omitted)
            0.0999966,    # cdress
            0,            # cdrink (omitted)
            0,            # cinterest (omitted)
            -2.871895,    # cmaritalstatus
            -3.923935,    # cpayment
            0,            # cpersonality (omitted)
            0,            # creligion (omitted)
            0,            # csmoker (omitted)
            0.1052153,    # ctransport
            0,            # raccessibility (omitted)
            0,            # ralcohol (omitted)
            0,            # rambience (omitted)
            0,            # rarea (omitted)
            -0.001764,    # rcuisine
            0,            # rdress (omitted)
            0,            # rfranchise (omitted)
            0,            # rotherservices (omitted)
            0,            # rparking (omitted)
            0.985995,     # rpayment
            0,            # rprice (omitted)
            0,            # rsmoking (omitted)
            0.0589095,    # averagehoursopen
            0,            # birth_year (omitted)
            0,            # height (omitted)
            0,            # weight (omitted)
            0             # numdaysopen (omitted)
        ])
    }

    y_train_pred = x_train @ ests[outcome_name]
    y_val_pred = x_val @ ests[outcome_name]

    # these thresholds ensure the predictions from our model maintain the true distribution of outcomes
    thresholds = [-1, 2.9]

    y_train_pred = update_thresholds(y_train_pred, thresholds)
    y_val_pred = update_thresholds(y_val_pred, thresholds)
    get_metrics(y_train=y_train, y_train_pred=y_train_pred,
                y_val=y_val, y_val_pred=y_val_pred, outcome_name=outcome_name,
                conf_train=conf_train, conf_val=conf_val,
                conf_test=conf_test, model_type=model_type)

    return ests[outcome_name], thresholds


def train_gradient_boosting_regressor(x_train, y_train, x_val=None, y_val=None,
                                      outcome_name='rating', conf_train=False,
                                      conf_val=False, conf_test=False,
                                      model_type="Linear Regression"):

    booster = GradientBoostingRegressor()

    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 0.9, 1.0]
    }

    grid_search = GridSearchCV(estimator=booster, param_grid=param_grid,
                               cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    grid_search.fit(x_train, y_train[outcome_name])
    best_params = grid_search.best_params_
    print(f'optimal model parameters: {best_params}')
    booster = grid_search.best_estimator_

    booster.fit(x_train, y_train[outcome_name])
    y_train_pred = booster.predict(x_train)
    y_val_pred = booster.predict(x_val)

    # these thresholds ensure the predictions from our model maintain the true distribution of outcomes
    thresholds = [0.75, 1.5]
    y_train_pred = update_thresholds(y_train_pred, thresholds)
    y_val_pred = update_thresholds(y_val_pred, thresholds)
    get_metrics(y_train=y_train, y_train_pred=y_train_pred,
                y_val=y_val, y_val_pred=y_val_pred, outcome_name=outcome_name,
                conf_train=conf_train, conf_val=conf_val,
                conf_test=conf_test, model_type=model_type)

    return booster, thresholds


def train_linear_regression(x_train, y_train, x_val=None, y_val=None,
                            outcome_name='rating', conf_train=False,
                            conf_val=False, conf_test=False,
                            model_type="Linear Regression"):

    reg = LinearRegression()
    reg.fit(x_train, y_train[outcome_name])
    y_train_pred = reg.predict(x_train)
    y_val_pred = reg.predict(x_val)

    # these thresholds ensure the predictions from our model maintain the true distribution of outcomes
    thresholds = [0.8, 1.25]
    y_train_pred = update_thresholds(y_train_pred, thresholds)
    y_val_pred = update_thresholds(y_val_pred, thresholds)
    get_metrics(y_train=y_train, y_train_pred=y_train_pred,
                y_val=y_val, y_val_pred=y_val_pred, outcome_name=outcome_name,
                conf_train=conf_train, conf_val=conf_val,
                conf_test=conf_test, model_type=model_type)

    return reg, thresholds


def train_ridge_regression(x_train, y_train, x_val=None, y_val=None,
                           outcome_name='rating', conf_train=False,
                           conf_val=False, conf_test=False,
                           model_type="Linear Regression"):

    l2 = Ridge()

    param_grid = {
        'alpha': [1.0, 10.0, 50.0, 100.0, 200.0],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }

    grid_search = GridSearchCV(
        l2, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(x_train, y_train)
    best_params = grid_search.best_params_
    print(f'optimal model paramaters: {best_params}')

    l2 = grid_search.best_estimator_

    l2.fit(x_train, y_train[outcome_name])
    y_train_pred = l2.predict(x_train)
    y_val_pred = l2.predict(x_val)

    # these thresholds ensure the predictions from our model maintain the true distribution of outcomes
    thresholds = [0.8, 1.25]
    y_train_pred = update_thresholds(y_train_pred, thresholds)
    y_val_pred = update_thresholds(y_val_pred, thresholds)
    get_metrics(y_train=y_train, y_train_pred=y_train_pred,
                y_val=y_val, y_val_pred=y_val_pred, outcome_name=outcome_name,
                conf_train=conf_train, conf_val=conf_val,
                conf_test=conf_test, model_type=model_type)

    return l2, thresholds


def main(outcome_name='rating'):

    # for ease, only keep the first restaurant ranked by a user
    df_wide = prepare_data()
    df_wide = df_wide[[
        col for col in df_wide.columns if 'restaurant 2' not in col and 'restaurant 3' not in col]]
    df_wide = df_wide.rename(
        columns=lambda x: x.replace('restaurant 1', '').strip())

    x_train, x_val, x_test, y_train, y_val, y_test = get_model_inputs(df_wide)
    x_train, x_val, x_test = standardize_data(x_train, x_val, x_test)

    """
    ***** Linear Regression *****
    """

    print("*******************************\n ** LINEAR REGRESSION ** \n*******************************")

    reg, thresholds = train_linear_regression(
        x_train, y_train, x_val, y_val, outcome_name,
        model_type="Linear Regression")

    y_test_pred = reg.predict(x_test)
    y_test_pred = update_thresholds(y_test_pred, thresholds)

    get_metrics(y_test=y_test, y_test_pred=y_test_pred,
                outcome_name=outcome_name, conf_test=True,
                model_type="Linear Regression")

    """
    ***** L2 Regularization *****
    """

    print("*******************************\n ** L2 REGULARIZATION ** \n*******************************")

    l2reg, thresholds = train_ridge_regression(
        x_train, y_train, x_val, y_val, outcome_name,
        model_type="Ridge Regression")

    y_test_pred = l2reg.predict(x_test)
    y_test_pred = update_thresholds(y_test_pred, thresholds)
    get_metrics(y_test=y_test, y_test_pred=y_test_pred,
                outcome_name=outcome_name, conf_test=True,
                model_type="Ridge Regression")

    """
    ***** Gradient Boosting Regressor *****
    """

    print("*******************************\n ** GRADIENT BOOSTING REGRESSOR ** \n*******************************")

    booster, thresholds = train_gradient_boosting_regressor(
        x_train, y_train, x_val, y_val, outcome_name=outcome_name,
        model_type="Gradient Boosting Regressor")

    y_test_pred = booster.predict(x_test)
    y_test_pred = update_thresholds(y_test_pred, thresholds)
    get_metrics(y_test=y_test, y_test_pred=y_test_pred,
                outcome_name=outcome_name, conf_test=True,
                model_type="Gradient Boosting Regressor")

    """
    ***** Decision Tree model *****
    """

    print("*******************************\n ** DECISION TREE REGRESSOR ** \n*******************************")

    tree_reg, thresholds = train_decision_tree_regressor(
        x_train, y_train, x_val, y_val,
        outcome_name=outcome_name, model_type="Decision Tree Regressor")

    # Only run once we've maximized validation accuracy!
    y_test_pred = tree_reg.predict(x_test)
    y_test_pred = update_thresholds(y_test_pred, thresholds)

    get_metrics(y_test=y_test, y_test_pred=y_test_pred,
                outcome_name=outcome_name, conf_test=True,
                model_type="Decision Tree Regressor")

    """
    ***** Random Forest Model *****
    """

    print("*******************************\n ** RANDOM FOREST REGRESSOR ** \n*******************************")

    forest_reg, thresholds = train_random_forest_regressor(
        x_train, y_train, x_val, y_val,
        outcome_name=outcome_name, model_type="Random Forest Regressor")

    # Only run once we've maximized validation accuracy!
    y_test_pred = forest_reg.predict(x_test)
    y_test_pred = update_thresholds(y_test_pred, thresholds)

    get_metrics(y_test=y_test, y_test_pred=y_test_pred,
                outcome_name=outcome_name, conf_test=True,
                model_type="Random Forest Regressor")

    """
    ***** One versus Rest model *****
    """

    print("*******************************\n ** ONE VERSUS REST ** \n*******************************")

    ovr = train_one_versus_rest(
        x_train, y_train, x_val, y_val,
        outcome_name=outcome_name, model_type="OvR")

    # Only run once we've maximized validation accuracy!
    y_test_pred = ovr.predict(x_test)
    y_test_pred = update_thresholds(y_test_pred, [0.8, 1.35])
    get_metrics(y_test=y_test, y_test_pred=y_test_pred,
                outcome_name=outcome_name, conf_test=True,
                model_type="OvR")

    """
    ***** Fixed Effects model *****
    """

    print("*******************************\n ** FIXED EFFECTS ** \n*******************************")

    df, (x_train, x_val, x_test, y_train, y_val, y_test) = get_panel_data()
    x_train, x_val, x_test = standardize_data(x_train, x_val, x_test)
    fe_params, thresholds = train_fixed_effects(
        x_train, y_train, x_val, y_val,
        outcome_name=outcome_name, model_type="Fixed Effects")

    # Only run once we've maximized validation accuracy!
    y_test_pred = x_test @ fe_params
    y_test_pred = update_thresholds(y_test_pred, thresholds)
    get_metrics(y_test=y_test, y_test_pred=y_test_pred,
                outcome_name=outcome_name, conf_test=True,
                model_type="Fixed Effects")


if __name__ == '__main__':

    main(outcome_name='rating')
