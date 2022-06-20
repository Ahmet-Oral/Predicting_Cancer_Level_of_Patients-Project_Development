import sys

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Plotting Functions
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest  # Feature Selector
from sklearn.feature_selection import f_classif  # F-ratio statistic for categorical values

# Aesthetics Update Seaborn for Plotting

import seaborn as sns

sns.set_style('ticks')  # No grid with ticks

import os

cancer = pd.read_excel('dataset.xlsx', header=None, engine='openpyxl')
# To use the data we will need to fix the header
new_header = cancer.iloc[0]  # Reads the first row which contains the headers
cancer = cancer[1:]  # Slices the rest of the data frame from header
cancer.columns = new_header  # Sets the header labels
cancer.info()
cancer.head()
cancer_label = cancer.columns
for label in cancer_label:
    print('***', label, 'labels:', cancer[label].unique())


def Plotter(plot, x_label, y_label, x_rot=None, y_rot=None, fontsize=12, fontweight=None, legend=True, save=False,
            save_name=None):

    # Ticks
    ax.tick_params(direction='out', length=5, width=3, colors='k',
                   grid_color='k', grid_alpha=1, grid_linewidth=2)
    plt.xticks(fontsize=fontsize, fontweight=fontweight, rotation=x_rot)
    plt.yticks(fontsize=fontsize, fontweight=fontweight, rotation=y_rot)

    # Legend
    if legend == True:
        plt.legend()
    else:
        ax.legend().remove()

    # Labels
    plt.xlabel(x_label, fontsize=fontsize, fontweight=fontweight, color='k')
    plt.ylabel(y_label, fontsize=fontsize, fontweight=fontweight, color='k')

    # Removing Spines and setting up remianing, preset prior to use.
    ax.spines['top'].set_color(None)
    ax.spines['right'].set_color(None)
    ax.spines['bottom'].set_color('k')
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_color('k')
    ax.spines['left'].set_linewidth(3)

    if save == True:
        plt.savefig(save_name)

fig, ax=plt.subplots()#Required outside of function. This needs to be activated first when plotting in every code block
plot=sns.scatterplot(data=cancer, x='Coughing of Blood',y='Passive Smoker', hue='Level', palette=['darkblue','darkred','darkgreen'], s=50, marker='o')#Count plot
Plotter(plot, 'Coughing of Blood', 'Passive Smoker', legend=True, save=True, save_name='Level Dependence on Blood Cough and Passive Smoking.png')#Plotter function for aesthetics
plot

# Feature Selection
X = cancer.drop(['Level', 'Patient Id'], axis=1)
Y = cancer['Level']
bestfeatures = SelectKBest(score_func=f_classif, k='all')
fit = bestfeatures.fit(X, Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
# concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Feature', 'Score']  # naming the dataframe columns

# Visualize the feature scores
fig, ax = plt.subplots(figsize=(7, 7))
plot = sns.barplot(data=featureScores, x='Score', y='Feature', palette='viridis', linewidth=0.5, saturation=2,
                   orient='h')
Plotter(plot, 'Score', 'Feature', legend=False, save=True,
        save_name='Feature Importance.png')  # Plotter function for aesthetics
plot

# Selection method
selection = featureScores[featureScores['Score'] >= 200]  # Selects features that scored more than 200
selection = list(selection['Feature'])  # Generates the features into a list
selection.append('Level')  # Adding the Level string to be used to make new data frame
new_cancer = cancer[selection]  # New dataframe with selected features
new_cancer.head()  # Lets take a look at the first 5

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(new_cancer.drop(['Level'], axis=1), new_cancer['Level'],
                                                    test_size=0.25, random_state=0)

# Checking the shapes
print("X_train shape :", X_train.shape)
print("Y_train shape :", y_train.shape)
print("X_test shape :", X_test.shape)
print("Y_test shape :", y_test.shape)

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)  # Scaling and fitting the training set to a model
X_test_scaled = scaler.transform(X_test)  # Transformation of testing set based off of trained scaler model

from sklearn.svm import SVC  # Classifier
# Packages
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV  # Paramterizers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Accuracy metrics
import itertools  # Used for iterations


def Searcher(estimator, param_grid, search, train_x, train_y, test_x, test_y, label=None):

    try:
        if search == "grid":
            clf = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring=None,
                n_jobs=-1,
                cv=10,  # Cross-validation at 10 replicates
                verbose=0,
                return_train_score=True
            )
        elif search == "random":
            clf = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                n_iter=10,
                n_jobs=-1,
                cv=10,
                verbose=0,
                random_state=1,
                return_train_score=True
            )
    except:
        print('Search argument has to be "grid" or "random"')
        sys.exit(0)  # Exits program if not grid or random

    # Fit the model
    clf.fit(X=train_x, y=train_y)

    # Testing the model

    try:
        if search == 'grid':
            cfmatrix = confusion_matrix(
                y_true=test_y, y_pred=clf.predict(test_x))

            # Defining prints for accuracy metrics of grid
            print("**Grid search results of", label, "**")
            print("The best parameters are:", clf.best_params_)
            print("Best training accuracy:\t", clf.best_score_)
            print('Classification Report:')
            print(classification_report(y_true=test_y, y_pred=clf.predict(test_x))
                  )
        elif search == 'random':
            cfmatrix = confusion_matrix(
                y_true=test_y, y_pred=clf.predict(test_x))

            # Defining prints for accuracy metrics of grid

            print("**Random search results of", label, "**")
            print("The best parameters are:", clf.best_params_)
            print("Best training accuracy:\t", clf.best_score_)
            print('Classification Report:')
            print(classification_report(y_true=test_y, y_pred=clf.predict(test_x))
                  )
    except:
        print('Search argument has to be "grid" or "random"')
        sys.exit(0)  # Exits program if not grid or random

    return clf, cfmatrix;  # Returns a trained classifier with best parameters


def plot_confusion_matrix(cm, label, color=None, title=None):
    classes = sorted(label)
    plt.imshow(cm, interpolation='nearest', cmap=color)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    thresh = cm.mean()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black")


svm_param = {
    "C": [.01, .1, 1, 5, 10, 100],  # Specific parameters to be tested at all combinations
    "gamma": [0, .01, .1, 1, 5, 10, 100],
    "kernel": ["rbf", "linear""poly"],
    "random_state": [1]}

# Randomized Grid Search SVM Parameters
svm_dist = {
    "C": np.arange(0.01, 2, 0.01),  # By using np.arange it will select from randomized values
    "gamma": np.arange(0, 1, 0.01),
    "kernel": ["rbf", "linear""poly"],
    "random_state": [1]}


# Grid Search SVM
svm_grid, cfmatrix_grid = Searcher(SVC(), svm_param, "grid", X_train_scaled, y_train, X_test_scaled, y_test,
                                   label='SVC Grid')

print('_____' * 20)  # Spacer

# Random Search SVM
svm_rand, cfmatrix_rand = Searcher(SVC(), svm_dist, "random", X_train_scaled, y_train, X_test_scaled, y_test,
                                   label='SVC Random')

# Plotting the confusion matrices
plt.subplots(1, 2)
plt.subplots_adjust(left=-0.5, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
plot_confusion_matrix(cfmatrix_rand, title='Random Search Confusion Matrix', label=new_cancer['Level'].unique(),
                      color=plt.cm.Greens)  # grid matrix function
plt.subplot(121)
plot_confusion_matrix(cfmatrix_grid, title='Grid Search Confusion Matrix', label=new_cancer['Level'].unique(),
                      color=plt.cm.Blues)  # randomized matrix function

plt.savefig('confusion.png')
