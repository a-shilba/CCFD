# Author A. Shilba Alabass

from joblib import dump, load
import time
import numpy
from catboost import CatBoostRegressor
from numpy import fromstring
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, auc
from sklearn.metrics._plot import regression
from sklearn.model_selection import train_test_split  # Import train_test_split function
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import re
from catboost import Pool, CatBoostClassifier, cv
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve)
import matplotlib.pyplot as plt
from sklearn.svm._libsvm import predict_proba

csvColumn = ["Algorithm", "Test Size", "Round", "Accuracy", "Testing Time", "Training Time", "Precision", "Recall",
             "F1score", "Confusion Matrix", "True Positive Rate", "False Positive Rate", "AUC"]

df = pd.read_csv("creditcard.csv")
df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()
df = df.reset_index(drop=True)
df.shape

y = df["Class"]
del df['Class']
df.shape
x = df.loc[:, :]

resultList = []
scriptTime = time.time()

for testSize in range(3, 4):

    testSizeValue = "0." + str(testSize)
    testSizeValue = float(testSizeValue)
    print("the type: ", type, " - ", testSizeValue)

    for testRound in range(1, 2):
        model = None
        auc = None
        fpr = None
        tpr = None
        splitTime = time.time()
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=testSizeValue, random_state=1)
        splitTime = "%s" % (time.time() - splitTime)
        print("Split Time", splitTime)

        print("Lap Value: ", testSize, testRound, testSizeValue)

        start_time = time.time()

        algorithmName = "xGBoost"
        print(algorithmName)

        trainTime = time.time()
        model = None
        auc = None
        fpr = None
        tpr = None
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)
        trainTime = "%s" % (time.time() - trainTime)

        expected_y = y_test
        predicted_y = model.predict(X_test)

        accuracy = accuracy_score(y_test, predicted_y)

        print(algorithmName + " Model Accuracy: ", accuracy)

        modelName = "saved-saved-models/" + algorithmName + "-testSize-" + str(testSize).zfill(2) + "-round-" + str(
            testRound).zfill(2) + ".joblib"

        dump(model, modelName)

        predictedTime = time.time()
        predicted_y = model.predict(X_test)
        predictedTime = "%s" % (time.time() - predictedTime)

        conf_matrix = confusion_matrix(y_test, predicted_y)
        print(conf_matrix)
        print(type(conf_matrix))
        conf_matrix_str = conf_matrix.tobytes()
        # print(type(conf_matrix_str))
        # print(fromstring(conf_matrix_str, dtype=int))
        conf_matrix_str = fromstring(conf_matrix_str, dtype=int)
        # print(converted, type(converted))

        precision = precision_score(y_test, predicted_y)
        recall = recall_score(y_test, predicted_y)
        f1score = f1_score(y_test, predicted_y)

        print(f"Precision = {precision}")
        print(f"Recall = {recall}")
        print(f"F1 Score = {f1score}")

        # define metrics
        y_pred_proba = model.predict_proba(X_test)[::, 1]
        # _ represent the thresh hold
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        xGBoostAUC = metrics.roc_auc_score(y_test, y_pred_proba)

        # create ROC curve
        plt.clf()
        plt.plot(fpr, tpr, label="AUC-" + algorithmName + "=" + str(xGBoostAUC))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        figureName = "figures/" + "ROC-AUC-" + algorithmName + "-testSize-" + str(testSize).zfill(2) + "-round-" + str(
            testRound).zfill(2) + '.png'
        plt.savefig(figureName, bbox_inches='tight')

        resultList.append(
            [algorithmName, testSizeValue, testRound, accuracy, predictedTime, trainTime, precision, recall, f1score,
             conf_matrix_str, tpr, fpr, xGBoostAUC])

        print(" ")
        print(" ------------------------------- ")
        print(" ")


        algorithmName = "adaBoost"
        print(algorithmName)

        trainTime = time.time()
        model = None
        auc = None
        fpr = None
        tpr = None
        model = AdaBoostClassifier(n_estimators=50, learning_rate=1)
        model.fit(X_train, y_train)
        trainTime = "%s" % (time.time() - trainTime)

        expected_y = y_test
        predicted_y = model.predict(X_test)

        accuracy = accuracy_score(y_test, predicted_y)

        print(algorithmName + " Model Accuracy: ", accuracy)

        modelName = "saved-saved-models/" + algorithmName + "-testSize-" + str(testSize).zfill(2) + "-round-" + str(
            testRound).zfill(2) + ".joblib"

        dump(model, modelName)

        predictedTime = time.time()
        predicted_y = model.predict(X_test)
        predictedTime = "%s" % (time.time() - predictedTime)

        conf_matrix = confusion_matrix(y_test, predicted_y)
        print(conf_matrix)
        print(type(conf_matrix))
        conf_matrix_str = conf_matrix.tobytes()
        # print(type(conf_matrix_str))
        # print(fromstring(conf_matrix_str, dtype=int))
        conf_matrix_str = fromstring(conf_matrix_str, dtype=int)
        # print(converted, type(converted))

        precision = precision_score(y_test, predicted_y)
        recall = recall_score(y_test, predicted_y)
        f1score = f1_score(y_test, predicted_y)

        print(f"Precision = {precision}")
        print(f"Recall = {recall}")
        print(f"F1 Score = {f1score}")

        # define metrics
        y_pred_proba = model.predict_proba(X_test)[::, 1]
        # _ represent the thresh hold
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        adaBoostAUC = metrics.roc_auc_score(y_test, y_pred_proba)

        # create ROC curve
        plt.clf()
        plt.plot(fpr, tpr, label="AUC-" + algorithmName + "=" + str(adaBoostAUC))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        figureName = "figures/" + "ROC-AUC-" + algorithmName + "-testSize-" + str(testSize).zfill(2) + "-round-" + str(
            testRound).zfill(2) + '.png'
        plt.savefig(figureName, bbox_inches='tight')

        resultList.append(
            [algorithmName, testSizeValue, testRound, accuracy, predictedTime, trainTime, precision, recall, f1score,
             conf_matrix_str, tpr, fpr, adaBoostAUC])


        print(" ")
        print(" ------------------------------- ")
        print(" ")

        algorithmName = "catBoost"
        print(algorithmName)

        trainTime = time.time()
        model = None
        auc = None
        fpr = None
        tpr = None
        model = CatBoostRegressor(iterations=2, learning_rate=1, depth=2,)
        model.fit(X_train, y_train)
        trainTime = "%s" % (time.time() - trainTime)

        expected_y = y_test
        predicted_y = model.predict(X_test)

        # accuracy = accuracy_score(y_test, predicted_y)
        accuracy = metrics.accuracy_score(y_test, predicted_y.round())

        print(algorithmName + " Model Accuracy: ", accuracy)

        modelName = "saved-saved-models/" + algorithmName + "-testSize-" + str(testSize).zfill(2) + "-round-" + str(
            testRound).zfill(2) + ".joblib"

        dump(model, modelName)

        predictedTime = time.time()
        predicted_y = model.predict(X_test)
        predictedTime = "%s" % (time.time() - predictedTime)

        conf_matrix = confusion_matrix(y_test, predicted_y.round())
        print(conf_matrix)
        print(type(conf_matrix))
        conf_matrix_str = conf_matrix.tobytes()
        # print(type(conf_matrix_str))
        # print(fromstring(conf_matrix_str, dtype=int))
        conf_matrix_str = fromstring(conf_matrix_str, dtype=int)
        # print(converted, type(converted))

        precision = precision_score(y_test, predicted_y.round())
        recall = recall_score(y_test, predicted_y.round())
        f1score = f1_score(y_test, predicted_y.round())

        print(f"Precision = {precision}")
        print(f"Recall = {recall}")
        print(f"F1 Score = {f1score}")

        # define metrics
        y_pred_proba = model.predict(X_test, prediction_type='Probability')[::, 1]

        # _ represent the thresh hold
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        catBoostAUC = metrics.roc_auc_score(y_test, y_pred_proba)

        # create ROC curve
        plt.clf()
        plt.plot(fpr, tpr, label="AUC-" + algorithmName + "=" + str(catBoostAUC))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        figureName = "figures/" + "ROC-AUC-" + algorithmName + "-testSize-" + str(testSize).zfill(2) + "-round-" + str(
            testRound).zfill(2) + '.png'
        plt.savefig(figureName, bbox_inches='tight')

        resultList.append(
            [algorithmName, testSizeValue, testRound, accuracy, predictedTime, trainTime, precision, recall, f1score,
             conf_matrix_str, tpr, fpr, catBoostAUC])
        
        print(" ")
        print(" ------------------------------- ")
        print(" ")

        algorithmName = "lightGBM"
        print(algorithmName)

        trainTime = time.time()
        model = None
        auc = None
        fpr = None
        tpr = None
        model = lgb.LGBMClassifier(n_estimators=50, learning_rate=1)
        model.fit(X_train, y_train)
        trainTime = "%s" % (time.time() - trainTime)

        expected_y = y_test
        predicted_y = model.predict(X_test)

        accuracy = accuracy_score(y_test, predicted_y)

        print(algorithmName + " Model Accuracy: ", accuracy)

        modelName = "saved-saved-models/" + algorithmName + "-testSize-" + str(testSize).zfill(2) + "-round-" + str(
            testRound).zfill(2) + ".joblib"

        dump(model, modelName)

        predictedTime = time.time()
        predicted_y = model.predict(X_test)
        predictedTime = "%s" % (time.time() - predictedTime)

        conf_matrix = confusion_matrix(y_test, predicted_y)
        print(conf_matrix)
        print(type(conf_matrix))
        conf_matrix_str = conf_matrix.tobytes()
        # print(type(conf_matrix_str))
        # print(fromstring(conf_matrix_str, dtype=int))
        conf_matrix_str = fromstring(conf_matrix_str, dtype=int)
        # print(converted, type(converted))

        precision = precision_score(y_test, predicted_y)
        recall = recall_score(y_test, predicted_y)
        f1score = f1_score(y_test, predicted_y)

        print(f"Precision = {precision}")
        print(f"Recall = {recall}")
        print(f"F1 Score = {f1score}")

        # define metrics
        y_pred_proba = model.predict_proba(X_test)[::, 1]
        # _ represent the thresh hold
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        lightGBMAUC = metrics.roc_auc_score(y_test, y_pred_proba)

        # create ROC curve
        plt.clf()
        plt.plot(fpr, tpr, label="AUC-" + algorithmName + "=" + str(lightGBMAUC))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        figureName = "figures/" + "ROC-AUC-" + algorithmName + "-testSize-" + str(testSize).zfill(2) + "-round-" + str(
            testRound).zfill(2) + '.png'
        plt.savefig(figureName, bbox_inches='tight')

        resultList.append(
            [algorithmName, testSizeValue, testRound, accuracy, predictedTime, trainTime, precision, recall, f1score,
             conf_matrix_str, tpr, fpr, lightGBMAUC])

        print(" ")
        print(" ------------------------------- ")
        print(" ")

        algorithmName = "randomForest"
        print(algorithmName)

        trainTime = time.time()
        model = None
        auc = None
        fpr = None
        tpr = None
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        trainTime = "%s" % (time.time() - trainTime)

        expected_y = y_test
        predicted_y = model.predict(X_test)

        accuracy = accuracy_score(y_test, predicted_y)

        print(algorithmName + " Model Accuracy: ", accuracy)

        modelName = "saved-saved-models/" + algorithmName + "-testSize-" + str(testSize).zfill(2) + "-round-" + str(
            testRound).zfill(2) + ".joblib"

        dump(model, modelName)

        predictedTime = time.time()
        predicted_y = model.predict(X_test)
        predictedTime = "%s" % (time.time() - predictedTime)

        conf_matrix = confusion_matrix(y_test, predicted_y)
        print(conf_matrix)
        print(type(conf_matrix))
        conf_matrix_str = conf_matrix.tobytes()
        # print(type(conf_matrix_str))
        # print(fromstring(conf_matrix_str, dtype=int))
        conf_matrix_str = fromstring(conf_matrix_str, dtype=int)
        # print(converted, type(converted))

        precision = precision_score(y_test, predicted_y)
        recall = recall_score(y_test, predicted_y)
        f1score = f1_score(y_test, predicted_y)

        print(f"Precision = {precision}")
        print(f"Recall = {recall}")
        print(f"F1 Score = {f1score}")

        # define metrics
        y_pred_proba = model.predict_proba(X_test)[::, 1]
        # _ represent the thresh hold
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        randomForestAUC = metrics.roc_auc_score(y_test, y_pred_proba)

        # create ROC curve
        plt.clf()
        plt.plot(fpr, tpr, label="AUC-" + algorithmName + "=" + str(randomForestAUC))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        figureName = "figures/" + "ROC-AUC-" + algorithmName + "-testSize-" + str(testSize).zfill(2) + "-round-" + str(
            testRound).zfill(2) + '.png'
        plt.savefig(figureName, bbox_inches='tight')

        resultList.append(
            [algorithmName, testSizeValue, testRound, accuracy, predictedTime, trainTime, precision, recall, f1score,
             conf_matrix_str, tpr, fpr, randomForestAUC])

        figureName = "figures/" + "ROC-AUC-ALL-" + "testSize-" + str(testSize).zfill(2) + "-round-" + str(
            testRound).zfill(2) + '.png'
        plt.savefig(figureName, bbox_inches='tight')

        print("%s" % (time.time() - start_time))
        print("-------------------> End of Round")

        fileName = "comparison/dataset-01-comparison-" + str(testSize).zfill(2) + "-" + str(testRound).zfill(2) + ".csv"

        csvDataFrame = pd.DataFrame(resultList, columns=csvColumn)
        csvDataFrame.to_csv(fileName)

# Script execution Time
print("%s" % (time.time() - scriptTime))
