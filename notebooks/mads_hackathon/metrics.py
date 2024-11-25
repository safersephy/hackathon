import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class Accuracy:
    def __repr__(self) -> str:
        return "Accuracy"

    def __call__(self, y, yhat):
        return (np.argmax(yhat, axis=1) == y).sum() / len(yhat)


class F1Score:
    def __init__(self, average: str):
        self.average = average

    def __repr__(self) -> str:
        return f"F1score{(self.average)}"

    def __call__(self, y, yhat):
        return f1_score(
            y, np.argmax(yhat, axis=1), average=self.average, zero_division=np.nan
        )


class Recall:
    def __init__(self, average: str):
        self.average = average

    def __repr__(self) -> str:
        return f"Recall{(self.average)}"

    def __call__(self, y, yhat):
        return recall_score(
            y, np.argmax(yhat, axis=1), average=self.average, zero_division=np.nan
        )


class Precision:
    def __init__(self, average: str):
        self.average = average

    def __repr__(self) -> str:
        return f"Precision{(self.average)}"

    def __call__(self, y, yhat):
        return precision_score(
            y, np.argmax(yhat, axis=1), average=self.average, zero_division=np.nan
        )


def caluclate_cfm(model, teststreamer):
    y_true = []
    y_pred = []

    testdata = teststreamer.stream()
    for _ in range(len(teststreamer)):
        X, y = next(testdata)
        yhat = model(X)
        yhat = yhat.argmax(dim=1)  # we get the one with the highest probability
        y_pred.append(yhat.cpu().tolist())
        y_true.append(y.cpu().tolist())

    yhat = [x for y in y_pred for x in y]
    y = [x for y in y_true for x in y]

    cfm = confusion_matrix(y, yhat)
    cfm = cfm / np.sum(cfm, axis=1, keepdims=True)
    return cfm
