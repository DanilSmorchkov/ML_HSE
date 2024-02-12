import numpy as np
import copy
from typing import List, NoReturn
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import tqdm


# Task 1

class Module:
    """
    Абстрактный класс. Его менять не нужно. Он описывает общий интерфейс
    взаимодействия со слоями нейронной сети.
    """
    def forward(self, x):
        pass

    def backward(self, d):
        pass

    def update(self, alpha):
        pass


class Linear(Module):
    """
    Линейный полносвязный слой.
    """
    def __init__(self, in_features: int, out_features: int):
        """
        Parameters
        ----------
        in_features : int
            Размер входа.
        out_features : int
            Размер выхода.

        Notes
        -----
        W и b инициализируются случайно.
        """
        self.X = None
        self.grad = None
        self.weights = np.random.uniform(-0.3, 0.3, size=(in_features+1, out_features))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = Wx + b.

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.
            То есть, либо x вектор с in_features элементов,
            либо матрица размерности (batch_size, in_features).

        Return
        ------
        y : np.ndarray
            Выход после слоя.
            Либо вектор с out_features элементами,
            либо матрица размерности (batch_size, out_features)

        """
        if len(x.shape) == 1:
            x = x.reshape((1, x.shape[0]))
        self.X = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        return self.X @ self.weights

    def backward(self, d: np.ndarray) -> np.ndarray:
        """
        Считает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        if len(d.shape) == 1:
            d = d.reshape((1, d.shape[0]))
        self.grad = np.copy(d)
        return (self.grad @ self.weights.T)[:, 1:]

    def update(self, alpha: float) -> NoReturn:
        """
        Обновляет W и b с заданной скоростью обучения.

        Parameters
        ----------
        alpha : float
            Скорость обучения.
        """
        self.weights -= alpha * self.X.T @ self.grad / self.X.shape[0]


class ReLU(Module):
    """
    Слой, соответствующий функции активации ReLU. Данная функция возвращает
    новый массив, в котором значения меньшие 0 заменены на 0.
    """
    def __init__(self):
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = max(0, x).

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.

        Return
        ------
        y : np.ndarray
            Выход после слоя (той же размерности, что и вход).

        """
        x = np.copy(x)
        self.mask = x > 0
        x[~self.mask] = 0
        return x

    def backward(self, d) -> np.ndarray:
        """
        Считает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        return self.mask * d


class SoftMax(Module):
    """
    Реализует Softmax на выходе сетки - вероятности классов.
    """

    def __init__(self):
        self.softmax = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = softmax(x) - вероятности соответствующих классов

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.

        Return
        ------
        y : np.ndarray
            Выход после слоя (той же размерности, что и вход).

        """
        self.softmax = np.exp(x) / np.sum(np.exp(x), axis=1).reshape((x.shape[0], 1))
        return self.softmax

    def backward(self, d) -> np.ndarray:
        """
        Считает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Матрица правильных меток классов для каждого объекта батча
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        return self.softmax - d


# Task 2

class MLPClassifier:
    def __init__(self, modules: List[Module], epochs: int = 40,
                 alpha: float = 0.01, batch_size: int = 32):
        """
        Parameters
        ----------
        modules : List[Module]
            Cписок, состоящий из ранее реализованных модулей и
            описывающий слои нейронной сети.
            В конец необходимо добавить Softmax.
        epochs : int
            Количество эпох обучения.
        alpha : float
            Cкорость обучения.
        batch_size : int
            Размер батча, используемый в процессе обучения.
        """
        self.modules = modules + [SoftMax()]
        self.epochs = epochs
        self.alpha = alpha
        self.batch_size = batch_size

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает нейронную сеть заданное число эпох.
        В каждой эпохе необходимо использовать cross-entropy loss для обучения,
        а так же производить обновления не по одному элементу, а используя
        батчи (иначе обучение будет нестабильным и полученные результаты
        будут плохими).

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения.
        y : np.ndarray
            Вектор меток классов для данных.
        """
        y = np.array(y).reshape(X.shape[0], 1)
        for _ in range(self.epochs):
            X_y = np.concatenate((X, y), axis=1)
            np.random.shuffle(X_y)
            splitted_shuffled_X_y = np.split(X_y, np.arange(0, X.shape[0], self.batch_size), axis=0)[1:]
            for batch in splitted_shuffled_X_y:
                X_batch, y_batch = batch[:, :-1], batch[:, -1]
                y_matrix_one_hot = np.zeros((X_batch.shape[0], len(np.unique(y))))
                for i in range(y_matrix_one_hot.shape[0]):
                    y_matrix_one_hot[i, int(y_batch[i])] = 1
                # forward
                output = np.copy(X_batch)
                for layer in self.modules:
                    output = layer.forward(output)
                # loss-function
                loss = -np.sum(y_matrix_one_hot * np.log(output))
                # backward
                output = np.copy(y_matrix_one_hot)
                for layer in reversed(self.modules):
                    output = layer.backward(output)
                    if isinstance(layer, Linear):
                        layer.update(self.alpha)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает вероятности классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.

        Return
        ------
        np.ndarray
            Предсказанные вероятности классов для всех элементов X.
            Размерность (X.shape[0], n_classes)

        """
        output = np.copy(X)
        for layer in self.modules:
            output = layer.forward(output)
        return output

    def predict(self, X) -> np.ndarray:
        """
        Предсказывает метки классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.

        Return
        ------
        np.ndarray
            Вектор предсказанных классов

        """
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)

# Task 3


classifier_moons = MLPClassifier([
    Linear(2, 40),
    ReLU(),
    Linear(40, 10),
    ReLU(),
    Linear(10, 2),
], batch_size=8, epochs=60)  # Нужно указать гиперпараметры


classifier_blobs = MLPClassifier([
    Linear(2, 40),
    ReLU(),
    Linear(40, 15),
    ReLU(),
    Linear(15, 3),
], batch_size=8, epochs=60)  # Нужно указать гиперпараметры

# Task 4


class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=36, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=36, out_channels=36, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=36, out_channels=84, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=84, out_channels=84, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=84, out_channels=196, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=196, out_channels=324, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=324, out_channels=324, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(324 * 2 * 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv1_1(x)
        x = self.pool(self.conv1_2(x))
        x = self.dropout(x)

        x = self.conv2_1(x)
        x = self.pool(self.conv2_2(x))
        x = self.dropout(x)

        x = self.conv3_1(x)
        x = self.pool(self.conv3_2(x))
        x = self.dropout(x)

        x = self.conv4_1(x)
        x = self.pool(self.conv4_2(x))
        x = self.dropout(x)

        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        self.fc3(x)

        return x

    def load_model(self):
        """
        Используйте torch.load, чтобы загрузить обученную модель
        Учтите, что файлы решения находятся не в корне директории, поэтому
        необходимо использовать следующий путь:
        `__file__[:-7] +"model.pth"`, где "model.pth" - имя файла сохраненной
        модели `
        """
        self.load_state_dict(torch.load(__file__[:-7] + 'model.pth'))

    def save_model(self):
        """
        Используйте torch.save, чтобы сохранить обученную модель
        """
        torch.save(self.state_dict(), 'model.pth')


def calculate_loss(X: torch.Tensor, y: torch.Tensor, model: TorchModel):
    """
    Cчитает cross-entropy.

    Parameters
    ----------
    X : torch.Tensor
        Данные для обучения.
    y : torch.Tensor
        Метки классов.
    model : Model
        Модель, которую будем обучать.

    """
    output = model(X)
    loss_cross = nn.CrossEntropyLoss()
    loss = loss_cross(output, y)
    return loss
