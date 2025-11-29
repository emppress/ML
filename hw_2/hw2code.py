import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    sort_order: np.ndarray = np.argsort(feature_vector)
    sorted_features: np.ndarray = feature_vector[sort_order]
    sorted_targets: np.ndarray = target_vector[sort_order]
    n_samples: int = len(feature_vector)

    change_points: np.ndarray = np.where(
        sorted_features[:-1] != sorted_features[1:])[0]

    if change_points.size == 0:
        return np.array([]), np.array([]), None, None

    thresholds: np.ndarray = (
        sorted_features[change_points] + sorted_features[change_points + 1]
    ) / 2

    cum_class_0: np.ndarray = np.cumsum(sorted_targets == 0)
    cum_class_1: np.ndarray = np.cumsum(sorted_targets == 1)
    total_class_0: int = int(cum_class_0[-1])
    total_class_1: int = int(cum_class_1[-1])

    left_size: np.ndarray = change_points + 1
    right_size: np.ndarray = n_samples - left_size

    left_class_0: np.ndarray = cum_class_0[change_points]
    left_class_1: np.ndarray = cum_class_1[change_points]
    right_class_0: np.ndarray = total_class_0 - left_class_0
    right_class_1: np.ndarray = total_class_1 - left_class_1

    left_class_0 = left_class_0 / left_size
    left_class_1 = left_class_1 / left_size
    right_class_0 = right_class_0 / right_size
    right_class_1 = right_class_1 / right_size

    gini_left: np.ndarray = 1 - left_class_0**2 - left_class_1**2
    gini_right: np.ndarray = 1 - right_class_0**2 - right_class_1**2

    gini_scores: np.ndarray = (
        -(left_size / n_samples) * gini_left -
        (right_size / n_samples) * gini_right
    )
    best_index = np.argmax(gini_scores)

    return thresholds, gini_scores, thresholds[best_index], gini_scores[best_index]


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, cur_depth):
        if np.all(sub_y == sub_y[0]):  # критерий останова - все 1-го класса
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._max_depth is not None and cur_depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    current_click = clicks.get(key, 0)
                    # отношение 1 ко всем
                    ratio[key] = current_click / current_count
                # тут мы берём названия, а не числа
                sorted_categories = list(
                    map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(
                    zip(sorted_categories, list(range(len(sorted_categories)))))
                feature_vector = np.array(list(
                    map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if threshold is None:
                continue
            if gini_best is None or (gini is not None and gini > gini_best):
                left = feature_vector < threshold
                right = ~left
                if self._min_samples_leaf is not None and (
                    left.sum() < self._min_samples_leaf
                    or right.sum() < self._min_samples_leaf
                ):
                    continue

                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold,  # type: ignore
                                                     categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            # без [0][0] это [(class, count)]
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split],
                       node["left_child"], cur_depth+1)
        self._fit_node(sub_X[np.logical_not(split)],  # type: ignore
                       sub_y[np.logical_not(split)],  # type: ignore
                       node["right_child"],
                       cur_depth+1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_idx = node["feature_split"]
        feature_val = x[feature_idx]

        if self._feature_types[feature_idx] == "real":
            if feature_val < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif self._feature_types[feature_idx] == "categorical":
            if feature_val in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError

    def fit(self, X, y):
        self._tree = {}
        self._fit_node(X, y, self._tree, 0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_params(self, deep=True):
        return {
            "feature_types": self._feature_types,
            "max_depth": self._max_depth,
            "min_samples_split": self._min_samples_split,
            "min_samples_leaf": self._min_samples_leaf,
        }

    def set_params(self, **params):
        if "feature_types" in params:
            self._feature_types = params["feature_types"]
        if "max_depth" in params:
            self._max_depth = params["max_depth"]
        if "min_samples_split" in params:
            self._min_samples_split = params["min_samples_split"]
        if "min_samples_leaf" in params:
            self._min_samples_leaf = params["min_samples_leaf"]
        return self
