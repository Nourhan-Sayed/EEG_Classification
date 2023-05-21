from support import get_subjects_data
# settings
fs = 256 # Sampling rate
condition = "INNER" # PRONOUNCED, INNER or VISUALIZED
random_state = 46

# Select the useful par of each trial. Time in seconds
t_start = 1.5 # start (in seconds)
t_end = 3.5 # end (in seconds)


data_array, label_array, group_array = get_subjects_data(condition=condition, t_start = t_start, t_end = t_end, fs = fs)
data_array.shape, label_array.shape, group_array.shape

from features import f_mean, f_std, f_ptp, f_var, f_minim, f_maxim, f_argminim, f_argmaxim, f_rms, f_abs_diff_signal, \
    f_skewness, f_kurtosis, generate_features

func_list = [f_mean, f_std, f_ptp, f_var, f_minim, f_maxim, f_argminim, f_argmaxim, f_rms, f_abs_diff_signal, f_skewness, f_kurtosis]

features_array = generate_features(data_array, func_list)
features_array.shape

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def split_train_test(data, labels, groups, size):
    # Stratify guarantees that the same proportion of the classes will be available in train and test
    x_tr, x_ts, y_tr, y_ts, g_tr, g_ts = train_test_split(data, labels, groups, test_size=size, stratify=y, random_state=random_state)
    # Apply the scaler in the training data
    ss = StandardScaler()
    x_tr = ss.fit_transform(x_tr)
    x_ts = ss.transform(x_ts)
    return x_tr, x_ts, y_tr, y_ts, g_tr, g_ts

# Run nested cross-validation and re-run using the best parameters
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from support import run_cross_validation, get_feature_selection_model, print_report_nested_cross_validation, print_report_classifier

X = features_array
y = label_array

feature_sm = get_feature_selection_model(X, y)

# Apply the Feature Selection Model without scaling the data
X = feature_sm.transform(X)
n_features_before = np.shape(features_array)
print("Feature transformation - number of features: Before {} - After {}".format(np.shape(features_array)[1], np.shape(X)[1]))

splits = [0.10, 0.20, 0.30]

# Run Nested cross-validation
inner_cv = StratifiedGroupKFold(n_splits=5)
outer_cv = StratifiedGroupKFold(n_splits=5)

classifiers = [
    ["Random Forest", RandomForestClassifier(), {'n_estimators': [200, 500, 1000, 2000], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth' : [4,5,6,7,8], 'criterion' :['gini', 'entropy']}],
    ["Linear SVC", LinearSVC(), {'C': [0.00001, 0.0001, 0.0005, 1, 10, 100, 1000], 'dual': (True, False)}],
    ["SVC", SVC(), [{"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [0.00001, 0.0001, 0.0005, 1, 10, 100, 1000]},
                    {"kernel": ["linear"], "C": [0.00001, 0.0001, 0.0005, 1, 10, 100, 1000]}, ]
     ]
]

for cls in classifiers:
    best_params = []
    best_scores = []

    for test_size in splits:
        x_train, x_test, y_train, y_test, g_train, g_test = split_train_test(X, y, group_array, test_size)
        clf = GridSearchCV(estimator=cls[1], param_grid=cls[2], cv=inner_cv, n_jobs=-1)
        clf.fit(x_train, y_train, groups=g_train)

        best_params.append(clf.best_params_)
        best_scores.append(clf.best_score_)

    # Get the best parameter
    best_param = best_params[np.argmax(best_scores)]

    acc_list = []
    cross_v_list = []
    # Run the same classifier using the best parameters
    for test_size in splits:
        x_train, x_test, y_train, y_test, g_train, g_test = split_train_test(X, y, group_array, test_size)
        best_param['random_state'] = random_state
        cls[1].set_params(**best_param)
        cls[1].fit(x_train, y_train)
        y_pred = cls[1].predict(x_test)
        acc_list.append(metrics.accuracy_score(y_test, y_pred))
        cross_v_list.append(run_cross_validation(cls[1], outer_cv, x_train, y_train, g_train))

    print('\n{}: {} '.format("Classifier", cls[0]))
    print_report_nested_cross_validation(splits, best_params, best_scores)
    print_report_classifier(splits, acc_list, cross_v_list)
    print(f_std(best_scores))