from numpy.core.numeric import cross
from sklearn.ensemble import RandomForestClassifier 
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from hyperopt.pyll import scope
import seaborn as sns
from sklearn.model_selection import train_test_split
import time 
 #fmin: find, tpe: algo, hp: specify domain 
# param_hyperopt = {
#     'max_depth': scope.int(hp.quniform('AAAA', 5, 15, 1)),
#     'n_estimators': scope.int(hp.quniform('BBBB',40, 100, 2)),
#     'bootstrap': hp.choice('CCCC', [True, False])
#     # 'bootstrap': hp.choice('bootstrap', [True, False])
# }

# param2 = {
#     'n_neighbors': scope.int(hp.quniform('n_neighbors', 5, 30, 1)),
#     # 'metric': hp.choice('metric', ['eucliden', 'minkowski']),
#     'weights': hp.choice('weight', ['uniform', 'distance'])

# }


param3 = hp.choice('clf', [
    {
        'type': 'random_forest',
        'max_depth': scope.int(hp.quniform('max_depth',5,15,1)),
        'n_estimators': scope.int(hp.quniform('n_estimators',40,100,2))
    },

    {
        'type': 'n_neighbors',
        'n_neighbors': scope.int(hp.quniform('n_neighbors', 5,30,1)),
        'weights': hp.choice('weight', ['uniform', 'distance'])
    }
    ] 
)


def hyperopt(param, X_train, y_train, X_test, y_test, num_eval):
    start = time.time()

    def objective_function(params): 
        clf = KNeighborsClassifier(**params)
        score = cross_val_score(clf, X_train, y_train).mean()
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(objective_function, param, algo=tpe.suggest, max_evals=num_eval, trials=trials)
    print(best)

def hyperopt2(param, X_train, y_train, X_test, y_test, num_eval):
    start = time.time()

    def objective_function(params):
        classifier = params['type']
        del params['type']
        if classifier == 'random_forest':
            clf = RandomForestClassifier(**params)
            score = cross_val_score(clf, X_train, y_train).mean()
        else:
            clf = KNeighborsClassifier(**params)
            score = cross_val_score(clf, X_train, y_train).mean()
        return {'loss': -score, 'status': STATUS_OK}
    trials = Trials()
    best = fmin(objective_function, param, algo=tpe.suggest, max_evals=num_eval, trials=trials)
    print(best)
data = sns.load_dataset('iris')
dict = {'versicolor': 0, 'setosa': 1, 'virginica':2}
data['species'] = data['species'].map(dict)
X_train, X_test, y_train, y_test = train_test_split(data.drop(['species'], axis=1).values, data['species'].values, test_size=0.3, shuffle=True)
(hyperopt2(param3, X_train, y_train, X_test, y_test, num_eval=100))
