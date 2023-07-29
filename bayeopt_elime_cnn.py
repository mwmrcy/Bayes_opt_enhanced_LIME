import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPClassifier # neural network
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from bayes_opt import BayesianOptimization, UtilityFunction
import warnings
warnings.filterwarnings("ignore")

from explainer_tabular import LimeTabularExplainer,Sklearn_Lime
from load_Gaitrecdataset2 import LoadDataset

test = LoadDataset()
X = test.data.data
Y = test.data.target

feature_names = test.data.feature_names
target_names = test.data.target_names

#print(target_names)

x_train, x_test, y_train, y_test = train_test_split(X, Y ,test_size = 0.2, shuffle=False)

nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=0)
nn.fit(x_train, y_train)

#mean_accuracy = nn.score(x_test, y_test)
y_pred = nn.predict(x_test)

print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred.round())))
#print (mean_accuracy)

fig = plot_confusion_matrix(nn, x_test, y_test, display_labels=nn.classes_)
fig.figure_.suptitle("Confusion Matrix for Gaitrec Dataset")
plt.show()

print(classification_report(y_test, y_pred))

i = np.random.randint(0, x_test.shape[0])

# GMM clustering
model = GaussianMixture(n_components=4).fit(X)
labels = model.predict(X)
names = list(feature_names)+["membership"]
clustered_data = np.column_stack([X,labels])

#print(clustered_data)

# KNN classification
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(x_train)
distances, indices = nbrs.kneighbors(x_test)
clabel = labels

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

def jaccard_distance(usecase):
    sim = []
    for l in usecase:
        i_sim = []
        for j in usecase:
            i_sim.append(1 - jaccard_similarity(l, j))
        sim.append(i_sim)
    return sim

x = 1200
maxRsquared=0.9

clustered_data2 = np.delete(clustered_data, 104, axis=1)
def OptiLIME_loss(kernel_width):
    single_lime = Sklearn_Lime(mode="classification",
                               verbose=False,
                               discretize_continuous=True,
                               kernel_width=kernel_width,
                               maxRsquared=maxRsquared,
                               epsilon=None,
                               clustered_data =clustered_data2)
                                                           
    single_lime.fit(clustered_data2)
 
    return single_lime.score(clustered_data2[x,:], nn.predict_proba)
pbounds={'kernel_width':[0.1,10.0]}
optimizer=BayesianOptimization(f=OptiLIME_loss, pbounds=pbounds,verbose=2,random_state=4)
optimizer.maximize(init_points=5,n_iter=10)
best_width =optimizer.max["params"]["kernel_width"]
print(best_width)


lime_exp=[]
elime_exp = []

explainer = LimeTabularExplainer(x_train,
                                 mode="classification",
                                 verbose=False,
                                 discretize_continuous=True,
                                 feature_names=feature_names,
                                 class_names=target_names,
                                 kernel_width=best_width,
                                 )

for i in range(0, 10):
        p_label = clabel[indices[x]]
        N = clustered_data[clustered_data[:,104] == clabel[p_label]]
        subset = np.delete(N, 104, axis=1)
                  
        exp_elime = explainer.explain_instance_gmmclust(x_test[x],
                                             nn.predict_proba,
                                             num_features=10,
                                             model_regressor=LinearRegression(),
                                             clustered_data = subset,
                                             regressor = 'linear', explainer='elime', labels=(0,1))
        exp_elime.show_in_notebook(show_table=True)
       
        fig_elime, r_features = exp_elime.as_pyplot_to_figure(type='h', name = i+.6, label='0')
        elime_exp.append(r_features)
        

        exp_lime = explainer.explain_instance_gmmclust(x_test[x],
                                             nn.predict_proba,
                                             num_features=10,
                                             model_regressor=LinearRegression(),
                                             regressor = 'linear', explainer='lime', labels=(0,1))
        exp_lime.show_in_notebook(show_table=True)

        fig_elime, r_features = exp_lime.as_pyplot_to_figure(type='h', name = i+.5, label='0')
        lime_exp.append(r_features)
        
     
################################################
sim = jaccard_distance(elime_exp)
np.savetxt("results/rf_dlime_jdist_ildp.csv", sim, delimiter=",")
print(np.asarray(sim).mean())
plt.matshow(sim);
plt.colorbar()


