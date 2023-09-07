import numpy as np
import copy
from tqdm import tqdm
import torch
class KMeans:
    
    def __init__(self,n_clusters=10,max_iter=500):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.loss_per_iteration = []

    def init_centroids(self):
        np.random.seed(np.random.randint(0,100000))
        self.centroids = []
        self.diameter = []
        for i in range(self.n_clusters):
            rand_index = np.random.choice(range(len(self.fit_data)))
            self.centroids.append(self.fit_data[rand_index])
            self.diameter.append(0.0)
    
    def init_clusters(self):
        self.clusters = {'data':{i:[] for i in range(self.n_clusters)}}
        self.clusters['labels']={i:[] for i in range(self.n_clusters)}
        self.clusters['diameter_list'] = {i: [] for i in range(self.n_clusters)}

    def fit(self,fit_data,fit_labels):
        self.fit_data = fit_data
        self.fit_labels = fit_labels
        self.predicted_labels = [None for _ in range(self.fit_data.shape[0])]
        self.init_centroids()
        self.iterations = 0
        #old_centroids = [np.zeros(shape=(fit_data.shape[1],)) for _ in range(self.n_clusters)]
        old_predicted_labels = [None for _ in range(self.fit_data.shape[0])]
        while not self.converged(self.iterations,old_predicted_labels,self.predicted_labels):
            old_centroids = copy.deepcopy(self.centroids)
            old_predicted_labels = copy.deepcopy(self.predicted_labels)
            self.init_clusters()
            for j,sample in tqdm(enumerate(self.fit_data)):
                min_dist = float('inf')
                for i,centroid in enumerate(self.centroids):
                    dist = np.linalg.norm(sample-centroid, ord=np.inf)
                    if dist<min_dist:
                        min_dist = dist
                        self.predicted_labels[j] = i
                if self.predicted_labels[j] is not None:
                        self.clusters['data'][self.predicted_labels[j]].append(sample)                    
                        self.clusters['labels'][self.predicted_labels[j]].append(self.fit_labels[j])
                        self.clusters['diameter_list'][self.predicted_labels[j]].append(min_dist)
            self.reshape_cluster()
            self.update_centroids()
            self.update_diameter()
            self.calculate_loss()
            print(f"diameter {self.diameter}")
            print("\nIteration:",self.iterations,'Loss:',self.loss)
            self.iterations+=1
        self.calculate_accuracy()

    def update_centroids(self):
        for i in range(self.n_clusters):
            cluster = self.clusters['data'][i]
            if cluster.shape[0] == 0:
                self.centroids[i] = self.fit_data[np.random.choice(range(len(self.fit_data)))]
                self.diameter[i] = 0.0
            elif cluster.shape[0] == 1:
                self.centroids[i] = cluster[0]
                self.diameter[i] = 0.0
            else:
                self.centroids[i] = np.mean(np.vstack((self.centroids[i],cluster)),axis=0)
                # distance = torch.nn.functional.pdist(torch.Tensor(cluster), torch.inf)
                # arg_max = torch.argmax(distance).item()
                # first_point = 0
                # second_point = 0
                # count_ = 0
                # num_add = cluster.shape[0]-1
                # while count_ < arg_max:
                #     if count_ + num_add < arg_max:
                #         count_ += num_add
                #         num_add -= 1
                #         first_point += 1
                #     else:
                #         second_point = first_point + (arg_max - count_)
                #         count_ += num_add
                # self.centroids[i] = np.mean([cluster[first_point], cluster[second_point]])
                # self.diameter[i] = torch.max(distance) / 2

    def reshape_cluster(self):
        for id,mat in list(self.clusters['data'].items()):
            self.clusters['data'][id] = np.array(mat)

    def update_diameter(self):
        for i in range(self.n_clusters):
            cluster = self.clusters['diameter_list'][i]
            if cluster == []:
                self.diameter[i] = 0.0
            else:
                self.diameter[i] = np.max(np.array(cluster))

    def converged(self,iterations,labels,updated_labels):
        if labels[0] is None or updated_labels[0] is None:
            return False
        
        sum_differences = np.sum(np.abs((np.array(labels) - np.array(updated_labels))))
        if sum_differences == 0:
            return True
        return False
        
        # if iterations > self.max_iter:
        #     return True
        # self.centroids_dist = np.linalg.norm(np.array(updated_centroids)-np.array(centroids), ord=np.inf)
        # if self.centroids_dist<=1e-10:
        #     print("Converged! With distance:",self.centroids_dist)
        #     return True
        # return False

    def calculate_loss(self):
        self.loss = 0
        for key,value in list(self.clusters['data'].items()):
            if value is not None:
                for v in value:
                    self.loss += np.linalg.norm(v-self.centroids[key], ord=np.inf)
        self.loss_per_iteration.append(self.loss)
    
    def calculate_accuracy(self):
        self.clusters_labels = []
        self.clusters_info = []
        self.clusters_accuracy = []
        for clust,labels in list(self.clusters['labels'].items()):
            if isinstance(labels[0],(np.ndarray)):
                labels = [l[0] for l in labels]
            occur = 0
            max_label = max(set(labels), key=labels.count)
            self.clusters_labels.append(max_label)
            for label in labels:
                if label == max_label:
                    occur+=1
            acc = occur/len(list(labels))
            self.clusters_info.append([max_label,occur,len(list(labels)),acc])
            self.clusters_accuracy.append(acc)
            self.accuracy = sum(self.clusters_accuracy)/self.n_clusters
        self.labels_ = []
        for i in range(len(self.predicted_labels)):
            self.labels_.append(self.clusters_labels[self.predicted_labels[i]])
        print('[cluster_label,no_occurence_of_label,total_samples_in_cluster,cluster_accuracy]',self.clusters_info)
        print('Accuracy:',self.accuracy)
