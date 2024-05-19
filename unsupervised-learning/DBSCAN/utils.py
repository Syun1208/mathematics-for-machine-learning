import matplotlib.pyplot as plt
import numpy as np

def calculate_covariance_matrix(X, Y=None):
    """ Calculate the covariance matrix for the dataset X """
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples-1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))

    return np.array(covariance_matrix, dtype=float)

class Plot:
    def __init__(self): 
        self.cmap = plt.get_cmap('viridis')

    def _transform(self, X, dim):
        covariance = calculate_covariance_matrix(X)
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        # Sort eigenvalues and eigenvector by largest eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:dim]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :dim]
        # Project the data onto principal components
        X_transformed = X.dot(eigenvectors)

        return X_transformed


    def plot_regression(self, lines, title, axis_labels=None, mse=None, scatter=None, legend={"type": "lines", "loc": "lower right"}):
        
        if scatter:
            scatter_plots = scatter_labels = []
            for s in scatter:
                scatter_plots += [plt.scatter(s["x"], s["y"], color=s["color"], s=s["size"])]
                scatter_labels += [s["label"]]
            scatter_plots = tuple(scatter_plots)
            scatter_labels = tuple(scatter_labels)

        for l in lines:
            li = plt.plot(l["x"], l["y"], color=s["color"], linewidth=l["width"], label=l["label"])

        if mse:
            plt.suptitle(title)
            plt.title("MSE: %.2f" % mse, fontsize=10)
        else:
            plt.title(title)

        if axis_labels:
            plt.xlabel(axis_labels["x"])
            plt.ylabel(axis_labels["y"])

        if legend["type"] == "lines":
            plt.legend(loc="lower_left")
        elif legend["type"] == "scatter" and scatter:
            plt.legend(scatter_plots, scatter_labels, loc=legend["loc"])

        plt.show()



    # Plot the dataset X and the corresponding labels y in 2D using PCA.
    def plot_in_2d(self, X, y=None, title=None, accuracy=None, legend_labels=None):
        X_transformed = self._transform(X, dim=2)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]
        class_distr = []

        y = np.array(y).astype(int)

        colors = [self.cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

        # Plot the different class distributions
        for i, l in enumerate(np.unique(y)):
            _x1 = x1[y == l]
            _x2 = x2[y == l]
            _y = y[y == l]
            class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

        # Plot legend
        if not legend_labels is None: 
            plt.legend(class_distr, legend_labels, loc=1)

        # Plot title
        if title:
            if accuracy:
                perc = 100 * accuracy
                plt.suptitle(title)
                plt.title("Accuracy: %.1f%%" % perc, fontsize=10)
            else:
                plt.title(title)

        # Axis labels
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        plt.show()

    # Plot the dataset X and the corresponding labels y in 3D using PCA.
    def plot_in_3d(self, X, y=None):
        X_transformed = self._transform(X, dim=3)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]
        x3 = X_transformed[:, 2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x1, x2, x3, c=y)
        plt.show()
        
        
class DBSCAN:
    
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_pts = min_samples

        
    def predict(self, data):
        self.data = data
        self.labels = np.zeros(len(self.data), dtype=np.int64)
        for i in range(len(self.labels)):
            self.labels[i] = -2  # unprocessed

        c_id = -1  # offset start
        for i in range(0, len(self.data)):
            if self.labels[i] != -2:  # unprocessed
                continue

            neighbors = self.region_query(i)
            if len(neighbors) < self.min_pts:
                self.labels[i] = -1  # noise
            else: 
                c_id += 1
                self.expand(i, neighbors, c_id)

        return self.labels


    def expand(self, p, neighbors, c_id):
        self.labels[p] = c_id
        i = 0
        while i < len(neighbors): 
            pn = neighbors[i]
            
            if self.labels[pn] == -1:
                self.labels[pn] = c_id
            elif self.labels[pn] == -2:
                self.labels[pn] = c_id
                
            new_neighbors = self.region_query(pn)
            
            if len(new_neighbors) >= self.min_pts:
                neighbors = neighbors + new_neighbors
            i += 1   

    @staticmethod
    def euclidean_distance(x1, x2):
        """ Calculates the l2 distance between two vectors """
        distance = 0
        # Squared distance between each coordinate
        for i in range(len(x1)):
            distance += pow((x1[i] - x2[i]), 2)
        return np.sqrt(distance)


    def region_query(self, p):
        # return indices of data items that are close to data[p]
        result = []
        for i in range(len(self.data)):
            if self.euclidean_distance(self.data[p], self.data[i]) < self.eps:
                result.append(i)
        return result 
