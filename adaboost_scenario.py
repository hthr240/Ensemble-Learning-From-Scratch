import numpy as np
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

from adaboost import AdaBoost
from decision_stump import DecisionStump

def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y

# helpers #
def train_adaboost(train_X, train_y,test_X, test_y, n_learners):

    # Train AdaBoost model
    adaboost = AdaBoost(lambda: DecisionStump(), n_learners)
    adaboost._fit(train_X, train_y)
    print(f"Number of models: {len(adaboost.models_)}")

    # Compute training and testing errors
    train_errors = []
    test_errors = []

    for t in range(1, len(adaboost.models_) + 1):
        # Calculate train error after adding t weak learners
        train_preds = adaboost.partial_predict(train_X, t)
        train_error = np.mean(train_preds != train_y)  # Misclassification error
        train_errors.append(train_error)

        # Calculate test error after adding t weak learners
        test_preds = adaboost.partial_predict(test_X, t)
        test_error = np.mean(test_preds != test_y)  # Misclassification error
        test_errors.append(test_error)

    return train_errors, test_errors

def plot_errors(train_errors, test_errors, noise, n_learners=250):

    # n_learners = len(train_errors)  
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, n_learners + 1), train_errors, label='Train Error', color='blue')
    plt.plot(np.arange(1, n_learners + 1), test_errors, label='Test Error', color='orange')
    plt.title('Train and Test Errors vs Number of Learners (Noise={})'.format(noise))
    plt.xlabel('Number of Learners')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_decision_boundaries(model, test_X, test_y, T_vals, lims, noise):

    fig = make_subplots(rows=1, cols=len(T_vals),
                        subplot_titles=[f"T = {T}" for T in T_vals],
                        horizontal_spacing=0.05)

    for i, T in enumerate(T_vals):
        predict_func = lambda X: model.partial_predict(X, T)

        # Decision surface
        surface = decision_surface(predict_func, lims[0], lims[1], showscale=False)

        # Test data scatter
        scatter = go.Scatter(
            x=test_X[:, 0],
            y=test_X[:, 1],
            mode='markers',
            marker=dict(
                color=test_y,
                colorscale=[custom[0][1], custom[-1][1]],  # blue for -1, red for 1
                cmin=-1, cmax=1,
                line=dict(color='black', width=0.5)
            ),
            showlegend=False
        )

        fig.add_trace(surface, row=1, col=i+1)
        fig.add_trace(scatter, row=1, col=i+1)

    fig.update_layout(
        title="Decision Boundaries of AdaBoost on Test Set",
        width=300 * len(T_vals),
        height=400,
        margin=dict(t=40)
    )

    fig.write_html(f"decision_boundary_noise={noise}.html")
    # fig.show()

def plot_best_decision_surface(model, test_X, test_y, test_errors, lims, noise):
    best_t = np.argmin(test_errors) + 1  # +1 because t starts from 1
    accuracy = 1 - test_errors[best_t - 1]

    predict_func = lambda X: model.partial_predict(X, best_t)
    surface = decision_surface(predict_func, lims[0], lims[1], showscale=False)

    scatter = go.Scatter(
        x=test_X[:, 0], y=test_X[:, 1],
        mode='markers',
        marker=dict(color=test_y, colorscale='RdBu', line=dict(width=0.5, color='black')),
        showlegend=False
    )

    fig = go.Figure(data=[surface, scatter])
    fig.update_layout(
        title=f"Best Ensemble at T={best_t} (Accuracy={accuracy:.3f})",
        width=600,
        height=500
    )
    fig.write_html(f"best_decesion_surface_noise={noise}.html")
    # fig.show()

def plot_weighted_decision_surface(model, train_X, train_y, lims, noise):

    print("Sample weights stats: min", np.min(model.D_), "max", np.max(model.D_))
    # Normalize weights for plotting
    D_normalized = 10 * model.D_ / np.max(model.D_)

    # Prediction function (use full ensemble)
    predict_func = lambda X: model.partial_predict(X, len(model.models_))

    # Get decision surface
    surface = decision_surface(predict_func, lims[0], lims[1], showscale=False)
    
    # Scatter plot of training points with size and symbol
    scatter = go.Scatter(
        x=train_X[:, 0],
        y=train_X[:, 1],
        mode='markers',
        marker=dict(
            size=D_normalized,
            color='gray',  
            symbol='circle',
            line=dict(width=0.5, color='white'),
        ),
        showlegend=False
    )

    # Combine into figure
    fig = go.Figure(data=[surface, scatter])
    fig.update_layout(
        title="Decision Surface with Weighted Training Points",
        xaxis_title='x1', yaxis_title='x2',
        width=700, height=600
    )
    fig.write_html(f"weighted_decision_surface_noise={noise}.html")
    # fig.show()

def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    model = AdaBoost(DecisionStump,n_learners).fit(train_X,train_y)
    train_errors = [model.partial_loss(train_X, train_y, t) for t in range(1, len(model.models_) + 1)]
    test_errors = [model.partial_loss(test_X, test_y, t) for t in range(1, len(model.models_) + 1)]
    plot_errors(train_errors, test_errors, noise)
    
    # # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    lims = np.clip(lims, -1.0, 1.0) 
    plot_decision_boundaries(model, test_X, test_y, T, lims, noise) 

    # Question 3: Decision surface of best performing ensemble
    plot_best_decision_surface(model, test_X, test_y, test_errors, lims, noise)
    
    # Question 4: Decision surface with weighted samples
    plot_weighted_decision_surface(model, train_X, train_y, lims, noise)

if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)