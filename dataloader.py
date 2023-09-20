import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import sklearn.datasets
import torch


plt.rcParams['figure.dpi'] = 140

DEFAULT_X_RANGE = (-3.5, 3.5)
DEFAULT_Y_RANGE = (-2.5, 2.5)
DEFAULT_CMAP = colors.ListedColormap(["#377eb8", "#ff7f00"])
DEFAULT_NORM = colors.Normalize(vmin=0, vmax=1,)
DEFAULT_N_GRID = 100

def make_training_data(sample_size=500):
  """Create two moon training dataset."""
  train_examples, train_labels = sklearn.datasets.make_moons(
      n_samples=2 * sample_size, noise=0.1)

  # Adjust data position slightly.
  train_examples[train_labels == 0] += [-0.1, 0.2]
  train_examples[train_labels == 1] += [0.1, -0.2]

  return train_examples, train_labels

def make_testing_data(x_range=DEFAULT_X_RANGE, y_range=DEFAULT_Y_RANGE, n_grid=DEFAULT_N_GRID):
  """Create a mesh grid in 2D space."""
  # testing data (mesh grid over data space)
  x = np.linspace(x_range[0], x_range[1], n_grid)
  y = np.linspace(y_range[0], y_range[1], n_grid)
  xv, yv = np.meshgrid(x, y)
  return np.stack([xv.flatten(), yv.flatten()], axis=-1)

def make_ood_data(sample_size=500, means=(2.5, -1.75), vars=(0.01, 0.01)):
  return np.random.multivariate_normal(
      means, cov=np.diag(vars), size=sample_size)

#Load the train, test and OOD datasets.
def two_moon_data(sample_size=500,show_plot=None):
    train_examples, train_labels = make_training_data(
        sample_size=sample_size)
    test_examples = make_testing_data()
    ood_examples = make_ood_data(sample_size=sample_size)

    test_data = torch.tensor(test_examples)
    ood_data = torch.tensor(ood_examples)

    if show_plot:
        
        pos_examples = train_examples[train_labels == 0]
        neg_examples = train_examples[train_labels == 1]
        plt.figure(figsize=(7, 5.5))

        plt.scatter(pos_examples[:, 0], pos_examples[:, 1], c="#377eb8", alpha=0.5)
        plt.scatter(neg_examples[:, 0], neg_examples[:, 1], c="#ff7f00", alpha=0.5)
        plt.scatter(ood_examples[:, 0], ood_examples[:, 1], c="red", alpha=0.1)

        plt.legend(["Positive", "Negative", "Out-of-Domain"])

        plt.ylim(DEFAULT_Y_RANGE)
        plt.xlim(DEFAULT_X_RANGE)

        plt.show()

    return train_examples, train_labels,test_data, ood_data

def plot(examples,ood_examples,labels):
    
    pos_examples = examples[labels == 0]
    neg_examples = examples[labels == 1]
    plt.figure(figsize=(7, 5.5))

    plt.scatter(pos_examples[:, 0], pos_examples[:, 1], c="#377eb8", alpha=0.5)
    plt.scatter(neg_examples[:, 0], neg_examples[:, 1], c="#ff7f00", alpha=0.5)
    plt.scatter(ood_examples[:, 0], ood_examples[:, 1], c="red", alpha=0.1)

    plt.legend(["Positive", "Negative", "Out-of-Domain"])

    plt.ylim(DEFAULT_Y_RANGE)
    plt.xlim(DEFAULT_X_RANGE)

    plt.show()


def plot_uncertainty_surface(test_uncertainty, ax,cmap=None):

    # Normalize uncertainty for better visualization.
    test_uncertainty = test_uncertainty.detach().numpy()
    test_uncertainty = test_uncertainty / np.max(test_uncertainty)

    # Set view limits.
    ax.set_ylim(DEFAULT_Y_RANGE)
    ax.set_xlim(DEFAULT_X_RANGE)

    # Plot normalized uncertainty surface.
    pcm = ax.imshow(
        np.reshape(test_uncertainty, [DEFAULT_N_GRID, DEFAULT_N_GRID]),
        cmap=cmap,
        origin="lower",
        extent=DEFAULT_X_RANGE + DEFAULT_Y_RANGE,
        vmin=DEFAULT_NORM.vmin,
        vmax=DEFAULT_NORM.vmax,
        interpolation='bicubic',
        aspect='auto')

    train_examples,train_labels, _, ood_examples = two_moon_data()

    # train_examples, train_labels = next(iter(train_data))
    # Plot training data.
    ax.scatter(train_examples[:, 0], train_examples[:, 1],
                c=train_labels, cmap=DEFAULT_CMAP, alpha=0.5)
    ax.scatter(ood_examples[:, 0], ood_examples[:, 1], c="red", alpha=0.1)

    return pcm

def plot_predictions(pred_probs, filename, model_name=""):
    """Plot normalized class probabilities and predictive uncertainties."""

    # Compute predictive uncertainty.
    uncertainty = pred_probs * (1. - pred_probs)

    # Initialize the plot axes.
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Plots the class probability.
    pcm_0 = plot_uncertainty_surface(pred_probs, ax=axs[0])
    # Plots the predictive uncertainty.
    pcm_1 = plot_uncertainty_surface(uncertainty, ax=axs[1])

    # Adds color bars and titles.
    fig.colorbar(pcm_0, ax=axs[0])
    fig.colorbar(pcm_1, ax=axs[1])

    axs[0].set_title(f"Class Probability, {model_name}")
    axs[1].set_title(f"(Normalized) Predictive Uncertainty, {model_name}")

    plt.savefig(filename,dpi=300)
    # plt.show()