import torch
import numpy as np
import matplotlib.pyplot as plt


def update_H(W: np.array, H: np.array, V: np.array) -> np.array:
  """
  Update rule for H using a Euclidean distance cost, based on Lee & Seung (2000)
  """
  return H * (W.T @ V)/(W.T @ W @ H)

def update_W(W: np.array, H: np.array, V: np.array) -> np.array:
  """
  Update rule for W using a Euclidean distance cost, based on Lee & Seung (2000)
  """
  return W * (V @ H.T)/(W @ H @ H.T)

def calc_euclidean_dist(V: np.array, H: np.array, W: np.array) -> float:
  """
  Euclidean distance cost, based on Lee & Seung (2000)
  """
  res = V - W @ H 
  return np.sum(np.square(res))

def do_NMF(V: np.array, rank: int=10, iter: int=100) -> tuple:
  """
  Update rules using a Euclidean distance cost, based on Lee & Seung (2000)
  """
  W = np.random.rand(V.shape[0], rank)
  H = np.random.rand(rank, V.shape[1])
  history = []
  history.append(calc_euclidean_dist(V, H, W))
  for _ in range(iter):
    H = update_H(W, H, V)
    H = np.where(H >= 0, H, 0)
    W = update_W(W, H, V)
    W = np.where(W >= 0, W, 0)
    history.append(calc_euclidean_dist(V, H, W))
  return W, H, history


def plot_gallery(images: np.array, 
                 n_col: int = 3, 
                 n_row: int = 2, 
                 image_shape: int=64,
                 cmap=plt.cm.gray):
    fig, axs = plt.subplots(
        nrows=n_row,
        ncols=n_col,
        figsize=(2.0 * n_col, 2.3 * n_row),
        facecolor="white",
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
    fig.set_edgecolor("black")
    for ax, vec in zip(axs.flat, images):
        vmax = max(vec.max(), -vec.min())
        im = ax.imshow(
            vec.reshape(image_shape),
            cmap=cmap,
            interpolation="nearest",
            vmin=0,
            vmax=vmax,
        )
        ax.axis("off")
    fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.99, aspect=40, pad=0.01)
    plt.show()


