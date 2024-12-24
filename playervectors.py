import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


class PlayerHeatMap:
    def __init__(self,
                 shape: tuple[int, int]=(50, 50),
                 n_components: int=1,
                 sigma: float = 1.0,
                 player_name: str | None=None,
                 player_id: int | None=None,
                 acion_name: str | None=None,
                 action_id: int | None=None):
        self.shape_ = shape 
        self.components = n_components 
        self.sigma = sigma
        self.player_name = player_name
        self.player_id = player_id
        self.action_name = acion_name
        self.action_id = action_id

        self.raw_counts_ = np.zeros(shape=self.shape_, dtype=np.int16)
        self.heatmap_ = np.zeros(shape=self.shape_, dtype=np.float16)
        self.weights_ = None

    def fit(self, x: np.ndarray | list[int], y: np.ndarray | list[int]) -> None:
        self.raw_counts_, _, _ = np.histogram2d(x, y, bins=[self.shape_[0], self.shape_[1]])
        self.heatmap_ = gaussian_filter(self.raw_counts_, sigma=self.sigma)

    def shape(self) -> tuple[int, int]:
        return self.shape 

    def heatmap(self) -> plt.plot:
        plt.title(f'Heatmap for action: {self.action_name}')
        return sns.heatmap(self.heatmap_)
    
    def raw_counts(self) -> plt.plot:
        plt.title(f'Raw counts for action: {self.action_name}')
        return sns.heatmap(self.raw_counts_)