import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import NMF


class PlayerHeatMap:
    """
    A class to represent a heatmap for a soccer-player with specified actions 
    """ 
    
    def __init__(self,
                 shape: tuple[int, int]=(50, 50),
                 n_components: int=1,
                 sigma: float = 1.0,
                 player_name: str | None=None,
                 player_id: int | None=None,
                 action_name: str | None=None,
                 action_id: int | None=None):
        """
        Parameters:
        __________
        shape : tuple[int, int]
            Dimension of heatmap matrix
        
        n_components : int
            Components for NMF
        
        sigma : float
            Parameter for gaussian_filter

        player_name : str
            Name of the player

        player_id : int
            ID for player

        action_name : str
            Name of the action

        action_id : int
            ID for action 
        """ 
        self.shape_ = shape 
        self.components = n_components 
        self.sigma = sigma
        self.player_id = player_id
        self.action_id = action_id
        self.raw_counts_ = np.zeros(shape=self.shape_, dtype=np.int16)
        self.normed_counts_ = np.zeros(shape=self.shape_, dtype=np.float16)
        self.heatmap_ = np.zeros(shape=self.shape_, dtype=np.float16)
        self.weights_ = None

        # Decode bytes to string, or handle strings as-is
        if isinstance(player_name, bytes):
            self.player_name = player_name.decode('utf-8')
        else:
            self.player_name = player_name

        if isinstance(action_name, bytes):
            self.action_name = action_name.decode('utf-8')
        else:
            self.action_name = action_name

    def fit(self, x: np.ndarray | list[int],
            y: np.ndarray | list[int],
            minutes_played: float=0.0) -> None:
        """
        Fits Heatmap to Soccer-Action data 

        Parameters:
        -----------
        x : np.ndarray or list[int]
            List of x coordinates where player perfroms specified action
        
        y : np.ndarray or list[int]
            List of y coordinates where player perfroms specified action

        minutes_played : float
            Total minutes played (should be with respect to x, y coordinates)
        """ 
        self.raw_counts_, _, _ = np.histogram2d(x, y, bins=[self.shape_[0], self.shape_[1]])
        self.heatmap_ = gaussian_filter(self.raw_counts_, sigma=self.sigma)
        X = self.raw_counts_
        
        # Normalize if played minutes are given. 
        if minutes_played > 0.0: 
            self.normed_counts_ = self.raw_counts_ / minutes_played
            X = self.normed_counts_

        self.heatmap_ = gaussian_filter(X, sigma=self.sigma)

    def shape(self) -> tuple[int, int]:
        """
        Returns the shape of the heatmap as a tuple[int, int] 
        """
        return self.shape_

    def heatmap(self) -> plt.plot:
        if self.player_name is not None: 
            plt.title(f'Heatmap of {self.player_name} for action: {self.action_name}')
        else: 
            plt.title(f'Heatmap for action: {self.action_name}')
        return sns.heatmap(self.heatmap_)
    
    def raw_counts(self) -> plt.plot:
        plt.title(f'Raw counts for action: {self.action_name}')
        return sns.heatmap(self.raw_counts_)

    def normed_counts(self) -> plt.plot:
        plt.title(f'Normed counts for action: {self.action_name}')
        return sns.heatmap(self.normed_counts_)
