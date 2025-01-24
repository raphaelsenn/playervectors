import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import NMF


class PlayerVectors:
    """
    A class to fit a playervectors to a soccer-player with actions event stream data
    """ 
    def __init__(self,
                 shape: tuple[int, int] = (50, 50), 
                 sigma: float=1.0, 
                 random_state: int=0) -> None:
        """
        Parameters:
        -----------
        
        
        """


        self.shape_ = shape
        self.sigma_ = sigma
        self.random_state = random_state

        self.playerID_to_heatmap = {}

        self.M = None
        self.W = None
        self.H = None

    def fit(self,
            action,
            coordinates,
            k_components,
            minutes_played,
            player_names) -> None:
        """
        Fit data to playervectors 
        
        Parameters:
        -----------



        """ 
        # TODO: Not finished, i will work on this.

        # 1. Calculate for every player a heatmap
        # 1.1 - 1.3: Counting + Normalizing + Smoothing
        
        playerID_to_heatmap = {}
        for playerID, pairXY in list(coordinates[action].items()):
            x, y = pairXY[0], pairXY[1] 

            # Check for played minutes 
            minutes = 0.0 
            if playerID in minutes_played: 
                minutes = minutes_played[playerID] 

            # Check for player name
            player_name = None
            if playerID in player_names:
                player_name = player_names[playerID]

            # Build Heatmap
            heatmap = PlayerHeatMap(shape=self.shape_,
                                    player_name=player_name,
                                    player_id=playerID,
                                    action_name=action,
                                    sigma=self.sigma_)
            
            # Counting + Normalizing + Smoothing 
            heatmap.fit(x, y, minutes)

            playerID_to_heatmap[playerID] = heatmap

        # 2. Build matrix M
        num_players = len(playerID_to_heatmap)
        
        M = [] 
        for _, X in playerID_to_heatmap.items():
            X_reshape = X.heatmap_.reshape(self.shape_[0] * self.shape_[1], 1)
            M.append(X_reshape)
        self.M = np.array(M).reshape(num_players, self.shape_[0] * self.shape_[1])

        # 2.1 Apply NMF
        model = NMF(n_components=k_components, init='random', random_state=0)
        self.W = model.fit_transform(M)
        self.H = model.components_



class PlayerHeatMap:
    """
    A class to represent a heatmap for a soccer-player with specified actions 
    """ 
    
    def __init__(self,
                 shape: tuple[int, int]=(50, 50),
                 map_size: tuple[tuple[int, int], tuple[int, int]] = ((0, 100), (0, 100)),
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
            resolution of heatmap matrix
        map_size: tuple[tuple[int, int], tuple[int, int]]
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
        self.map_size = map_size
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
        
        NOTE: Please note that the histogram does not follow the Cartesian convention where x values are on the abscissa,
        and y values on the ordinate axis. Rather, x is histogrammed along the first dimension of the array (vertical),
        and y along the second dimension of the array (horizontal). This ensures compatibility with histogramdd. 
        """ 
        # Building a Player Heatmap 

        # 1. Counting 
        self.raw_counts_, _, _ = np.histogram2d(y, x, bins=[self.shape_[0], self.shape_[1]], range=self.map_size)
        X = self.raw_counts_

        # 2. Normalizing (only if minutes_played > 0.0)
        if minutes_played > 0.0: 
            self.normed_counts_ = self.raw_counts_ / minutes_played
            X = self.normed_counts_
    
        # 3. Smoothing
        self.heatmap_ = gaussian_filter(X, sigma=self.sigma)

    def shape(self) -> tuple[int, int]:
        """
        Returns the shape of the heatmap as a tuple[int, int] 
        """
        return self.shape_

    def heatmap(self) -> plt.plot:
        """
        Uses seaborn to plot a smoothed heatmap 
        """ 
        if self.player_name is not None: 
            plt.title(f'Heatmap of {self.player_name} for action: {self.action_name}')
        else: 
            plt.title(f'Heatmap for action: {self.action_name}')
        return sns.heatmap(self.heatmap_)
    
    def raw_counts(self) -> plt.plot:
        """
        Uses seaborn to plot heatmap with raw_counts (like a scatter plot) 
        """ 
        if self.player_name is not None: 
            plt.title(f'Raw counts of {self.player_name} for action: {self.action_name}')
        else: 
            plt.title(f'Raw counts for action: {self.action_name}')
        return sns.heatmap(self.raw_counts_)

    def normed_counts(self) -> plt.plot:
        """
        Uses seaborn to plot heatmap with normalized counts (like a scatter plot) 
        """ 
        if self.player_name is not None: 
            plt.title(f'Normalized counts of {self.player_name} for action: {self.action_name}')
        else: 
            plt.title(f'Normed counts for action: {self.action_name}')
        return sns.heatmap(self.normed_counts_)