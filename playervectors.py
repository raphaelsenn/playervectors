import numpy as np
import seaborn as sns
import warnings
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
                 actions: list[str] = ['shot', 'cross', 'dribble', 'pass'],
                 components: list[int] =[4, 4, 5, 5],
                 init_nmf: str='nndsvd',
                 max_iter_nmf: int=500,
                 random_nmf: int=42,
                 loss_nmf: str='frobenius') -> None:
        """
        Parameters:
        -----------
        """

        # ------------------------------------------------------
        # Hyperparameters:
        self.grid = shape
        self.sigma = sigma
        self.actions = actions
        self.components = components
        self.init_nmf = init_nmf
        self.max_iter = max_iter_nmf
        self.random_nmf = random_nmf
        self.loss_nmf = loss_nmf

        # Mapping player'ids to corresponding k-component player vector
        self.player_vectors = None
    
    def fit(self,
            coordinates: dict[str, dict[int, tuple[list[int], list[int]]]],
            minutes_played: dict[int, int],
            player_names: dict[int, str],
            verbose: bool=False) -> None:
        """
        Fit data to playervectors 
        
        Parameters:
        -----------



        """
        # --------------------------------------------------------------------
        # 1. Selecting Relevant Action Types 
        # Done, when PlayerVectors object is created.
        
        # --------------------------------------------------------------------
        # 2. Constructing Heatmaps 
        
        # Mapping of player IDs to a list of heatmaps for their respective actions
        # i.e. 43 -> [heatmap_shot, heatmap_cross, heatmap_dribble, heatmap_pass], 54 -> [heatmap_shot, ...]
        player_heatmaps = {}
        for action in self.actions:
            for playerID, coordinates_xy in list(coordinates[action].items()):
                if playerID not in player_heatmaps:
                    player_heatmaps[playerID] = [] 
                
                x, y = coordinates_xy[0], coordinates_xy[1] 

                # Check for played minutes 
                minutes = 0.0 
                if playerID in minutes_played: 
                    minutes = minutes_played[playerID] 

                # Check for player name
                player_name = None
                if playerID in player_names:
                    player_name = player_names[playerID]

                # Build Player-Vectors
                heatmap = PlayerHeatMap(shape=self.grid,
                                        player_name=player_name,
                                        player_id=playerID,
                                        action_name=action,
                                        sigma=self.sigma,
                                        map_size=[[0, 100], [0, 100]])

                # 1 - 3: Counting + Normalizing + Smoothing
                heatmap.fit(x, y, minutes)
                player_heatmaps[playerID].append(heatmap.heatmap_)

        
        # --------------------------------------------------------------------
        # (3): Compressing Heatmaps to Vectors

        # Mapping each action type to its corresponding matrix (M_action)
        # i.e. shot -> M_shot, cross -> M_cross, dirbble -> M_dribble, pass -> M_pass
        action_to_matrix = {action: [] for action in self.actions}

        # Store corresponding player IDs
        # i.e. shot -> [34, 234, 232, ... list of player_ids for action shot], cross -> [454, 643, ...]
        action_to_player = {action: [] for action in self.actions}

        # (3.1): Reshaping
        for playerID, list_heatmaps in player_heatmaps.items():
            num_players = len(player_heatmaps)
            
            # list_heatmaps of structure: [heatmap_shot, heatmap_cross, heatmap_dribble, heatmap_pass]
            for action_index, X in enumerate(list_heatmaps):
                # Reshaping heatmap of shape (m, n) to vector of shape (n * m, 1) 
                X_reshape = X.reshape(self.grid[0] * self.grid[1], 1)
                
                # Store reshaped heatmap and corresponding player ID
                action_to_matrix[self.actions[action_index]].append(X_reshape)
                
                # Save corresponding player_ids
                action_to_player[self.actions[action_index]].append(playerID)

        # (3.2): Construction of matrix M:
        for action in action_to_matrix:
            M = np.array(action_to_matrix[action])
            M = M.reshape(self.grid[0] * self.grid[1], len(M))
            action_to_matrix[action] = M


        # -----------------------------------------------------------------------------
        # (3.3): Compress matrix M by applying non-negative matrix factorization (NMF)
        # -----------------------------------------------------------------------------

        # Dictionary to store the resulting feature vectors for each action type
        # i.e. shot -> []
        actions_to_vectors = {}
        for i, (action, M) in enumerate(action_to_matrix.items()):
            # Suppress all convergence warnings from sklearn
            warnings.filterwarnings("ignore", category=UserWarning) 
            
            # Apply NMF to compress the heatmap matrix M into lower-dimensional space
            model = NMF(n_components=self.components[i],
                        init=self.init_nmf,
                        random_state=self.random_nmf,
                        max_iter=self.max_iter) 

            # Factorizing M: M ≈ W @ H
            W = model.fit_transform(M)
            H = model.components_

            principal_vector = np.split(H, H.shape[1], axis=1) 
            actions_to_vectors[action] = principal_vector

            if verbose:
                print(f'Action: {action}\tShape of M: {M.shape}\tShape of W: {W.shape}\tShape of H: {H.shape}')


        # ------------------------------------------------------
        # (4): Assemble Player Vectors

        # Mapping player'ids to corresponding 18-component player vector
        player_to_vector = {}

        for action, princ_vectors in actions_to_vectors.items():
            for i, vector in enumerate(princ_vectors):
                playerID = action_to_player[action][i]
                if playerID not in player_to_vector:
                    player_to_vector[playerID] = []
                player_to_vector[playerID].extend(vector.flatten())
        
        self.player_vectors = player_to_vector


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

    def heatmap(self, figsize=(10, 5)) -> plt.plot:
        """
        Uses seaborn to plot a smoothed heatmap 
        """ 
        # Adjust figure size 
        plt.figure(figsize=figsize) 

        if self.player_name is not None: 
            plt.title(f'Heatmap of {self.player_name} for action: {self.action_name}')
        else: 
            plt.title(f'Heatmap for action: {self.action_name}')

        # Plot heatmap
        ax = sns.heatmap(self.heatmap_)

        # Add Attacking direction
        plt.xlabel('Attack -->')  

        # Return current figure object
        return plt.gcf() 
    
    def raw_counts(self, figsize=(10, 5)) -> plt.plot:
        """
        Uses seaborn to plot heatmap with raw_counts (like a scatter plot) 
        """ 
        # Adjust figure size 
        plt.figure(figsize=figsize) 
        
        if self.player_name is not None: 
            plt.title(f'Raw counts of {self.player_name} for action: {self.action_name}')
        else: 
            plt.title(f'Raw counts for action: {self.action_name}')
        
        # Plot heatmap
        ax = sns.heatmap(self.raw_counts_)

        # Add Attacking direction
        plt.xlabel('Attack -->')  

        # Return current figure object
        return plt.gcf()  

    def normed_counts(self, figsize=(10, 5)) -> plt.plot:
        """
        Uses seaborn to plot heatmap with normalized counts (like a scatter plot) 
        """ 
        # Adjust figure size 
        plt.figure(figsize=figsize) 
        
        if self.player_name is not None: 
            plt.title(f'Normalized counts of {self.player_name} for action: {self.action_name}')
        else: 
            plt.title(f'Normed counts for action: {self.action_name}')
        
        # Plot heatmap
        ax = sns.heatmap(self.normed_counts_)

        # Add Attacking direction
        plt.xlabel('Attack -->')  

        # Return current figure object
        return plt.gcf() 