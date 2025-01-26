import numpy as np
import seaborn as sns
import warnings
import math
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
        self.player_vectors = {}
        self.action_M = {}
        self.action_W = {}
        self.action_H = {}

    def count_norm_smooth(self, coordinates: dict[str, dict[int, tuple[list[int], list[int]]]],
            actions: list[str],
            player_names: dict[int, str],
            minutes_played: dict[int, int]) -> dict:
        """
        1. Counting: 
            Counting We overlay a grid of size (m x n) on the soccer field.
            Per grid cell X[i][j] , we count the number of actions that started in that cell.

        2. Normalizing:
        """ 
        
        # --------------------------------------------------------------------
        # Constructing Heatmaps 
        
        # Mapping of player IDs to a list of heatmaps for their respective actions
        # i.e. 43 -> [heatmap_shot, heatmap_cross, heatmap_dribble, heatmap_pass], 54 -> [heatmap_shot, ...]
        player_heatmaps = {}
        for action in actions:
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
        return player_heatmaps

    def reshaping(self, player_heatmaps: dict) -> dict:
        # --------------------------------------------------------------------
        # Reshaping Heatmaps to Vectors + Construction of Matrix M

        # Mapping each action type to its corresponding matrix (M_action)
        # i.e. shot -> M_shot, cross -> M_cross, dirbble -> M_dribble, pass -> M_pass
        action_to_matrix = {action: [] for action in self.actions}

        # Store corresponding player IDs
        # i.e. shot -> [34, 234, 232, ... list of player_ids for action shot], cross -> [454, 643, ...]
        action_to_player = {action: [] for action in self.actions}

        # Reshaping
        for playerID, list_heatmaps in player_heatmaps.items():
            
            # list_heatmaps of structure: [heatmap_shot, heatmap_cross, heatmap_dribble, heatmap_pass]
            for action_index, X in enumerate(list_heatmaps):
                if len(list_heatmaps) == len(self.components): 
                    # Reshaping heatmap of shape (m, n) to vector of shape (n * m, 1) 
                    X_reshape = X.reshape(self.grid[0] * self.grid[1], 1)
                    
                    # Store reshaped heatmap and corresponding player ID
                    action_to_matrix[self.actions[action_index]].append(X_reshape)
                    
                    # Save corresponding player_ids
                    action_to_player[self.actions[action_index]].append(playerID)

        # Construction of matrix M:
        for action in action_to_matrix:
            M = np.array(action_to_matrix[action])
            # M = M.reshape(self.grid[0] * self.grid[1], len(M))
            M = M.reshape(len(M), self.grid[0] * self.grid[1]).T 
            action_to_matrix[action] = M
        return action_to_matrix, action_to_player

    def compress(self,
                 action_to_matrix: dict,
                 verbose: bool=False) -> dict:
        # -----------------------------------------------------------------------------
        # Compress matrix M by applying non-negative matrix factorization (NMF)

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

            # Update member variables 
            self.action_M[action] = M
            self.action_W[action] = W
            self.action_H[action] = H
 
            if verbose:
                print(f'Action: {action}\tShape of M: {M.shape}\tShape of W: {W.shape}\tShape of H: {H.shape}')
        return actions_to_vectors

    def assemble(self,
                 actions_to_vectors: dict,
                 action_to_player: dict) -> dict:
        # ------------------------------------------------------
        # Assemble Player Vectors

        # Mapping player'ids to corresponding 18-component player vector
        player_to_vector = {}

        for action, princ_vectors in actions_to_vectors.items():
            for i, vector in enumerate(princ_vectors):
                playerID = action_to_player[action][i]
                if playerID not in player_to_vector:
                    player_to_vector[playerID] = []
                player_to_vector[playerID].extend(vector.flatten())
        return player_to_vector

    def fit(self,
            coordinates: dict[str, dict[int, tuple[list[int], list[int]]]],
            minutes_played: dict[int, int],
            player_names: dict[int, str],
            verbose: bool=False) -> None:
        """
        Fit data to playervectors 
        
        Parameters:
        -----------
        
        
        Variables: 
        -----------
        player_heatmaps : dict[int, list[np.ndarray]]
            Mapping of player IDs to a list of heatmaps for their respective actions
            i.e. 43 -> [heatmap_shot, heatmap_cross, heatmap_dribble, heatmap_pass], 54 -> [heatmap_shot, ...] 
        
        action_to_matrix : dict[str, np.ndarray]
            Mapping each action type to its corresponding matrix (M_action)
            i.e. shot -> M_shot, cross -> M_cross, dirbble -> M_dribble, pass -> M_pass
        
        action_to_player : dict[str, list[int]]
            Store corresponding player IDs
            i.e. shot -> [34, 234, 232, ... list of player_ids for action shot], cross -> [454, 643, ...]
        
        actions_to_vector : dict[str, np.ndarray] 
            Dictionary to store the resulting feature vectors for each action type
            i.e. shot -> []

        self.player_vectors : dict[int, np.ndarray]
            Mapping player'ids to corresponding 18-component player vector
        """
        # --------------------------------------------------------------------
        # 1. Selecting Relevant Action Types 
        # Done, when PlayerVectors object is created.
        
        # --------------------------------------------------------------------
        # 2. Constructing Heatmaps (Counting + Normalizing + Smoothing)
        player_heatmaps = self.count_norm_smooth(coordinates,
                                                 self.actions,
                                                 player_names,
                                                 minutes_played)
        
        # --------------------------------------------------------------------
        # (3): Compressing Heatmaps to Vectors
        action_to_matrix, action_to_player = self.reshaping(player_heatmaps)

        # -----------------------------------------------------------------------------
        # (3.3): Compress matrix M by applying non-negative matrix factorization (NMF)
        actions_to_vectors = self.compress(action_to_matrix, verbose=verbose)
 

        # ------------------------------------------------------
        # (4): Assemble Player Vectors
        self.player_vectors = self.assemble(actions_to_vectors, action_to_player)

    def plot_principle_components(self,
                                  figsize: tuple[int, int]=(20, 40)) -> plt.plot:
        total_components = sum(self.components)

        # Determine the number of rows, ensuring it's an integer
        rows = math.ceil(total_components / 2)
        
        # Create subplots with calculated rows and 2 columns
        fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=figsize)

        # Flatten axes if there's more than one row, to simplify indexing
        if rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes]  # Make it iterable if only one row

        n = 0
        for i, (action, W) in enumerate(self.action_W.items()):
            for k in range(self.components[i]):
                W_k = W.T[k].reshape(self.grid[0], self.grid[1])
                sns.heatmap(W_k, ax=axes[n])
                axes[n].set_title(f'Component {n + 1} ({action})')
                axes[n].set_xlabel('Attack -->') 
                n += 1

        # Adjust layout to avoid overlap
        plt.tight_layout()

        return fig  # Return the figure object


class PlayerHeatMap:
    """
    A class to represent a heatmap for a soccer-player with specified actions 
    """ 
    
    def __init__(self,
                 shape: tuple[int, int]=(50, 50),
                 map_size: tuple[tuple[int, int], tuple[int, int]] = ((0, 100), (0, 100)),
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
        
        map_size : tuple[tuple[int, int], tuple[int, int]]
            Dimension of heatmap matrix
        
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
        # Parameters 
        self.shape_ = shape
        self.map_size = map_size
        self.sigma = sigma
        self.player_id = player_id
        self.action_id = action_id
        
        # Member variables 
        self.raw_counts_ = np.zeros(shape=self.shape_, dtype=np.int16)
        self.normed_counts_ = np.zeros(shape=self.shape_, dtype=np.float16)
        self.heatmap_ = np.zeros(shape=self.shape_, dtype=np.float16)
        
        self.player_name = player_name
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
            self.normed_counts_ = self.raw_counts_ * (90.0 / minutes_played)
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