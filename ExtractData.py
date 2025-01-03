import pandas as pd


def ExtractPlayers(df: pd.DataFrame,
                   wy_id: str,
                   attributes: list[str]) -> dict[int, list]:
    """
    Returns a Dictionary which Maps WyScout Player-ID to the Players attributes.

    E.g.
    >>> import pandas as pd
    >>> data = {'ID': [1, 2, 3],
    ...         'FirstName': ['Manuel', 'Bastian', 'Oliver'],
    ...         'LastName': ['Neuer', 'Schweinsteiger', 'Kahn'],
    ...         'Age': [38, 40, 55]}
    
    >>> df_players = pd.DataFrame(data)
    >>> players = ExtractPlayers(df=df_players, wy_id='ID', attributes=['FirstName', 'LastName', 'Age']) 
    
    >>> players 
    {1: ['Manuel', 'Neuer', 38], 2: ['Bastian', 'Schweinsteiger', 40], 3: ['Oliver', 'Kahn', 55]}
    """
    # Map Player-ID to list of player attributes
    players = {}

    # Iterate over Dataframe.
    for _, row in df.iterrows():
        # Extract Player id from DataFrame 
        player_id = row[wy_id]

        # Extract Attributes from DataFrame
        player_attributes = [row[attrib] for attrib in attributes]

        # Add to dictionary
        players[player_id] = player_attributes
    return players


def ExtractTeams(df: pd.DataFrame,
                 wy_id: str,
                 attributes: list[str]) -> dict[int, list]:
    """
    Returns a Dictionary which Maps WyScout Team-ID to the Team attributes.
    
    E.g.
    >>> import pandas as pd
    >>> data = {'ID': [1, 2, 3],
    ...         'TeamName': ['FC Barcelona', 'FC Bayern Muenchen', 'SC Freiburg'],
    ...         'City': ['Barcelona', 'Muenchen', 'Freiburg']}
    >>> df_teams = pd.DataFrame(data)
    >>> teams = ExtractTeams(df=df_teams, wy_id='ID', attributes=['TeamName', 'City']) 
    
    >>> teams
    {1: ['FC Barcelona', 'Barcelona'], 2: ['FC Bayern Muenchen', 'Muenchen'], 3: ['SC Freiburg', 'Freiburg']}
    """
    # Map Team-ID to list of team attributes
    teams = {}

    # Iterate over Dataframe.
    for _, row in df.iterrows():
        # Extract Team-ID from DataFrame 
        team_id = row[wy_id]

        # Extract Attributes from DataFrame
        team_attributes = [row[attrib] for attrib in attributes]

        # Add to dictionary
        teams[team_id] = team_attributes
    return teams


def ExtractMinutesPlayed(df: pd.DataFrame, attributes: list[str]) -> dict[int, float]:
    """
    attributes: [ID, minutes]

    Map PlayerID to sum of played minutes for every game.

    E.g.
    >>> import pandas as pd
    >>> data = {'playerID': [1, 2, 1, 2, 3], 'minutes': [90, 87, 90, 90, 19]} 
    >>> df = pd.DataFrame(data)    
    >>> minutes = ExtractMinutesPlayed(df=df, attributes=['playerID', 'minutes']) 
    
    >>> minutes
    {1: 180, 2: 177, 3: 19}
    """
    # Map playerID to sum of played minutes 
    minutes_played = {}

    for _, row in df.iterrows():
        id, minutes = row[attributes[0]], row[attributes[1]]
        if id not in minutes_played:
            minutes_played[id] = minutes
        else:
            minutes_played[id] += minutes
    return minutes_played


def ExtractCoordinates(df: pd.DataFrame,
                       column_player_id: str,
                       column_event_name: str,
                       column_x: str,
                       column_y: str,
                       actions: list[str]) -> dict[str,
                                                   dict[int, tuple[list[int], list[int]]]]:
    """
    {
     'pass' -> {1 -> [coordinates (x, y) where player with id=1 perfroms action 'pass'], 2 -> [coordinates (x, y) where player with id=2 perfroms action 'pass'], ...}
     'shot' -> {1 -> [coordinates (x, y) where player with id=1 perfroms action 'shot'], ... }
     'dribble' -> {1 -> [coordinates (x, y) where player with id=1 perfroms action 'dribble'], ... }
     ... 
    }
    
    E.g. 
    >>> import pandas as pd
    
    >>> data = {'playerID': [1, 1, 1, 1, 2, 2, 2, 3, 3, 3],
    ...         'action': ['pass', 'pass', 'shot', 'shot', 'pass', 'cross', 'cross', 'pass', 'pass', 'pass'],
    ...         'x': [55, 50, 53, 54, 43, 43, 42, 17, 10, 15],
    ...         'y': [45, 40, 43, 44, 33, 23, 32, 57, 50, 55]}
    >>> df = pd.DataFrame(data=data)
    
    >>> df
       playerID action   x   y
    0         1   pass  55  45
    1         1   pass  50  40
    2         1   shot  53  43
    3         1   shot  54  44
    4         2   pass  43  33
    5         2  cross  43  23
    6         2  cross  42  32
    7         3   pass  17  57
    8         3   pass  10  50
    9         3   pass  15  55
    
    >>> action_coordinates = ExtractCoordinates(df=df,
    ...                                         column_player_id='playerID',
    ...                                         column_event_name='action',
    ...                                         column_x='x',
    ...                                         column_y='y',
    ...                                         actions=['pass', 'shot', 'cross'])
    >>> action_coordinates
    {'pass': {1: ([55, 50], [45, 40]), 2: ([43], [33]), 3: ([17, 10, 15], [57, 50, 55])}, 'shot': {1: ([53, 54], [43, 44])}, 'cross': {2: ([43, 42], [23, 32])}}

    >>> action_coordinates = ExtractCoordinates(df=df,
    ...                                         column_player_id='playerID',
    ...                                         column_event_name='action',
    ...                                         column_x='x',
    ...                                         column_y='y',
    ...                                         actions=['shot', 'dribble'])
    >>> action_coordinates
    {'shot': {1: ([53, 54], [43, 44])}, 'dribble': {}}
    """
    action_coordinates = {} 
    for action in actions:
        df_action = df.loc[df[column_event_name] == action]
        df_action = df_action.groupby([column_player_id]).agg({column_x: list, column_y: list}).reset_index()
        dict_action = df_action.set_index(column_player_id).apply(tuple, axis=1).to_dict()
        action_coordinates[action] = dict_action
    return action_coordinates