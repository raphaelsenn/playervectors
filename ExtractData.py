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