import pandas as pd
import ast
from typing import  Callable
def ExtractPlayers(df: pd.DataFrame,
                   wy_id: str,
                   attributes: list[str]) -> dict[int, list]:
    """
    Returns a Dictionary which Maps WyScout Player-ID to the Players attributes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        A DataFrame from which to extract relevant information

    wy_id : str
        Column name of DataFrame where playerID is stored

    attributes : list[str]
        List of column names of DataFrame df that should be extracted

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
        player_attributes = {}#[row[attrib] for attrib in attributes]
        for attrib in attributes:
            player_attributes[attrib] = row[attrib]
        # Add to dictionary
        players[player_id] = player_attributes
    return players


def ExtractGoalkeepers(df: pd.DataFrame,
                        id_col: str = 'wyId',
                        role_col: str='role',
                        wy_code: str='code2',
                        keeper_str: str='GK') -> dict:
    goalkeepers = {} 
    for _, row in df.iterrows():
        role_string = row[role_col]
        role_dict = ast.literal_eval(role_string)
        player_ID = row[id_col]  
        player_role = role_dict[wy_code] 

        if player_role == keeper_str:
            if player_ID not in goalkeepers:
                goalkeepers[player_ID] = True
    return goalkeepers
def get_directions(keepers,   match_diretion_filp):
    """
    a wrapper that take an dict with only keepers 
    and returns a func which stores in the dict:match_diretion_filp
    as a key an bool if the coordinates (depending on match , team and half time) musst be flip
    """
    def store_macthes_and_team_direction(dataset: pd.DataFrame)-> pd.DataFrame:
        for _, row in dataset.iterrows():
            player_id = row["playerId"]
            if keepers.get(player_id, False):
                if int(row["pos_orig_y"] )>50:
                    team_id = row["teamId"]#player[player_id]["currentTeamId"]
                    match_id = row["matchId"]
                    halftime = row["matchPeriod"]
                    if (team_id,match_id,halftime) not in match_diretion_filp:
                        match_diretion_filp[(team_id,match_id,halftime)] = True
        return dataset
    return store_macthes_and_team_direction

def outside_player_to_dataframe(player_heatmap:dict[int,pd.DataFrame]):
    """
    a wrapper for a function which  creates a dict,
    to  map  all player id to a data set which contains
    only the action of the player
    """
    def player_to_dataframe(dataset:pd.DataFrame):
        for i in range(len(dataset["playerId"])):
            player_id = dataset.loc[i, "playerId"]
            if player_id not in player_heatmap:
                columen_names = dataset.columns.tolist()
                player_heatmap[player_id] = pd.DataFrame(columns= columen_names)
            player_heatmap[player_id]  = pd.concat([player_heatmap[player_id], dataset.iloc[[i]]], ignore_index=True)
            
        return dataset


                
    return player_to_dataframe

def outside_flip_coor(match_diretion_filp: dict):
    """
    wrapper for a function that flips every coordinate in a data set 
    depending if row has all attributes which corresponds to a key in dict: match_diretion_filp
    """

    def flip_coor( dataset: pd.DataFrame)->pd.DataFrame:
        for i in range(len(dataset["pos_orig_y"])):
            if (dataset.iloc[i]["teamId"],dataset.iloc[i]["matchId"],dataset.iloc[i]["matchPeriod"]) in match_diretion_filp.keys():
                dataset.iloc[i]["pos_orig_y"] = 100- dataset.iloc[i]["pos_orig_y"] 
                dataset.iloc[i]["pos_orig_x"] = 100- dataset.iloc[i]["pos_orig_x"] 
                dataset.iloc[i]["pos_dest_y"] = 100- dataset.iloc[i]["pos_dest_y"] 
                dataset.iloc[i]["pos_dest_x"] = 100- dataset.iloc[i]["pos_dest_x"] 
        return dataset
    return flip_coor
def get_number(str_input:str, start:int = 0):
    """
    gives you the first number in a string starting at index=start
    """
    str_to_int = ""
    index= start
    break_var = False
    while True:
        if index>=len(str_input):
            return None
        if str_input[index] in "0123456789":
            str_to_int+=str_input[index]
            break_var =True
        elif break_var:
            break
        index+=1
    return int(str_to_int)
def get_abc(str_input:str, start:int = 0):
    """
    give you the first word in a string
    (word is def as a string of letters)
    """
    str_new = ""
    index= start
    break_var = False
    while True:
        if index>=len(str_input):
            return None
        if str_input[index].isalpha():
            str_new+=str_input[index]
            break_var =True
        elif break_var:
            break
        index+=1
    return str_new
def get_match_context(match_to_home:dict[int, list[tuple[int,int,str, int,int,int]]])-> Callable[[pd.DataFrame],pd.DataFrame]:
    """
    is a wrapper for a funtion who stores in the dict you giva as argument
    all machtid map to kontext parameters

    """
    def inside_get_match_context(dataset:pd.DataFrame)-> pd.DataFrame:
        for i in range(len(dataset["teamsData"])):


            side_team1 =dataset.iloc[i]["team1.side"]
            side_team2 =dataset.iloc[i]["team2.side"]
            team_id1 =dataset.iloc[i]["team1.teamId"] 
            team_id2 =dataset.iloc[i]["team2.teamId"]
            team1__score = int(dataset.iloc[i]["team1.score"])
            team2__score = int(dataset.iloc[i]["team2.score"])
            team1_winner = "win"
            team2_winner = "lose"
            if team1__score<team2__score:
                team1_winner ="lose"
                team2_winner ="win"
            elif team1__score==team2__score:
                team1_winner = "tie"
                team2_winner = "tie"



            match_to_home[int( dataset.iloc[i]["wyId"])] =[ 
                (team_id1,side_team1, team1_winner, team1__score,int(dataset.iloc[i]["gameweek"]), get_number(dataset.iloc[i]["referees"],str(dataset.iloc[i]["referees"]).find("refereeId"))),
                (team_id2,side_team2, team2_winner,team2__score,int(dataset.iloc[i]["gameweek"]),  get_number(dataset.iloc[i]["referees"],str(dataset.iloc[i]["referees"]).find("refereeId")))]
        return dataset 


    return inside_get_match_context
