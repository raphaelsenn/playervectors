import pandas as pd
import ast
import ConditionData as CData
import numpy as np
import playervectors as playVec
import matplotlib.pyplot as plt
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
def get_directions(keepers, player: dict[int,dict[str,str|int]],  match_diretion_filp):

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


    def flip_coor( dataset: pd.DataFrame)->pd.DataFrame:
        for i in range(len(dataset["pos_orig_y"])):
            if (dataset.iloc[i]["teamId"],dataset.iloc[i]["matchId"],dataset.iloc[i]["matchPeriod"]) in match_diretion_filp.keys():
                dataset.iloc[i]["pos_orig_y"] = 100- dataset.iloc[i]["pos_orig_y"] 
                dataset.iloc[i]["pos_orig_x"] = 100- dataset.iloc[i]["pos_orig_x"] 
                dataset.iloc[i]["pos_dest_y"] = 100- dataset.iloc[i]["pos_dest_y"] 
                dataset.iloc[i]["pos_dest_x"] = 100- dataset.iloc[i]["pos_dest_x"] 
        return dataset
    return flip_coor

if __name__ == "__main__":
    data = pd.read_csv(filepath_or_buffer= "data/players.csv", sep=",")
    print(len(ExtractGoalkeepers(data).keys()))
    empyt_dict ={}
    player_dataframe ={}

    func = get_directions(ExtractGoalkeepers(data),
                          ExtractPlayers(df= data,  wy_id='wyId', attributes=['firstName', 'lastName', 'currentTeamId']),
                            empyt_dict)
    func2 = outside_flip_coor(empyt_dict)

    func3 = outside_player_to_dataframe(player_dataframe)
    con = CData.ConditionData(dataset_name="test",
                              _dataset_link="data/example_data.csv",
                               _conditions = [], 
                               _rewrite=[func,func2, func3],
                                 _delete_colum = [],
                                 flip_sec_half_coordinates=False)
    con.create_conditionData()

    player_map= CData.ConditionData("heatmap", [],[],[],"")
    player_map.dataset= player_dataframe[279772]
    player_map.create_conditionData(read_again_csv=False)
    print( player_map.dataset)

    plt.figure(1, figsize=(12, 6))
    player_map.fit("pos_orig_x", "pos_orig_y")
    player_map.heatmap()
    plt.show()
    con.creat_file("data/Example")

