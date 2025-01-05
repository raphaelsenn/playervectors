
import playervectors as player
import pandas as pd
from dataclasses import dataclass,field,InitVar
from typing import Callable,Any
import matplotlib.pyplot as plt




@dataclass
class ConditionData:
    """
    dataset_name: name of the instance
    _conditions: InitVar for private variable __conditions
    dataset_link: source data set
    playerheatmap: PlayerHeatMap is an object which handelts the represetation
        of Heatmaps
    flip_sec_half_coordinates: if active, flips the coordinates of the second
        half to match the directions in the first half
    
    __actuell: is a dict with 2 variables which assures that, when an output
        is needed, all datasets fulfil the conditions
    __dataset: private variable witch storage the dataset after applying the
        conditions
    __conditions: private list of functions that dictate which elements are
        allowed in the new dataset
        __conditions has en setter who  also keeps track on the control
        signals  __actuell
    """
    dataset_name: str
    _conditions: InitVar[list[Callable[[Any], bool]]]
    _rewrite: InitVar[list[Callable[[pd.DataFrame], pd.DataFrame]]] 
    _delete_colum: InitVar[list[str]]
    _dataset_link: InitVar[str]
    playerheatmap: player.PlayerHeatMap = field(default_factory=player.PlayerHeatMap)
    flip_sec_half_coordinates: bool = True


    
    __delete_colum: list[str] = field(init=False, default_factory=list)
    __rewrite:list[Callable[[pd.DataFrame], pd.DataFrame]] = field( init=False, default_factory=list)
    __actuell: dict[str, bool] = field(init=False,default_factory=lambda: {"dataset": False, "playerheatmap": False})
    __dataset: pd.DataFrame = field(init=False,default_factory=pd.DataFrame)
    __conditions: list[Callable[[Any], bool]] = field(init=False,default_factory=list)
    __dataset_link: str = field(init=False,default_factory=str)
    def __post_init__(self, _conditions, _rewrite, _delete_colum, _dataset_link):
        self.__conditions = _conditions
        self.__rewrite = _rewrite
        self.__delete_colum = _delete_colum
        self.__dataset_link = _dataset_link

    @property
    def dataset_link(self) -> str:
        return self.__dataset_link

    @dataset_link.setter
    def dataset_link(self, new: str):
        self.__dataset_link = new
        for i in self.__actuell.keys():
            self.__actuell[i] = False

    @property
    def conditions(self) -> list[Callable[[Any], bool]] :
        return self.__conditions

    @conditions.setter
    def conditions(self, new_cond:  list[Callable[[Any], bool]] ):
        self.__conditions = new_cond
        for i in self.__actuell.keys():
            self.__actuell[i] = False

    @property
    def rewrite(self) -> list[Callable[[pd.DataFrame], pd.DataFrame]]:
        return self.__rewrite

    @rewrite.setter
    def rewrite(self, new: list[Callable[[pd.DataFrame], pd.DataFrame]]):
        self.__rewrite = new
        for i in self.__actuell.keys():
            self.__actuell[i] = False
    @property
    def dataset(self):
        """getter for __dataset"""
        if not self.__actuell["dataset"]:
            raise ValueError("self.dataset is not up to date")
        return self.__dataset

    @property
    def delete_colum(self)-> list[str]:
        return self.__delete_colum

    @delete_colum.setter
    def delete_colum(self, new : list[str]):
        self.__delete_colum = new
        for i in self.__actuell.keys():
            self.__actuell[i] = False

    def create_conditionData(self, indicator_2half ='matchPeriod' , des_2haf='2H',startx= "pos_orig_x" , starty= "pos_orig_y", endx= "pos_dest_x", endy = "pos_dest_y"):
        """Creats a dataset only with the row who fulfills the conditions"""
        self.__actuell["dataset"]=True
        data_begin = pd.read_csv(self.dataset_link)
        data_condition = data_begin[data_begin.apply(lambda row: all(cond(row) for cond in self.conditions), axis=1)]

        if self.flip_sec_half_coordinates:
            data_condition.loc[data_condition[indicator_2half] == des_2haf, [starty,startx,endy,endx] ]=100-data_condition.loc[data_condition[indicator_2half] == des_2haf, [starty, startx,endy,endx] ] 
        for rewrites in self.rewrite:

            data_condition = rewrites(data_condition)
        
        data_condition = data_condition.drop(columns=self.delete_colum)
        self.__dataset = data_condition

    def creat_file(self, pfad_name = ""):
        """Creates a CSV file with the actual __dataset"""
        if not self.__actuell["dataset"]:
            raise ValueError("self.dataset is not up to date (run self.create_conditionData())")
        if self.__dataset is not None:
            if pfad_name =="":
                pfad_name = self.dataset_name
            self.__dataset.to_csv(f"{pfad_name}.csv", sep=",", index=False)
            return None
        raise ValueError("self.dataset is None") 
    def fit(self ,coor_x:str, coor_y:str):
        """calls fit function of the self.playerheatmap"""
        if not self.__actuell["dataset"]:
            raise ValueError("self.dataset is not up to date (run self.create_conditionData())")
        self.__actuell["playerheatmap"] = True
        self.playerheatmap.fit(self.__dataset[coor_x],self.__dataset[coor_y])
    def shape(self) -> tuple[int, int]:
        """returns self.playerheatmap.shape"""

        return self.playerheatmap.shape() 

    def heatmap(self) -> plt.plot:
        """calls heatmap function of the self.playerheatmap"""
        if not self.__actuell["playerheatmap"]:
            raise ValueError("self.playerheatmap is not up to date ( run self.fit())")
        return self.playerheatmap.heatmap()
    
    def raw_counts(self) -> plt.plot:
        """calls raw_counts function of the self.playerheatmap"""
        if not self.__actuell["playerheatmap"]:
            raise ValueError("self.playerheatmap is not up to date ( run self.fit())")
        return self.playerheatmap.raw_counts()

    
if __name__ == "__main__":
    def testfunc(x):
        var = 0
    
        
        for i in  range(len(x["pos_orig_y"])):
            x.loc[i, "pos_orig_y"] = var
            #i["pos_orig_y"] = var
            var+=1
        print(x["pos_orig_y"])
        return x
    test = ConditionData(dataset_name="Example",_conditions=[lambda x: x["subEventName"] == "Simple pass"],_rewrite=[],_delete_colum = [],
                        _dataset_link="data\\example_data.csv", 
                        playerheatmap=player.PlayerHeatMap( action_name="Simple pass1",shape=(50,50)))
    test.create_conditionData()
    test2 = ConditionData(dataset_name="Example2",_conditions=[lambda x: x["subEventName"] == "Simple pass"],_rewrite=[testfunc],_delete_colum = ["tags"],
                        _dataset_link="data\\example_data2.csv",playerheatmap=player.PlayerHeatMap( action_name="Simple pass2",shape=(50,50)))
    test2.create_conditionData()
    test2.creat_file("data\\example_data2")
    plt.figure(1,figsize=(12, 6))
    test.fit("pos_orig_x","pos_orig_y")
    test.raw_counts()
    plt.figure(2,figsize=(12, 6))

    test2.fit("pos_orig_x","pos_orig_y")
    test2.raw_counts()
    plt.show()
