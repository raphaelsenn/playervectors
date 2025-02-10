# playervectors
Implementation of [Player Vectors: Characterizing Soccer Players Playing Style from Match Event Streams](https://ecmlpkdd2019.org/downloads/paper/701.pdf) in python.

## Install

```bash
pip install playervectors
```

## Usage

## Expected Format for `df_events` (SPADL Format)

The `df_events` DataFrame used in `PlayerVectors.fit()` must follow the **SPADL format**, with the following required column names:

| Column Name  | Description |
|-------------|------------|
| **player_id**  | Unique identifier for the player. |
| **action_type** | Type of action or event (e.g., shot, pass, cross, dribble). |
| **x_start** | X-coordinate where the action starts. |
| **y_start** | Y-coordinate where the action starts. |
| **x_end** | X-coordinate where the action ends. |
| **y_end** | Y-coordinate where the action ends. |

If not, change the mapping in `PlayerVectors.fit(column_names=new_column_names)` such that:

```python
new_column_names = {'player_id': 'your_player_id',
                    'action_type': 'your_action_type',
                    'x_start': 'your_x_start',
                    'y_start': 'your_y_start',
                    'x_end': 'your_x_end',
                    'y_end': 'your_y_end'}
```

### Fitting PlayerVectors
Building **18**-component **PlayerVectors** with selected actions **shot**, **cross**, **dribble** and **pass** with respective components **4**, **4**, **5** and **5**.

```python
from playervectors import PlayerVectors


pvs = PlayerVectors(grid=(50, 50),
                    actions=['shot', 'cross', 'dribble', 'pass'],
                    components=[4, 4, 5, 5])

pvs.fit(df_events=df_events,
        minutes_played=minutes_played,
        player_names=player_names)
```

| Parameter  | Description |
|-------------|------------|
| **df_events**  | Event Stream Data in SPADL-Format. |
| **minutes_played** | A dictionary that maps each player_id to the total minutes they played across all events in df_events|
| **player_names** | Mapping player_id to player_name. |


### Plotting Principle Components

```python
import matplotlib.pyplot as plt

pvs.plot_principle_components()
plt.show()
```
![image](res/principle_components.png)

<p style="font-size: 12px; text-align: center;">
    <em>Output of: pvs.plot_principle_components()</em>
</p>


### Plotting Weight Distribution

```python
import matplotlib.pyplot as plt

pvs.plot_distribution()
plt.show()
```
![image](res/distribution_weights.png)

<p style="font-size: 12px; text-align: center;">
    <em>Output of: pvs.plot_distribution()</em>
</p>

### Plotting Weights of a Player

```python
import matplotlib.pyplot as plt

# wy_id of Kevin De Bruyne (Central midfielder)
pvs.plot_weights(player_id=38021)
plt.show()
```

![image](res/weights_kevin.png)

<p style="font-size: 12px; text-align: center;">
    <em>Output of: pvs.plot_weights(player_id=38021)</em>
</p>



## Building Player Vectors

### 1. Selecting Relevant Action Types
Let $k_t$ be the number of principal components chosen to compress heatmaps of action type $t$.

According to the paper, $k_t$ with $t \in$ {shot, cross, dribble, pass} with corresponding components {4, 4, 5, 5} is the minimal number of components needed to explain 70% of the variance in the heatmaps of action type $t$.


This parameter setting
was empirically found to work well because of the high variability of players
positions in their actions (see Challenge 1 in Section 2 in the paper).

Ignoring 30% of the variance allows to summarize a player’s playstyle only by his dominant regions
on the field rather than model every position on the field he ever occupied.


### 2. Constructing Heatmaps

#### 2.1 Counting
![image](res/counting.png)

<p style="font-size: 12px; text-align: center;">
    <em>Source: Tom Decroos and Jesse Davis, September 19th, 2019 ECMLPKDD</em>
</p>

#### 2.2 Normalizing
![image](res/counting_norm.png)

<p style="font-size: 12px; text-align: center;">
    <em>Source: Tom Decroos and Jesse Davis, September 19th, 2019 ECMLPKDD</em>
</p>

#### 3.3 Smoothing
![image](res/smoothing.png)

<p style="font-size: 12px; text-align: center;">
    <em>Source: Tom Decroos and Jesse Davis, September 19th, 2019 ECMLPKDD</em>
</p>


### 3. Compressing Heatmaps to Vectors

#### 3.1 Reshaping
![image](res/reshaping.png)

<p style="font-size: 12px; text-align: center;">
    <em>Source: Tom Decroos and Jesse Davis, September 19th, 2019 ECMLPKDD</em>
</p>


#### 3.2 Construct the matrix M

![image](res/matrix_m.png)

<p style="font-size: 12px; text-align: center;">
    <em>Source: Tom Decroos and Jesse Davis, September 19th, 2019 ECMLPKDD</em>
</p>


#### 3.3 Compress matrix M by applying non-negative matrix factorization (NMF)

![image](res/nmf.png)

<p style="font-size: 12px; text-align: center;">
    <em>Source: Tom Decroos and Jesse Davis, September 19th, 2019 ECMLPKDD</em>
</p>



### 4. Assembling Player Vectors
The player vector v of a player p is the concatenation of his compressed vectors
for the relevant action types.

## Detailed Algorithm
![image](res/algorithm.png)


## Use Repository with Data

#### 1. Download this [Dataset](https://www.kaggle.com/datasets/aleespinosa/soccer-match-event-dataset) on Kaggle

#### 2. Create a folder named event_streams in this Repository

```bash
mkdir event_streams
```

#### 3. Copy all .csv files from the Dataset in the folder event_streams



## About the Datasets

### Dataset 1
All the credit is to Luca Pappalardo and Emmanuele Massucco.

[https://www.kaggle.com/datasets/aleespinosa/soccer-match-event-dataset](https://www.kaggle.com/datasets/aleespinosa/soccer-match-event-dataset)

### Dataset 2
This dataset contains European football team stats.
Only teams of Premier League, Ligue 1, Bundesliga, Serie A and La Liga are listed.

[https://www.kaggle.com/datasets/vivovinco/football-analytics](https://www.kaggle.com/datasets/vivovinco/football-analytics)

## Citations

```bibtex
@article{ecmlpkdd2019,
  title     = {Player Vectors: Characterizing Soccer Players’
Playing Style from Match Event Streams},
  author    = {Tom Decroos, Jesse Davis},
  journal   = {ecmlpkdd2019},
  year      = {2019},
}
```