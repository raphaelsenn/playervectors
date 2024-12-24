# playervectors
Implementation of Player Vectors: Characterizing Soccer Players Playing Style from Match Event Streams

## Data

## About the Dataset
All the credit is to Luca Pappalardo and Emmanuele Massucco.

[https://www.kaggle.com/datasets/aleespinosa/soccer-match-event-dataset](https://www.kaggle.com/datasets/aleespinosa/soccer-match-event-dataset)

#### Load Action Data

```python
import pandas as pd
df = pd.read_csv('passes.csv')
x, y = list(df['x']), list(df['y'])
```
#### PlayerHeatMap: Visualize raw counts of action

```python
import matplotlib.pyplot as plt
from playervectors import PlayerHeatMap

# Create Action Component
action_pass = PlayerHeatMap(action_name='Pass')

# Fit action
action_pass.fit(x, y)

# Visualize Playing style for desired action as raw counts
plt.figure(figsize=(12, 6))
action.raw_counts()
plt.show()
```
![image](res/raw_counts_pass.png)

#### PlayerHeatMap: Visualize heatmap of action

```python
import matplotlib.pyplot as plt
from playervectors import PlayerHeatMap

# Create Action Component
action_pass = PlayerHeatMap(action_name='Pass')

# Fit action
action_pass.fit(x, y)

# Visualize Playing style for desired action as heatmap
plt.figure(figsize=(12, 6))
action_pass.heatmap()
plt.show()
```

![image](res/heatmap_pass.png)


## Citations

```bibtex
@article{ecmlpkdd2019,
  title     = {Player Vectors: Characterizing Soccer Players’
Playing Style from Match Event Streams},
  author    = {Tom Decroos, Jesse Davis},
  journal   = {ArXiv},
  year      = {2019},
}
```