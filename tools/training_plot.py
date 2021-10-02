import pandas as pd
import plotly.express as px

df = pd.read_csv('./log/training_log.csv')
df.head()
y1 = "training_loss"
y2 = "val_loss"
fig = px.line(df, x = 'epoch', y = y1 , title='loss' ,color='training_loss')
fig.show()