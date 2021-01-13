# Competition
https://minerl.io/docs/

# Download Dataset
A subset of the dataset can be downloaded with the following:

```
import minerl 
minerl.data.download('~/MineRL/data', experiment='MineRLObtainDiamondVectorObf-v0')
Also check out 
MineRLTreechop-v0, MineRLNavigate-v0, 
```

Alternatively, you can download the entire dataset (60 GBs!):
```
import minerl
minerl.data.download(directory="~/MineRL/data")
```

Once downloaded, run the following command to view the dataset
```
export MINERL_DATA_ROOT='~/MineRL/data'
```

```
python3 -m minerl.viewer MineRLObtainDiamondDense-v0
```

# Running model

If you want to use offline RL in rllib, you first need to convert the dataset to a json format. This can be done with the ```preprocess/convert_to_json.py``` script


# TODO
Get a baseline with fully offline rainbow:
Use MOReL: https://arxiv.org/pdf/2005.05951.pdf
Use a transformer cnn: https://www.reddit.com/r/MachineLearning/comments/j4xmht/d_paper_explained_an_image_is_worth_16x16_words/