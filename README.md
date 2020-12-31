# Download Dataset
A subset of the dataset can be downloaded with the following:

```
import minerl 
minerl.data.download('/your/local/path',experiment='MineRLObtainDiamondVectorObf-v0')
```

Alternatively, you can download the entire dataset (60 GBs!):
```
import minerl
minerl.data.download(directory="/your/local/path/")
```