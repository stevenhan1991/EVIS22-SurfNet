# SurfNet: Learning Surface Representations via Graph Convolutional Network
Pytorch implementation for SurfNet: Learning Surface Representations via Graph Convolutional Network.

## Prerequisites
- Linux
- CUDA >= 10.0
- Python >= 3.7
- Numpy
- Networkx
- Deep Graph Library
- Pytorch >= 1.0

## Data format

There are three types of files required for training SurfNet. The first file is a graphml, which stores the topological structure of a surface, i.e., adjacency matrix; the second file stores the information of each node on a surface, such as position and velocity; the third file stores all shortest paths between two nodes on a surface.

## Training models
```
cd Code 
```

- training
```
python3 main.py --mode 'train' --dataset '5cp'
```

- inference
```
python3 main.py --mode 'inf' --dataset '5cp'
```

## Citation 
```
@inproceedings{han2022surfnet,
  title={SurfNet: Learning surface representations via graph convolutional network},
  author={Han, Jun and Wang, Chaoli},
  booktitle={Computer Graphics Forum},
  volume={41},
  number={3},
  pages={109--120},
  year={2022}
}

```
## Acknowledgements
This research was supported in part by the U.S. National Science Foundation through grants IIS-1455886, CNS-1629914, DUE- 1833129, IIS-1955395, IIS-2101696, and OAC-2104158.
