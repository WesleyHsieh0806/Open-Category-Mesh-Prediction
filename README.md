# Open-Category-Mesh-Prediction
A Generalized Mesh Prediction Framework for Arbitrary Objects | CMU Learning for 3D Vision 16825 | Team 7 Project
   
   [**:pencil2: Citations**](#citations)


   * **Single-View Mesh Prediction for objects of open-categories**

---


  <h2> Table of Contents</h2>
  <ul>
    <li>
      <a href="#books-prepare-dataset">Prepare Dataset</a>
      <ul>
        <!-- <li><a href="#built-with">Built With</a></li> -->
      </ul>
    </li>
    <li>
      <a href="#running-usage---training">Usage</a>
    </li>
    <li>
      <a href="#citations">Citations</a>
    </li>
  </ul>



---

## :books: Prepare Dataset
   Please refer to [dataset.md](./docs/dataset.md) for further details.
   
   | Tasks | Datasets:point_down: | Input | Output |
   | - | - | - | - |
   | Single-View Mesh Prediction | [Objaverse](https://objaverse.allenai.org/)| RGB-Image | 3D Mesh|

## :running: Usage - Training
### Environment Setup
- Python 3.9
- Pytorch==1.13.1 
- torchvision==0.14.1
- CUDA 11.6
Use the provided [shell script](./environment_setup.sh) to setup the environment with only one command.
``` bash
bash environment_setup.sh
```

### Training
``` bash
python main.py \
  --data_root [/path/to/Objaverse/Root] \
  --split_data_path [/path/to/Objaverse/Split/Json] \
  --train True
```

## Citations
``` bash
Incoming
```
## Acknowledgement
Incoming
