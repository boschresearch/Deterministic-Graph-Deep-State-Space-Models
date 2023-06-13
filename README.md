# Deterministic-Graph-Deep-State-Space-Models

This is the companion code for the training method reported in 
[Cheap and Deterministic Inference for Deep State-Space Models of Interacting Dynamical Systems by Andreas Look et al., TMLR 2023](https://openreview.net/forum?id=dqgdBy4Uv5).

## Reproducing results
This repo contains a tutorial like jupyter notebook `examples/multimodal_toy.ipynb` to recreate the toy example from Fig. 2 in our paper as well 
as the training script `scripts/train.py` for the traffic forecasting experiments. After training, predictions can be visualized with `examples/visualize.ipynb`.

## Datasets 
The [rounD dataset](https://www.round-dataset.com/) can be processed with the file `data/preprocess_rounD.py`. We filter out non-moving objects as well as pedestrians. 

The [NGSIM dataset](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj) can be processed with code from [Nachiket Deo and Mohan M. Trivedi](https://github.com/nachiket92/conv-social-pooling/).

## Purpose of the project
This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be
maintained nor monitored in any way.

## License

This project is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in this project, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).
