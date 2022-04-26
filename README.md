# tSPM-Probabilistic-Scanpaths
Repository for the work "[A Probabilistic Time-Evolving Approach to Scanpath Prediction](https://arxiv.org/abs/2204.09404)"

![Teaser](https://github.com/DaniMS-ZGZ/tSPM-Probabilistic-Scanpaths/blob/main/img/teaser.jpg)

## Requeriments
You can install an environment with all required dependencies using the `environment.yml` file in Anaconda.

## Inference
The current version of the repository includes a basic, yet functional version to generate scanpaths from a single image using our model.

### Usage
There is currently one mode of usage for this code:
```
python main.py --mode inference 
```

## Inference parameters
```
image_path: The path from the image from which to generate scanpaths.
path_to_save: The path to save the generated scanpaths.
n: The number of scanpaths to generate (default = 20).
th: The probabilistic threshold (default = 0.5, see the paper for details).
```

This will read an image from `image_path = "data/test.jpg"` and generate a set of`n` scanpaths that will be saved in `path_to_save = "test/"`.

## Training the model
This option will be available soon.
