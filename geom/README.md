# README

## Installation instructions

### Initial instructions

In order to run the program, you will first need to create conda environment with all necessary packages. Open your conda terminal (e.g. Anaconda Prompt) and move to the folder of this project (where README.md is). For example (on Windows):  
`cd C:/Users/Filip/knosp_project`

Then create the environment with following command (its name will be `knosp`):  
`conda create --name knosp python==3.10.6 --file requirements.txt`

When the environment is created, activate it:  
`conda activate knosp`

You need to do this setup only the first time you use this program. After creating the conda environment, you can proceed with the instructions bellow any other time you want to run the program.

### Running the program 

If you already have the `knosp` environment, you can run the program. Open conda terminal and move to the folder of this project (same as above):  
`cd C:/Users/Filip/knosp_project`

Activate the conda environment:  
`conda activate knosp`

And run the program:  
`python main.py "input/.../path" "output/.../path" "auto/manual"`

You need to specify the input and output paths, the mode of the segmentation mask (`auto` or `manual`), and then eventually opt on/off (1/0) visualizations, saving patients' annotations in subfolders. The last position is reserved for a threshold of how far the tumour needs to extend behind the critical line to increase the grade (float number, optional).

You need to have the inputs in the folder called `data/.../path`. The outputs will be saved to `output/.../path` folder. 

### Accuracy evaluation

If you want to compute the accuracy of the results, you will first need to run the program `statistics.py`:  
`python statistics.py "input/.../path" "output/.../path"`

## Project structure

- *data* - folder with the input data (numbered subfolders for each patient)
- *output* - folder to which outputs are saved
- *src* - folder with source code files
- *main.py* should be located on the same level as the dataset (folder *data*)
- *README.md* - this introductory file
- *requirements.txt* - package requirements for the conda environment (see above)
