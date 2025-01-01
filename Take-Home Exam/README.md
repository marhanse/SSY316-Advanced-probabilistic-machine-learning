# Advanced Probabilistic Machine Learning - Take-Home Exam

## Overview
This repository contains the code and resources for the take-home exam of the SSY316 Advanced Probabilistic Machine Learning course.

Each solution is provided in a Jupyter notebook file named `PX.ipynb`, where `X` corresponds to the problem number.

## Prerequisites
- Python version 3.5 or higher
- Required Python packages (listed in `requirements.txt`)

## Installation
1. Install the required packages by running the following command:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
### Problem 1
...

### Problem 2
...

### Problem 4
For optimization using a specific algorithm, comment out the code related to the other algorithm to save computation time. For example, if you intend to run the genetic algorithm, comment out the simulated annealing method. 

To do so, modify the code as follows:
- Comment out the line:
  ```python
  planSA = sa.run(max_iterations=5000)
  ```
- Update the input argument of the `sim()` function to the corresponding plan, such as `planGA` for the genetic algorithm. Similarly, `planSA` refers to simulated annealing.

Ensure that only one optimization method is active at a time to streamline execution.
