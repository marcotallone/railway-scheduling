<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
[![Gmail][gmail-shield]][gmail-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/marcotallone/railway-scheduling">
    <img src="images/railway.png" alt="Logo" width="150" height="150">
  </a>

<h2 align="center">Scheduling of Railway Maintainance Projects</h2>
<h4 align="center">Mathematical Optimization Course Exam Project</h4>
<h4 align="center">SDIC Master Degree, University of Trieste (UniTS)</h4>
<h4 align="center">2024-2025</h4>

  <p align="center">
    Improving the scheduling of railway maintainance projects by minimizing passengers delays subject to event requests of railway operators through mixed integer linear programming optimisation.
    <br />
    <br />
    <table>
      <tr>
        <td><a
href="https://marcotallone.github.io/railway-scheduling/"><strong>Presentation</strong></a></td>
        <!-- <td><a href="./presentation/presentation.pdf"><strong>PDF Presentation</strong></a></td> -->
        <td><a href="https://github.com/marcotallone/railway-scheduling/issues"><strong>Report bug</strong></a></td>
        <td><a href="https://github.com/marcotallone/railway-scheduling/issues"><strong>Request Feature</strong></a></td>
    </table>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <!-- <summary><h3 style="color: #4078c0;">📑 Table of Contents</h3></summary> -->
  <summary>📑 Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#quick-overview">Quick Overview</a></li>
        <li><a href="#built-with">Built With</a></li>
        <li><a href="#project-structure">Project Structure</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#references">References</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- AUTHOR INFO-->
## Author Info

| Name | Surname | Student ID | UniTS mail | Google mail | Master |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Marco | Tallone | SM3600002 | <marco.tallone@studenti.units.it> | <marcotallone85@gmail.com> | **SDIC** |


>[!WARNING] 
>**Copyright Notice**:\
> This project is based on the work published in the referenced paper by *Y.R. de Weert et al.* [<a href="#ref1">1</a>] and aims to reproduce the results and implement the models originally presented by the authors in the context of a university exam project.\
> The main **contribution** proposed in this repository is **limited to the personal implementation of the models** and the **development of the new datasets to assess their performance** *(which might slightly differ from the ones proposed by the original authors)*.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ABOUT THE PROJECT -->
## About The Project

### Quick Overview

This project implements the mixed integer linear programming (MILP) models proposed by *Y.R. de Weert et al.* [<a href="#ref1">1</a>] that minimizes passengers delays while scheduling maintainance projects in a railway network. In summary, the main problem addressed by these models is to find a suitable schedule for maintainance jobs that have to be performed on the arcs of a railway network, in order to minimize the passengers delays while respecting the event requests of railway operators within a finite time horizon. Moreover, such models includes capacity constraints for alternative services in event request areas during the execution of the maintenance projects.\
Further details about the model and the problem can be found below, or in the referenced paper.\
The implemented models are developed in the [Python `railway` module](./src/railway/) and the relative job scheduling problems have been solved using the [`Gurobi` solver](https://www.gurobi.com/). All the methods implemented in the models have been largely documented and it's therefore possible to gain futher information using the Python `help()` function as shown below:

```python
import railway
help(railway.Railway)
```

Each model has been tested on small istances of the problem and a scalability analysis has also been performed to assess the performance of the models on larger instances as well. The datasets used for these tests have been randomly generated following the description provided in the original paper.\
Futher information about the models mathematical formulation and implementation, the scheduling problems istances and the results obtained can be found below after the [Usage Examples](#usage-examples) section or in the [presentation](https://marcotallone.github.io/railway-scheduling/) provided in the repository.

### Project Structure

The project is structured as follows:

```bash
# TODO: Add project structure
```
  
### Built With

![Gurobi](https://img.shields.io/badge/Gurobi-ec3826?style=for-the-badge&logo=gurobi&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Conda](https://img.shields.io/badge/Conda-44A833?style=for-the-badge&logo=anaconda&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Requirements

The project is developed in `Python` and uses the [`Gurobi` solver](https://www.gurobi.com/) for the optimisation problems, hence the installation of the following library is needed:

- `gurobipy`, version `12.0.0 build v12.0.0rc1`: the official `Gurobi` Python API, which requires to have an active `Gurobi` license and the solver installed on the machine. An [academic license](https://www.gurobi.com/academia/academic-program-and-licenses/) was used to develop the project.

The following **optional** Python libraries are needed in order to run the `Jupiter Notebooks` in the [notebooks/](./notebooks) folder:

- `tqdm`, version `4.67.0`
- `holoviews`, version `1.20.0`
- `hvplot`, version `0..11.1`
- `bokeh`, version `3.6.1`
- `jupiter_bokeh`

All the necessary libraries can be easily installed using the `pip` package manager.\
Additionally a [conda environment `yaml` file](./mathopt-conda.yaml) containing all the necessary libraries for the project is provided in the root folder. To create the environment you need to have installed a working `conda` version and then create the environment with the following command:

```bash
conda env create -f mathopt-conda.yaml
```

After the environment has been created you can activate it with:

```bash
conda activate mathopt
```

### Installation

The code presented in this repository develops the model proposed by *Y.R. de Weert et al.* [<a href="#ref1">1</a>] as a Python module that can easily be imported and used in common Python scripts or Jupiter Notebooks. Therefore, in order to correctly use the provided implementation, the [`railway` module](./src/railway/) must be added to the Python path.\
This can be done manually or by taking advantage of the provided [`setup.py`](./setup.py) file in the root directory. In the latter case, the module can be installed in editable mode, with the following command:

```bash
pip install -e .
```

from the root directory of the project. After that, the module can be imported in any Python script or notebook with the following command:

```python
import railway
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage Examples

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Problem Description

## Mathematical Formulation: Objective and Constraints

## Simulated Annealing Meta-Heuristic

## Valid Inequalities

## Dataset Generation

## Results

## Conclusions

<!-- CONTRIBUTING -->
## Contributing

<!-- Although this repository started as a simple university exam project, if you have a suggestion that would make this better or you attempted to implement one of the above mentioned improvements and want to share it with us, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement" or "extension". -->

The goal of this repository was to implement and reproduce the results of the models presented in paper by *Y.R. de Weert et al.* [<a href="#ref1">1</a>] in the context of a university exam project. However, if you have a suggestion that would make this better or extend its functionalities and want to share it with me, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement" or "extension".\
Suggested contribution procedure:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- REFERENCES -->
## References

<!-- Y.R. de Weert, K. Gkiotsalitis, E.C. van Berkum,
Improving the scheduling of railway maintenance projects by minimizing passenger delays subject to event requests of railway operators,
Computers & Operations Research,
Volume 165,
2024,
106580,
ISSN 0305-0548,
https://doi.org/10.1016/j.cor.2024.106580.
(https://www.sciencedirect.com/science/article/pii/S0305054824000522)
Abstract: In the Netherlands, it is expected that passenger activities on railway networks will double by 2050. To manage the passenger demand, railway capacity planning needs to be adapted. One fundamental aspect of the railway capacity planning is the scheduling of large maintenance projects. These maintenance requests should not be scheduled during major events to avoid the disruption of service. To do so, passenger operators can submit event request, i.e., a time period and location in which no maintenance project should be scheduled. Currently, these event requests are considered as hard constraints and the flexibility in maintenance scheduling decreases resulting in conflicts. In this study, the focus is on scheduling maintenance projects to minimize passenger delays while relaxing the hard constraints of event requests. This problem is addressed by introducing a Mixed Integer Linear Program (MILP) that minimizes passenger delays while scheduling maintenance projects, which includes capacity constraints for alternative services in event request areas during maintenance projects. The computational complexity of the proposed model is reduced by adding valid inequalities from the Single Machine Scheduling problem and using a simulated annealing meta-heuristic to find a favorable initial solution guess. Then, the MILP is solved to global optimality with Branch-and-Bound. A case study on the Dutch railway network shows improvements when event requests are not considered as hard constraints and an increase in the flexibility to schedule maintenance projects. This allows decision makers to choose from a set of optimal maintenance schedules with different characteristics.
Keywords: Railway; Maintenance scheduling; Events; Passenger hindrance -->
<a id="ref1"></a>
[1] Y.R. de Weert, K. Gkiotsalitis, E.C. van Berkum, *"Improving the scheduling of railway maintenance projects by minimizing passenger delays subject to event requests of railway operators"*, Computers & Operations Research, Volume 165, 2024, 106580, ISSN 0305-0548, [https://doi.org/10.1016/j.cor.2024.106580](https://www.sciencedirect.com/science/article/pii/S0305054824000522)

<!-- Natashia Boland, Thomas Kalinowski, Hamish Waterer, Lanbo Zheng,
Scheduling arc maintenance jobs in a network to maximize total flow over time,
Discrete Applied Mathematics,
Volume 163, Part 1,
2014,
Pages 34-52,
ISSN 0166-218X,
https://doi.org/10.1016/j.dam.2012.05.027.
(https://www.sciencedirect.com/science/article/pii/S0166218X12002338)
Abstract: We consider the problem of scheduling a set of maintenance jobs on the arcs of a network so that the total flow over the planning time horizon is maximized. A maintenance job causes an arc outage for its duration, potentially reducing the capacity of the network. The problem can be expected to have applications across a range of network infrastructures critical to modern life. For example, utilities such as water, sewerage and electricity all flow over networks. Products are manufactured and transported via supply chain networks. Such networks need regular, planned maintenance in order to continue to function. However the coordinated timing of maintenance jobs can have a major impact on the network capacity lost due to maintenance. Here we describe the background to the problem, define it, prove it is strongly NP-hard, and derive four local search-based heuristic methods. These methods integrate exact maximum flow solutions within a local search framework. The availability of both primal and dual solvers, and dual information from the maximum flow solver, is exploited to gain efficiency in the algorithms. The performance of the heuristics is evaluated on both randomly generated instances, and on instances derived from real-world data. These are compared with a state-of-the-art integer programming solver.
Keywords: Network flows; Scheduling; Maintenance planning; Local search; Hybrid algorithms -->
<a id="ref2"></a>
[2] Natashia Boland, Thomas Kalinowski, Hamish Waterer, Lanbo Zheng, *"Scheduling arc maintenance jobs in a network to maximize total flow over time"*, Discrete Applied Mathematics, Volume 163, Part 1, 2014, Pages 34-52, ISSN 0166-218X, [https://doi.org/10.1016/j.dam.2012.05.027](https://www.sciencedirect.com/science/article/pii/S0166218X12002338)




<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

- [Mathematical Optimisation course material (UniTS, Spring 2024)](https://sites.units.it/castelli/didattica/?file=mathopt.html) (*access restricted to UniTS students and staff*)
- [Best-README-Template](https://github.com/othneildrew/Best-README-Template?tab=readme-ov-file): for the README template
- [Flaticon](https://www.flaticon.com/free-icons/railway): for the icons used in the README

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[forks-shield]: https://img.shields.io/github/forks/marcotallone/railway-scheduling.svg?style=for-the-badge
[forks-url]: https://github.com/marcotallone/railway-scheduling/network/members
[stars-shield]: https://img.shields.io/github/stars/marcotallone/railway-scheduling.svg?style=for-the-badge
[stars-url]: https://github.com/marcotallone/railway-scheduling/stargazers
[issues-shield]: https://img.shields.io/github/issues/marcotallone/railway-scheduling.svg?style=for-the-badge
[issues-url]: https://github.com/marcotallone/railway-scheduling/issues
[license-shield]: https://img.shields.io/github/license/marcotallone/railway-scheduling.svg?style=for-the-badge
[license-url]: https://github.com/marcotallone/railway-scheduling/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-blue?style=for-the-badge&logo=linkedin&logoColor=white&colorB=0077B5
[linkedin-url]: https://linkedin.com/in/marco-tallone-40312425b
[gmail-shield]: https://img.shields.io/badge/-Gmail-red?style=for-the-badge&logo=gmail&logoColor=white&colorB=red
[gmail-url]: mailto:marcotallone85@gmail.com
