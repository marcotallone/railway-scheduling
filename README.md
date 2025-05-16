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
      </tr>
    </table>
</div>

<!-- TABLE OF CONTENTS -->
<div align="center">
  <table>
      <tr><td style="text-align: left;">
        <h2>Table of Contents&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</h2>
        <div style="display: inline-block; text-align: left;" align="left">
          <p>
            &nbsp;1. <a href="#author-info">Author Info</a><br>
            &nbsp;2. <a href="#about-the-project">About The Project</a><br>
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <a href="#quick-overview">Quick Overview</a><br>
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <a href="#built-with">Built With</a><br>
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <a href="#project-structure">Project Structure</a><br>
            &nbsp;3. <a href="#getting-started">Getting Started</a><br>
            &nbsp;4. <a href="#usage-examples">Usage Examples</a><br>
            &nbsp;5. <a href="#model-description">Model Description</a><br>
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <a href="#problem-description">Problem Description</a><br>
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <a href="#decision-variables">Decision Variables</a><br>
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <a href="#objective-and-constraints">Objective and Constraints</a><br>
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <a href="#simulated-annealing-meta-heuristic">Simulated Annealing Meta-Heuristic</a><br>
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <a href="#valid-inequalities">Valid Inequalities</a><br>
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- <a href="#dataset-generation">Dataset Generation</a><br>
            &nbsp;6. <a href="#results-and-scalability-analysis">Results and Scalability Analysis</a><br>
            &nbsp;7. <a href="#contributing">Contributing</a><br>
            &nbsp;8. <a href="#license">License</a><br>
            &nbsp;9. <a href="#references">References</a><br>
            10. <a href="#acknowledgments">Acknowledgments</a><br>
          </p>
        </div>
      </td></tr>
  </table>
</div>

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
The implemented models are developed in the [Python `railway` module](./src/railway/) and the relative job scheduling problems have been solved using the [`Gurobi` solver](https://www.gurobi.com/). All the methods implemented in the models have been largely documented and it's therefore possible to gain futher information using the Python `help()` function as shown below:

```python
import railway
help(railway.Railway)
```

Each model has been tested on small istances of the problem and a scalability analysis has also been performed to assess the performance of the models on larger instances as well. The datasets used for these tests have been randomly generated following the description provided in the original paper.\
Futher information about the models mathematical formulation and implementation, the scheduling problem istances and the results obtained can be found in the
[Model Description](#model-description) and in the [Results and Scalability Analysis](#results-and-scalability-analysis) sections below or in the [presentation](https://marcotallone.github.io/railway-scheduling/) provided in this repository.

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

- `gurobipy`, version `12.0.0`: the official `Gurobi` Python API, which requires to have an active `Gurobi` license and the solver installed on the machine. An [academic license](https://www.gurobi.com/academia/academic-program-and-licenses/) was used to develop the project.

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

from the root directory of the project. After that, the module can be imported in any Python script or notebook with the following import statement:

```python
from railway import Railway
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage Examples

The [`notebooks/`](./notebooks) folder contains a set of Jupiter Notebooks that show how to use the implemented models to solve the scheduling problem. In particular, the folder contains $3$ notebooks ([`model0.ipynb`](./notebooks/model0.ipynb), [`model1.ipynb`](./notebooks/model1.ipynb) and [`model2.ipynb`](./notebooks/model2.ipynb)) showing how to instantiate the $3$ models presented below using the implemented `Railway` class and its methods, while the [`model_formulation.ipynb`](./notebooks/model_formulation.ipynb) notebook provides a detailed description of the mathematical formulation of the models. The content of the latter is not used in practice but shows in details how each element of the mathematical formulation is implemented in practice in the final module.\
Furthermore, in the [`apps/`](./apps) folder contains the following Python scripts showing practical usage examples of the models:

- [`generate.py`](./apps/generate.py): a script that generates random istances of the scheduling problem for given parameters and saves them in the [`datasets/`](./datasets) folder as JSON files.
- [`test.py`](./apps/test.py): a script that tests the performance of all the models and compares them on the same istances of the scheduling problem.
- [`scalability.py`](./apps/scalability.py): a script that tests the scalability of the models on larger istances of the scheduling problem.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MODEL DESCRIPTION -->
## Model Description

In the following section a summary of the mathematical model presented by *Y.R. de Weert et al.* [<a href="#ref1">1</a>] is presented to better understand the scheduling problem addressed by the models implemented in this repository.

### Problem Description

In the railway maintainance scheduling problem addressed by the implemented models we consider a railway network over a finite discrete time horizon

$$
T = \{1, 2, \ldots, T_{end}\}
$$

with a set of nodes representing stations

$$
N = \{1, 2, \ldots, n\}
$$

and a set of arcs representing the direct railway lines connecting the stations

$$
\mathcal{A} = \{(i, j) \mid \forall i, j \in N, i < j\}
$$

where the arc $a = (i,j) \in \mathcal{A}$ represents the bidirectional connection between the $i^{th}$ and $j^{th}$ station. To each arc is associated a travel time by train $\omega^e_a \in \mathbb{R}$ and a travel time by alternative services $\omega^j_a \in \mathbb{R}$. In the case of this study, while the former is computed as the Euclidean distance between the stations (*randomly placed in a unitary circle*), the latter is modelled as the first one multiplied by some delay factor. To model this delay factor, the relation presented in the paper has been used:

$$
\omega^j_a = 1.35 \cdot \omega^e_a
$$

In normal conditions, the travel time by train is used, while in case of maintainance jobs on the arc the travel time by alternative services is used.\
Moreover, given any possible origin-desination pair $(o, d) \in N \times N \mid o \neq d$, we define the set $\Omega_{od}$ of average travel times by train from the given origin $o$ to the destination $d$. Each entry of this set is basically the sum of the travel times by train $\omega^e_a$ over the arcs of the shortest path connecting the origin and the destination.\
As anticipated, in this context we consider the set of maintainance jobs

$$
J = \{1, 2, \ldots, n_{\text{jobs}}\}
$$

that have to be scheduled on the arcs of the network. Each job $j \in J$ is characterized by a subset
$\mathcal{A}_j \subseteq \mathcal{A}$ of one or more arcs on which the maintainance has to be performed and by a processing time $\pi_j \in \mathbb{N}$ representing the duration of the job itself. Simmetrically, for each arc $a \in \mathcal{A}$, we also introduce the set

$$
J_a = \{j \in J \mid a \in \mathcal{A}_j\}
$$

of jobs that have to be scheduled on the arc $a$. Additionally, some arcs might require that a certain time interval $\tau_a \in \mathbb{N}$ has to pass between two consecutive jobs scheduled on the same arc.\
In some cases, it might be the case that some subset of arcs cannot be unavailable simultaneously. Hence, the set $C$ containing combinations of arcs on which jobs cannot be scheduled at the same time is introduced.\
For each time period $t \in T$, we then define the passenger deman $\phi_{odt}$ for each possible origin-destination pair, the share  passengers travelling in the peak moment in time $t$ from $o$ to $d$ as $\beta_{odt}$ and the limited capacity $\Lambda_{at}$ for alternative services substituting the train transportation in case a job is scheduled on the arc $a$ at time $t$.\
Trains that run on the arcs not subject to maintainance jobs are assumed to have an unlimited capacity, modelled by the variable $M \gg 1$. $M$ is set to a relatively big number and surely satisfies:

$$
M > \sum_{(o,d) \in N \times N} \sum_{t \in T} \phi_{odt}
$$

One final element to introduce in this model are event requests. It might happen that, at any time $t \in T$ an event might occur that increases the passenger demand in a subset of arcs of the network. This is modelled by the set $E$ consisting of tracks segments $s$, i.e. sets of consecutive arcs of different length, for each event happening at time $t$. The capacity of the service should always be sufficient during event requests to avoid overcrowding. This is always the case when trains are operational but it might not be the case when alternative services are used in case a job has been scheduled on the arcs of the event request. In the latter case the capacity is limited to

$$
\Lambda_{st} = \sum_{a \in s} \Lambda_{at},\quad \forall s \in E, \forall t \in T
$$

Finally, it's assumed that, when travelling from origin $o$ to destination $d$, passengers can consider up to $K \in \mathbb{N}$ possible alternative routes aiming for the one that minimizes the total travel time. Therefore, the set $R$ contains, for each origin-destination pair $(o,d) \in N \times N \mid o \neq d$, the $K$ shortest paths connecting the origin and the destination. These can be easily computed using [Yen's algorithm](https://en.wikipedia.org/wiki/Yen%27s_algorithm) once the topology of the network is known (implemented in the `YenKSP()` method of the `Railway` class).\
Further assumptions made for this model are listed below:

- there is no precedence relation between jobs, i.e. jobs can be scheduled in any order
- each job has equal urgency
- jobs cannot be interrupted once started
- all passengers travelling between the same origin-destination pair at any given time choose the same route, i.e. the shortest path connecting the origin and the destination
- outside event requests, the capacity of alternative services is always sufficient to avoid overcrowding

### Decision Variables

The decision variables of the model are the following:

- $y_{jt}$: binary variable that is equal to 1 if job $j$ starts at time $t$, 0 otherwise
- $x_{at}$: binary variable that is equal to 1 if arc $a$ is available (by train) at time $t$, 0 otherwise
- $h_{odtk}$: binary variable that is equal to 1 if route option $k$ is used when travelling from origin $o$ to destination $d$ at time $t$, 0 otherwise
- $w_{at}$: continuous variable representing the travel time traversing arc $a$ at time $t$
- $v_{odt}$: continuous variable representing the travel time from origin $o$ to destination $d$ at time $t$

### Objective and Constraints

The goal of the model is to **minimize the total passenger delay** in the network. The **delay** is defined as the **increase in the travel time compared to the average travel time** between an origin and a destination. According to this, the following objective function is defined:

$$
\underset{v}{\min}
  \sum_{(o,d) \in N \times N} \sum_{t \in T}
  \phi_{odt} (v_{odt} - \Omega_{odt})
  \tag{1}
$$

A **feasible solution** of this mathematical model **finds a schedule such that all the jobs are scheduled** within the time horizon $T_{end}$.\
Therefore a constraint must be set in order for a job to be started and finished within the time horizon:

$$
\sum_{t=1}^{T_{end} - \pi_j + 1} y_{jt} = 1, \quad \forall j \in J
\tag{2}
$$

While a job is processed, the subset $\mathcal{A}_j$ of arcs on which the job has to be performed must be unavailable for other jobs since they are occupied. This is modelled by the following constraint where, if $x_{at} = 1$ then the arc $a$ is available at time $t$ and no job could had been scheduled neither at time $t$ itself nor in the prior time interval $[t - \pi_j + 1, t]$ (respecting time horizon), but if the arc is not available then each job $j$ can only start once in the time period preceeding time $t$ (hence the sum of its $y$ variables is at most $1$):

$$
x_{at} + \sum_{t'=\max(1, t-\pi_j+1)}^{\min(T_{end}, t)} y_{jt'} \leq 1, \quad \forall a \in \mathcal{A}, \forall t \in T, \forall j \in J_a
\tag{3}
$$

(Notice that this constraint alone still does not guarantee that jobs cannot overlap, a further constraint is added for this purpose in $(7)$).\
Accordingly the travel time $w_{at}$ on an arc $a$ at time $t$ depends on the availability of the train services on that arc by the following constraint:

$$
w_{at} = \omega^e_a x_{at} + \omega^j_a (1 - x_{at}), \quad \forall a \in \mathcal{A}, \forall t \in T
\tag{4}
$$

Since variables $x_{at}$ are only bounded by $(3)$, an optional constraint is added to ensure the correct arc traversal time for free variables:

$$
\sum_{t \in T} x_{at} = T_{end} - \sum_{j \in J_a} \pi_j, \quad \forall a \in \mathcal{A}
\tag{5}
$$

In other words, the leftmost sum corresponds to the total time in which the arc $a$ is available, while the second term equals to the time horizon minus the total processing time of all jobs on the arc $a$. Logically, the two terms must be equal. Notice that cases for which the total sum of processing times in any arc (*rightmost sum*) is greater than the time horizon $T_{end}$ automatically correspond to infeasible solutions. Hence the second term is always positive or, at most, null.\
Due to the fact that there might be multiple combinations $c$ of arcs that cannot be unavailable simultaneously according to set $C$, the following constraint is added:

$$
\sum_{a \in c} (1 - x_{at}) \leq 1, \quad \forall c \in C, \forall t \in T
\tag{6}
$$

Since jobs on the same arc cannot overlap in time the following constriant is needed:

$$
\sum_{j \in J_a} \sum_{t'=\max(1, t-\pi_j-\tau_a)}^{T_{end}} y_{jt'} \leq 1, \quad \forall t \in T, \forall a \in \mathcal{A}
\tag{7}
$$

For a track segment $s$ included in an event request the passenger flow in the peak moment of the considered time period must be less than the capacity of the alternative services:

$$
\sum_{a \in s} \sum_{(o,d) \in N \times N} \sum_{\substack{i = 1 \\ a \in R_{odi}}}^K h_{odti} \beta_{odt} \phi_{odt} \leq \Lambda_{st} + M \sum_{a \in s} x_{at}, \quad \forall s \in E_t, \forall t \in T
\tag{8}
$$

Concerning alternative routes, we must ensure that passenger flow is served by at least one (and no more than one) of the $K$ possible routes:

$$
\sum_{i = 1}^K h_{odti} = 1, \quad \forall (o,d) \in N \times N, \forall t \in T
\tag{9}
$$

Accordingly, the following two constraints provide a lower and an upper bound for the travel time from origin $o$ to destination $d$ at time $t$:

$$
v_{odt} \geq \sum_{a \in R_{odi}} w_{at} - M (1 - h_{odti}), \quad \forall i \in \{1,\dots,K\}, \forall (o,d) \in N \times N, \forall t \in T
\tag{10}
$$

$$
v_{odt} \leq \sum_{a \in R_{odi}} w_{at}
\tag{11}
$$

In addition to these constraints, since in some cases variables could take values that are never included in any feasible solution, further constraints can be added to reduce the search space and the computational cost.\
In particular, the following two constraints express the availability of arcs that are never included in any job at any given time:

$$
x_{at} = 1, \quad \forall a \in \{a \in \mathcal{A} \mid J_a = \emptyset\}, \forall t \in T
\tag{12}
$$

$$
w_{at} = \omega^e_a, \quad \forall a \in \{a \in \mathcal{A} \mid J_a = \emptyset\}, \forall t \in T
\tag{13}
$$

An additional constraint deals with event requests. Consider the $K$ possible routes to travel from an origin $o$ to a destination $d$ at time $t$. Imagine that all the considered $K$ routes pass through the same arc $a$ at some point of the path and that such arc is included in an event request at time $t$. In this case, if $\beta_{odt} \phi_{odt}$ is greater than the capacity of the alternative services $\Lambda_{at}$ for such arc, we must ensure the availability of the arc to avoid an infeasible solution. This is embodied by the following constraint:

$$
x_{at} = 1, \quad \forall (a,t) \in \{
  \substack{(a,t) \in \mathcal{A} \times T \mid \exists s \in E_t \text{ s.t. } a \in s \\
  \land\ \exists (o,d) \in N \times N \text{ s.t. } a \in R_{odi} \forall i \in \{1,\dots,K\} \\ \land\ \beta_{odt} \phi_{odt} > \Lambda_{st}}
\}
\tag{14}
$$

Finally, given the $K$ routes from $o$ to $d$ at time $t$, if route option $\tilde{k} \in \{1, \dots, K\}$ is **never used**, we can set the associated variable $h_{odt\tilde{k}} = 0$. Hence the following constraint can be added to the model:

$$
h_{odti} = 0, 
\quad \forall (o, d, i) \in
\{(o, d, i) \in N \times N \times [K] \mid
\min_{i \in [K]} \Omega_{odi} < \max_{j \neq i \in [K]} \Omega_{odj
}\}
\tag{15}
$$

All of these constraints and the cited objective function have been implemented in a Gurobi `model` object in the `Railway` class. Such model can be optimized using the `optimize()` method of the class, which returns the optimal solution of the scheduling problem.

### Simulated Annealing Meta-Heuristic

Meta-heuristics can improve the computational times of MILP problems by providing a good initial solution guess in a reasonable amount of time. In this case, the [**simulated annealing**](https://en.wikipedia.org/wiki/Simulated_annealing) meta-heuristic has been used to find a good initial solution guess for the MILP model. This algorithm is applied to find an initial guess for the set of **starting times** $S = \{s_j \mid \forall j \in J\}$ of all jobs.\
The algorithm, implemented in the `simulated_annealing()` method of the `Railway` class as described in the paper [[1](#ref1)], can be summarized in the following pseudo-code:

```javascript
function simulated_annealing(
  S0: initial guess, 
  T0: initial temperature,
  c:  cooling factor,
  L:  iterations per temperature,
  STOP_CRITERION: stopping criterion
):
  S <- S0
  T <- T0
  while STOP_CRITERION do:
    for l in {1,...,L} do:
      Snew <- get_neighbor(S)
      Δf <- f(S) - f(Snew)
      if Δf >= 0:
        S <- Snew
      else:
        S <- Snew with probability exp(-Δf/T)
      end
    end
    T <- c * T
  end
  return S
```

In this algorithm, a new neighbor solution $S_{new}$ to a given one $S$ is generated every time by shifting the starting time of only one of the jobs in the solution and checking that such shift does not violate any constraint returning always a feasible solution.\
Such algorithm hence returns a good initial guess for the starting times of all jobs, but it's possible to get the values of the associated decision variables from these in the following way:

- $y$ can be obtained by setting the values $y_{js_j} = 1$ $\forall j \in J$ where $s_j \in S$ and all the other $y_{jt}$ to 0 for the remaining times $t \in T$
- $x$ is unavailable, hence $x_{at} = 0$ only for $a \in \mathcal{A}$ and for $t \in T$ for which $t \in [s_j, s_j + \pi_j], \forall j \in J_a$
- $w$ values can be easily obtained from constraint $(4)$ once $x$ values are known
- $h$ and $v$ are computed once the other $3$ decision variables are set. In this case, for each origin-destination pair we can compute the shortest route among the $K$ given ones and set the $h$ values accordingly, while the $v$ values are simply the sum of the travel times over the arcs of the chosen shortest route

The described procedure is the one implemented in the `get_vars_from_times(S)` method of the `Railway` class. Simmmetrically, it's very easy to obtain $S$ from the decision variables if needed as implemented in the `get_times_from_vars(y)` method.

### Valid Inequalities

To further reduce the search space and the computational cost of the MILP model, it's possible to set the following valid inequalities via the `set_valid_inequalities()` method implemented. These inequalities come from the ``Single Machine Scheduling'' problem:

- **Sousa and Wolsey** inequality:

  $$
  \sum_{t' \in Q_j} y_{jt'} + \sum_{\substack{j' \in J_a \\ j' \neq j}} \sum_{t' \in Q'_j} y_{j't'} \leq 1, \quad \forall a \in \mathcal{A}, \forall j \in J_a, \forall t \in T, \forall \Delta \in \{2,\dots,\Delta_{max}\}
  \tag{B1}
  $$

  with:
  - $Q_j = [t - \pi_j +1, t+ \Delta -1] \cap T$
  - $Q'_j = [t - \pi_j + \Delta, t] \cap T$
  - $\Delta_{max} = \underset{\substack{j' \in J_a \\ j' \neq j}}{\max} (\pi_{j'} + \tau_a)$
    *(i.e. maximum total processing time for the other jobs on the same arc as job $j$)*

- non overlapping jobs inequality:

  $$
  y_{jt} + y_{j't'} \leq 1, \quad \forall a \in \mathcal{A}, \forall j, j' \in J_a, \forall t, t' \in T \mid j \neq j' \land t' \in (t - \pi_{j'} - \tau_a, t + \pi_j + \tau_a) \cap T
  \tag{B2}
  $$

### Dataset Generation

Following the example provided in the [`generate.py`](./apps/generate.py) script, it's possible to generate random istances of the scheduling problem for given parameters.\
In particular, the implemented methods allows to generate an instance of the problem by selecting the desired total number of stations $n$, the number of jobs $n_{jobs}$, the time interval $T_{end}$, the number of alternative routes $K$ and the number of passengers in the network. With these values set, the remaining parameters of the model can be generated as explained in the following list:

- **Network topology**: i.e. Euclidean coordinates of the $n$ stations, with the associated travel times. These are randomly generated in a unitary circle (following the procedure presented in the paper) at model instantiation.

- $\pi_j$: processing times for each job $j$. Randomly generated in a desired range of integer values.

- $A_j$: arcs on which each job $j$ has to be performed. For each job, a random starting station is drawn from the set $N$. Then, the "length" of each job, meaning the number of arcs that the job is going to be performed on, is randomly generated in a desired range of integer values that can be given in input. Having set these, a random arc from the ones incident to the starting station is selected and added to a list of arcs for the current job and the starting station is updated to the one at the end of the selected arc. This is repeated until the desired number of arcs is reached. This process also filters out the arcs that are already included in the job list, so that no arc is selected twice for the same job. For further details, consult the `__generate_Aj()` method of the `Railway` class.

- $\tau_a$: minimum time interval between two consecutive jobs on the same arc $a$. Also in this case the user can set a minimum and maximum integer value for this parameter and the values are randomly generated in this range.

- $\phi$: passenger demand for each origin-destination pair $(o,d)$ and each time $t$. A minimum and maximum passenger demand percentages (%) can be set in the range $[0, 1]$. The integer values for the parameter are then randomly sampled between the minimum percentage mutiplied by the total number of passengers and the maximum percentage multiplied by the total number of passengers. (Total number of passengers is set at instantiation by the user).
  
- $\beta$: share of passengers travelling in the peak moment in time $t$ from $o$ to $d$. Uniformly randomly distributed in a set range of values that can be given in input, between $0$ and $1$.

- $\Lambda$: limited capacity for alternative services. A minimum and maximum capacity percentages (%) can be given in input in the range $[0, 1]$. The integer values for the parameter are then randomly sampled in the range having the following integer extremes:
    $$
    \Lambda_{min} = \lfloor \frac{\text{min percentage} \cdot \text{number of passengers}}{n} \rfloor
    $$
    $$
    \Lambda_{max} = \lfloor \frac{\text{max percentage} \cdot \text{number of passengers}}{n} \rfloor
    $$

- $E$: event requests. The user can set a maximum number of event requests at each time $t$ (minimum is always $0$) and decide minimum and maximum length of the event request (i.e. how many arcs it is going to cover). The event requests are randomly generated similarly to how the set $A_j$ is (explained above). For further details consult the method `__generate_E()` of the `Railway` class.

- $C$: combinations of arcs on which jobs cannot be scheduled at the same time. This can be manually selected by the user at instantiation. In general, the set $C$ has been left empty in the generated datasets since it easily leads to infeasible problems given the considerable restriction that it imposes on the scheduling of jobs. However, it might be manually defined as a list of tuples, where tuples contain the arcs that cannot be unavailable at the same time.

- $R$: set possible alternative routes for each origin-destination pair. Generated using either [Yen's algorithm](https://en.wikipedia.org/wiki/Yen%27s_algorithm) or a random path generation algorithm to compute the $K$ shortest paths after the user has set the maximum number of alternative routes $K$ at instantiation.

>[!NOTE]
> When a problem is generated, there is of course no guarantee that it will be feasible. For this reason, the final part of the `generate.py` script tries to solve the problem and solve the dataset only if a finite solution is found within a certain time limit.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- RESULTS -->
## Results and Scalability Analysis

<!-- TODO: complete description after analysis of the data -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

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
<!-- <a id="ref2"></a>
[2] Natashia Boland, Thomas Kalinowski, Hamish Waterer, Lanbo Zheng, *"Scheduling arc maintenance jobs in a network to maximize total flow over time"*, Discrete Applied Mathematics, Volume 163, Part 1, 2014, Pages 34-52, ISSN 0166-218X, [https://doi.org/10.1016/j.dam.2012.05.027](https://www.sciencedirect.com/science/article/pii/S0166218X12002338) -->

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
