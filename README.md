<img
align="left" width="200" height="120" 
src="./docs/icon.png" alt="Temporal Mapper Logo">
## Temporal Mapper

### V.0.4.0 - August 19 '24
-----------------------------------------------
This is a library for using the Mapper for temporal topic modelling.
Though things broadly work now, the edge cases have not been throughly 
tested.

Direct questions to Kaleb D. Ruscitti: kaleb.ruscitti at uwaterloo.ca .

Complete documentation is under construction on [Read The Docs](
https://temporal-mapper.readthedocs.io/en/latest/).

### Example:
#### arXiv Papers 
From the arXiv API, we can retrieve ~500,000 article titles and abstracts,
use `SBERT` to embed them, and then UMAP to reduce to 2D.

Using [DataMapPlot](https://github.com/tutteinstitute/datamapplot) and
[TopicNaming](https://github.com/tutteinstitute/topicnaming) we can
produce a static plot of this data:

![A DataMapPlot of ArXiV papers](./doc/arxiv_static.png 
"A DataMapPlot of ArXiV Papers")

Now, using this repository we can additionally analyse the temporal
information. Using the Mapper algorithm with time as our lens
function, we create a *temporal graph* of the topics (clusters)
through time. The code includes two types of plots to visualize this
graph:

Centroid Plot             |  Temporal-Semantic Plot
:-------------------------:|:-------------------------:
![](./docs/arxiv_centroids.png)  |  ![](./docs/arxiv_time.png)

### Installation
Clone the repo and install: 
`git clone https://github.com/TutteInstitute/temporal-mapper.git`
`cd temporal-mapper && pip install .`

### Usage
The file `doc/DemoNotebook.ipynb` is a start-to-finish
example of how to generate a Sankey diagram with this package.

### Parameters
For a complete listing of the parameters, check the
[repo's GitHub
wiki](https://github.com/TutteInstitute/temporal-mapper/wiki/API-Reference).
However, the most impactful choices are:

#### HDBSCAN parameters

`HDBSCAN(min_cluster_size=n)` This is the usual HDBSCAN parameter, but
now that the points are weighted, and the weights are strictly <= 1,
you generally want to set this a bit lower than you might usually do.

#### `tm.TemporalMapper()` parameters
Mapper works by clustering
inside time slices, and these time slices are determined by two
parameters, `checkpoints` and `overlap`. The checkpoints define the
center of the bins, and `overlap`, which should lie in `(0,1)`,
defines how much the bins will intersect eachother.

##### Checkpoints
You can either pass tm.TemporalMapper() a list of
checkpoints;  `checkpoints = arrayLike` or you can use the
`N_checkpoints = int` and `slice_method = str` parameters to have it
generate checkpoints for you.

`slice-method` takes either 'time' or 'data'. The time option
generates checkpoints evenly spaced in time, and the data option
generates checkpoints evenly spaced in the number of data points.

##### Overlap
The default value of `overlap=0.5` should work in most
cases, but if you find your graph is highly disconnected you can
increase the overlap.


##### Neighbours
Passing `neighbours = k` for some positive integer
`k` determines the number of nearest neighbours used to compute the
temporal density of the data. If you have a lot of data, you should
increase this parameter as much as your computational constraints will
allow.

#### Temporal kernel parameters
The temporal kernel is used to give
the points weight in time. You can pass a kernel function to
tm.TemporalGraph `kernel=myFunc`. The default is
`temporal_mapper.weighted_clusters.gaussian` which is a Gaussian
kernel. If your kernel function takes parameters, you can pass
`kernel_params = (param1, param2, ...)`

The parameter `rate_sensitivity` can be any number >=0, or -1. This
controls how sensitive the temporal kernel is to changes in the
temporal density of your data. This is an exponent factor; at the
default setting (= 1.) points with double the temporal density will
have a kernel that is half as wide. At sensitivity 2, double density
gives 1/4 as wide, and so on. The option -1 sets the scale to be
logarithmic; 10x as dense = 1/2 as wide.

If you want to recover original (non-fuzzy) mapper, you can pass
`kernel = temporal_mapper.weighted_clusters.square` and
`rate_sensitivity = 0`.
