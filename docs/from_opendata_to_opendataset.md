# From OpenData to OpenDataSet


Context:
-------

In supervised learning, you can't expect to obtain a good trained model from inacurate labels: Garbage In, Garbage Out. Data preparation, however, can be tedious if you don't use efficient tools. 

Even though OpenData datasets are widely available, those reliable enough to be used verbatim to train decents models are still scarse. Even with state of art model training algorithms, best results are only achieved by people who can afford to tag manually, with pixel accuracy, their own datasets.

So how can we load and qualify OpenData sets in order to create our own training samples ? That's what this tutorial is about !




Retrieve OpenData:
------------------

We use OpenData from <a href="https://rdata-grandlyon.readthedocs.io/en/latest/">Grand Lyon metropole</a> because it offers recent imagery and several vector layers throught standard Web Services.



First step is to set the spatial extent and the <a href="https://wiki.openstreetmap.org/wiki/Zoom_levels">zoom level</a>:

```bash
rsp cover --bbox 4.795,45.628,4.935,45.853 --zoom 18 ds/cover
```

Not to forget to create an ad hoc RoboSat.pink config file:
```bash
echo '
[[channels]]
  name   = "images"
  bands = [1, 2, 3]

[[classes]]
  title = "Building"
  color = "deeppink"

[model]
  nn = "Albunet"
  encoder = "Resnet50"
  loss = "Lovasz"
  loader = "SemSegTiles"
  da = "Strong"
  bs = 4
  lr = 0.000025
  metrics = ["iou"]
' > ds/config.toml
```

Download imagery using <a href="https://www.opengeospatial.org/standards/wms">WMS</a>:

```bash
rsp download --type WMS 'https://download.data.grandlyon.com/wms/grandlyon?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&LAYERS=Ortho2015_vue_ensemble_16cm_CC46&WIDTH=512&HEIGHT=512&CRS=EPSG:3857&BBOX={xmin},{ymin},{xmax},{ymax}&FORMAT=image/jpeg' --rate 20 ds/cover ds/images
```

NOTA:
- Retina resolution of 512px is preferred to a regular 256px, because it improves the training accuracy. 
- Roofprint choice is important here, as we use aerial imagery to retrieve patterns. If we used the building's footprints instead, the training accuracy would be poorer.


<a href="http://www.datapink.tools/rsp/opendata_to_opendataset/images/"><img src="img/from_opendata_to_opendataset/images.png" /></a>


Then to download buildings vector roofprints with <a href="https://www.opengeospatial.org/standards/wfs">WFS</a>, 

```bash
wget -O ds/lyon_roofprint.json 'https://download.data.grandlyon.com/wfs/grandlyon?SERVICE=WFS&REQUEST=GetFeature&TYPENAME=ms:fpc_fond_plan_communaut.fpctoit&VERSION=1.1.0&srsName=EPSG:4326&outputFormat=application/json; subtype=geojson'
```


Prepare DataSet
----------------

To transform the vector roofprints to raster labels:

```bash
rsp rasterize --config ds/config.toml --type Building --geojson ds/lyon_roofprint.json --cover ds/cover ds/labels
```

<a href="http://www.datapink.tools/rsp/opendata_to_opendataset/labels/"><img src="img/from_opendata_to_opendataset/labels.png" /></a>


Then to create a training and validation dataset, with imagery and related roofprint labels:

```bash
rsp cover --dir ds/images --splits 70/30 ds/training/cover ds/validation/cover
rsp subset --dir ds/images --cover ds/training/cover  ds/training/images
rsp subset --dir ds/labels --cover ds/training/cover  ds/training/labels
rsp subset --dir ds/images --cover ds/validation/cover  ds/validation/images
rsp subset --dir ds/labels --cover ds/validation/cover  ds/validation/labels
```


Train
-----

Now to launch a first model training:

```bash
rsp train --config ds/config.toml --epochs 10 ds ds/model
```

After ten epochs only, the building IoU metric on validation dataset is about **0.80**. 
It's already a fair result with plain Open Data, but we will see below how to improve it again.




Predictive masks
----------------

To create predictive masks from our first model, on the entire coverage:

```bash
rsp predict --config ds/config.toml --checkpoint ds/model/checkpoint-00010.pth --bs 4 ds ds/masks
```

<a href="http://www.datapink.tools/rsp/opendata_to_opendataset/masks/"><img src="img/from_opendata_to_opendataset/masks.png" /></a>


Compare
-------

Then to assess how our first model behaves with this raw data, we compute a composite stack image with imagery, label and predicted mask.

Pixel colors legend:
 - pink: predicted by the model (but not present on the initial labels)
 - green: present on the labels (but not predicted by the model)
 - grey: both model prediction and labels are agree.




```bash
rsp compare --images ds/images ds/labels ds/masks --mode stack ds/compare

rsp compare --mode list --labels ds/labels --maximum_qod 80 --minimum_fg 5 --masks ds/masks --geojson ds/compare/tiles.json
```

<a href="http://www.datapink.tools/rsp/opendata_to_opendataset/compare/"><img src="img/from_opendata_to_opendataset/compare.png" /></a>

We also run a csv list diff, in order to pick tiles with a low Quality of Data metric (here below 80% on QoD metric) and with at least a handful of buildings' pixels assumed to lie within the tile (5% of foreground building at minimum).

If we zoom back on the map, we can see tiles matching the previous filters:

<a href="http://www.datapink.tools/rsp/opendata_to_opendataset/compare/"><img src="img/from_opendata_to_opendataset/compare_zoom_out.png" /></a>


It is obvious that some areas are not labelled correctly in the original OpenData, so we ought to remove them from the training and validation dataset.

To do so, first step is to select the wrongly labelled tiles. The "compare" tool is again helpful,
as it allows to check several tiles side by side, and to manually select those we want to keep.

```bash
rsp compare --mode side --images ds/images ds/compare --labels ds/labels --maximum_qod 80 --minimum_fg 5 --masks ds/masks  ds/compare_side
```

<a href="http://www.datapink.tools/rsp/opendata_to_opendataset/compare_side/"><img src="img/from_opendata_to_opendataset/compare_side.png" /></a>




Filter
------

The compare selection produces a csv cover list into the clipboard.
We store the result in `ds/to_remove.cover`
```bash
wget -O ds/to_remove.cover http://datapink.tools/rsp/opendata_to_opendataset/to_remove.cover
```

Then we just remove all the related tiles from the dataset:
```bash
rsp subset --dir ds/training/images --cover ds/to_remove.cover --delete > /dev/null
rsp subset --dir ds/training/labels --cover ds/to_remove.cover --delete > /dev/null
rsp subset --dir ds/validation/images --cover ds/to_remove.cover --delete > /dev/null
rsp subset --dir ds/validation/labels --cover ds/to_remove.cover --delete > /dev/null
```

For information, we remove about 500 tiles from this raw dataset in order to clean it up.


Train 
-----

Having a cleaner training and validation dataset, we can launch a second and longer training:

```bash
rsp train --config ds/config.toml --epochs 100 ds ds/model_clean
```

Building IoU metrics on validation dataset:
 - After 10  epochs: **0.82** 
 - After 100 epochs: **0.86**
 
 

Predict and compare
-------------------

And now to generate masks prediction, and compare composite images, as previously done:

```bash
rsp predict --checkpoint ds/model_clean/checkpoint-00100.pth --bs 4 ds ds/masks_clean

rsp compare --images ds/images ds/labels ds/masks_clean --mode stack ds/compare_clean

rsp compare --mode list --labels ds/labels --maximum_qod 80 --minimum_fg 5 --masks ds/masks_clean --geojson ds/compare_clean/tiles.json
```

<a href="http://www.datapink.tools/rsp/opendata_to_opendataset/compare_clean/"><img src="img/from_opendata_to_opendataset/compare_clean.png" /></a>


And to compare only with filtered validation tiles, in side by side mode:

```bash
rsp cover --dir ds/validation/images  ds/validation/cover.clean

rsp subset --dir ds/compare_clean --cover ds/validation/cover.clean ds/validation/compare_clean

rsp subset --dir ds/masks_clean --cover ds/validation/cover.clean ds/validation/masks_clean

rsp compare --mode side --images ds/validation/images ds/validation/compare_clean ds/validation/compare_side_clean
```

<a href="http://www.datapink.tools/rsp/opendata_to_opendataset/compare_side_clean/"><img src="img/from_opendata_to_opendataset/compare_side_clean.png" /></a>
