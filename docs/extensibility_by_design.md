# How to extend RoboSat.pink: 

RoboSat.pink extensibility design allows you to custom it easily to your specifics needs.



## Use an alternate Web UI Templates ##
RoboSat.pink tools outputing XYZ tiles, can generate, on demand, a Web UI, with `--web_ui` parameter.
To switch to your own HTML template, just add `--web_ui_template` followed by your own template path.

Special values are substitued, if presents, in a template:
 - `{{base_url}}` XYZ directory base url, containing tiles. 
 - `{{ext}}` file extension to tiles image.
 - `{{zoom}}` z level from tiles coverage.
 - `{{center}}` lon, lat coordinates from center point of coverage. 
 - `{{tiles}}` JSON selected tiles list, if any. 


<details><summary>Click me, for a minimal Leaflet based template example</summary>
 
```
<!DOCTYPE html>
<html>
<head>
  <title>RoboSat.pink Leaflet WebUI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.3.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.3.4/dist/leaflet.js"></script>
</head>
<body>
  <div id="mapid" style="width:100%; height:100vh;"></div>
  <script>
   var m = L.map("mapid").setView({{center}}, {{zoom}});
   L.tileLayer("{{base_url}}/{z}/{x}/{y}.{{ext}}").addTo(m);
  </script>
</body>
</html>
```

</details>


## Use an alternate OSM type extractor ##
To allows `rsp extract` to handle new OSM types:
- If not alredy done, retrieve RoboSat.pink code source, and proceed to dev install: `make install`.
- Create in `robosat_pink/osm` directory, a `yourtypename.py` file.
- This file must contains at least a `yourtypenameHandler` class, with `__init__`, `ways` and `save` methods.

<details><summary>Click me, for a simple OSM parks extractor example</summary>

```
import osmium
import geojson
import shapely.geometry


class ParkHandler(osmium.SimpleHandler):

    def __init__(self):
        super().__init__()
        self.features = []

    def way(self, w):
        if "leisure" not in w.tags or w.tags["leisure"] != "park":
            return
            
        if not w.is_closed() or len(w.nodes) < 4:
            return

        geometry = geojson.Polygon([[(n.lon, n.lat) for n in w.nodes]])
        shape = shapely.geometry.shape(geometry)

        if shape.is_valid:
            feature = geojson.Feature(geometry=geometry)
            self.features.append(feature)

    def save(self, out):
        with open(out, "w") as fp:
            geojson.dump(geojson.FeatureCollection(self.features), fp)
```

Callable with `rsp extract --type Park`

</details>


## Use an alternate Metric ##

To allows `rsp train` to use a metric function of your own:
- If not alredy done, retrieve RoboSat.pink code source, and proceed to dev install: `make install`.
- Create in `robosat_pink/metrics` directory a yourmetricname.py file.
- This file must contains at least a class `get` function , with `get(label, predicted, config=None)` protototype, and returning a [0-1] float.
- Take note, that both label vars and predicted are located on GPU device (moving them back by any computation on CPU, would affect training performances).
- Then, to use it with `rsp train`, update config file value: `["model"]["metrics"]`, and add it's name in the list (in lowercase).

<details><summary>Click me, for a simple Precision metric example</summary>

```
from robosat_pink.metrics.core import confusion
  

def get(label, predicted, config=None):

    tn, fn, fp, tp = confusion(label, predicted)

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0.0

    return precision
```
</details>




## Use an alternate Loss function ##

To allows `rsp train` to use a loss function of your own:
- If not alredy done, retrieve RoboSat.pink code source, and proceed to dev install: `make install`.
- Create in `robosat_pink/losses` directory a yourlossname.py file.
- This file must contains at least a class `Yourlossname` , with `__init__` and `forward` methods.
- If your loss computation is not auto-differentiable by PyTorch, a related `backward` method, will be needed too.
- Then, to use it with `rsp train`, either:
  - update config file value: `["model"]["loss"]`
  - use `--loss` parameter

<details><summary>Click me, for a MIoU loss example</summary>

```
import torch

class Miou(torch.nn.Module):
    """mIoU Loss. cf http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf"""

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, config):

        N, C, H, W = inputs.size()

        softs = torch.nn.functional.softmax(inputs, dim=1).permute(1, 0, 2, 3)
        masks = torch.zeros(N, C, H, W).to(targets.device).scatter_(1, targets.view(N, 1, H, W), 1).permute(1, 0, 2, 3)

        inters = softs * masks
        unions = (softs + masks) - (softs * masks)
        mIoU = 1. - (inters.view(C, N, -1).sum(2) / unions.view(C, N, -1).sum(2)).mean()

        return mIoU
```

Callable with `rsp train --loss Miou`

</details>


## Use an alternate Data Augmentation setting ##

To allows `rsp train` to use a data augmentation setting of your own:
- If not alredy done, retrieve RoboSat.pink code source, and proceed to dev install: `make install`.
- Create in `robosat_pink/da` directory, a yourda.py file.
- Albumentations syntax documentation:  <a href="https://github.com/albu/albumentations/blob/master/docs/examples.rst">https://github.com/albu/albumentations/blob/master/docs/examples.rst</a>
- Then, to use it with `rsp train`, either:
  - update config file value: `["model"]["da"]`
  - use `--da` parameter

<details><summary>Click me, for a Minimal DataAugmentation example</summary>

```
from albumentations import Compose, Flip, Transpose
  
def transform(config, image, mask):

    try:
        p = config["model"]["dap"]
    except:
        p = 1

    assert 0 <= p <= 1

    return Compose(
        [
            Flip(), # implicit p=0.5
            Transpose(p=0.3)
        ]
    )(image=image, mask=mask, p=p)
```
Callable with `rsp train --da Yourda`
</details>


## Use an alternate Neural Network Model ##
To allows `rsp train` and `rsp predict` to use a model of your own:
- If not alredy done, retrieve RoboSat.pink code source, and proceed to dev install: `make install`.
- Create in `robosat_pink/nn` directory a `yourmodelname.py` file.
- This file must contains at least a `Model_name` class, with `__init__` and `forward` methods.
- `shape_in` and `shape_out` as obvious parameters
- Additional config information is restrain to train phase only (by design, to ensure ability to export model).
- Then, to use it, with `rsp train` and `rsp predict` either:
  - update config file value: `["model"]["nn"]`
  - use `--nn` parameter

<details><summary>Click me, for a UNet model example</summary>

```
import torch
import torch.nn as nn


class Unet(nn.Module):

    def __init__(self, shape_in, shape_out, train_config=None):

        self.doc = """UNet - cf https://arxiv.org/abs/1505.04597"""
        self.version = 1

        num_classes = shape_out[0]
        num_channels = shape_in[0]
        assert num_channels == 3, "This basic UNet example is RGB only."

        super().__init__()

        self.b1 = Block(3, 64)
        self.d1 = Downsample()
        self.b2 = Block(64, 128)
        self.d2 = Downsample()
        self.b3 = Block(128, 256)
        self.d3 = Downsample()
        self.b4 = Block(256, 512)
        self.d4 = Downsample()
        self.b5 = Block(512, 1024)
        self.u1 = Upsample(1024)
        self.b6 = Block(1024, 512)
        self.u2 = Upsample(512)
        self.b7 = Block(512, 256)
        self.u3 = Upsample(256)
        self.b8 = Block(256, 128)
        self.u4 = Upsample(128)
        self.b9 = Block(128, 64)
        self.b10 = nn.Conv2d(64, num_classes, kernel_size=1)

        self.initialize()

    def forward(self, x):
        b1 = self.b1(x)
        d1 = self.d1(b1)
        b2 = self.b2(d1)
        d2 = self.d2(b2)
        b3 = self.b3(d2)
        d3 = self.d3(b3)
        b4 = self.b4(d3)
        d4 = self.d4(b4)
        b5 = self.b5(d4)
        u1 = self.u1(b5)
        b6 = self.b6(torch.cat([b4, u1], dim=1))
        u2 = self.u2(b6)
        b7 = self.b7(torch.cat([b3, u2], dim=1))
        u3 = self.u3(b7)
        b8 = self.b8(torch.cat([b2, u3], dim=1))
        u4 = self.u4(b8)
        b9 = self.b9(torch.cat([b1, u4], dim=1))
        b10 = self.b10(b9)

        return b10

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def Block(num_in, num_out):
    return nn.Sequential(
        nn.Conv2d(num_in, num_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_out),
        nn.PReLU(num_parameters=num_out),
        nn.Conv2d(num_out, num_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_out),
        nn.PReLU(num_parameters=num_out),
    )


def Downsample():
    return nn.MaxPool2d(kernel_size=2, stride=2)


def Upsample(num_in):
    return nn.ConvTranspose2d(num_in, num_in // 2, kernel_size=2, stride=2)
```

Callable with `rsp train --nn Unet`

</details>
