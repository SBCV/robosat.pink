import os
import sys
import glob
import toml
from importlib import import_module

import re
import math
import colorsys
import webcolors
from pathlib import Path

from robosat_pink.tiles import tile_pixel_to_location, tiles_to_geojson


#
# Import module
#
def load_module(module):
    module = import_module(module)
    assert module, "Unable to import module {}".format(module)
    return module


#
# Config
#
def load_config(path):
    """Loads a dictionary from configuration file."""

    if not path:
        path = os.environ["RSP_CONFIG"] if "RSP_CONFIG" in os.environ else None
    if not path:
        path = os.path.expanduser("~/.rsp_config") if os.path.isfile(os.path.expanduser("~/.rsp_config")) else None
    if not path:
        sys.exit("CONFIG ERROR: Either ~/.rsp_config or RSP_CONFIG env var or --config parameter, is required.")

    config = toml.load(os.path.expanduser(path))
    assert config, "Unable to parse config file"
    config["classes"].insert(0, {"title": "Background", "color": "white"})  # Insert white Background

    # Set default values
    if "model" not in config.keys():
        config["model"] = {}

    if "ts" not in config["model"].keys():
        config["model"]["ts"] = (512, 512)

    if "ms" not in config["model"].keys():
        config["model"]["ms"] = 16

    if "pretrained" not in config["model"].keys():
        config["model"]["pretrained"] = True

    return config


def check_channels(config):
    assert "channels" in config.keys(), "At least one Channel is mandatory"

    # TODO Add name check

    # for channel in config["channels"]:
    #    if not (len(channel["bands"]) == len(channel["mean"]) == len(channel["std"])):
    #        sys.exit("CONFIG ERROR: Inconsistent channel bands, mean or std lenght in config file")


def check_classes(config):
    """Check if config file classes subpart is consistent. Exit on error if not."""

    assert "classes" in config.keys(), "At least one class is mandatory"

    for classe in config["classes"]:
        assert "title" in classe.keys() and len(classe["title"]), "Missing or Empty classes.title.value"
        assert "color" in classe.keys() and check_color(classe["color"]), "Missing or Invalid classes.color value"


def check_model(config):

    hps = {"nn": "str", "pretrained": "bool", "loss": "str", "da": "str"}
    for hp in hps:
        assert hp in config["model"].keys() and type(config["model"][hp]).__name__ == hps[hp], "Missing or Invalid model"


#
# Logs
#
class Logs:
    def __init__(self, path, out=sys.stderr):
        """Create a logs instance on a logs file."""

        self.fp = None
        self.out = out
        if path:
            if not os.path.isdir(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            self.fp = open(path, mode="a")

    def log(self, msg):
        """Log a new message to the opened logs file, and optionnaly on stdout or stderr too."""
        if self.fp:
            self.fp.write(msg + os.linesep)
            self.fp.flush()

        if self.out:
            print(msg, file=self.out)


#
# Colors
#
def make_palette(colors, complementary=False):
    """Builds a One Hot PIL color palette from Classes CSS3 color names, or hex values patterns as #RRGGBB."""

    assert 0 < len(colors) < 8  # 8bits One Hot encoding

    hex_colors = [webcolors.CSS3_NAMES_TO_HEX[color.lower()] if color[0] != "#" else color for color in colors]
    rgb_colors = [(int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)) for h in hex_colors]

    one_hot_colors = [(0, 0, 0) for i in range(256)]
    one_hot_colors[0] = rgb_colors[0]
    for c, color in enumerate(rgb_colors[1:]):
        one_hot_colors[int(math.pow(2, c))] = color

    for i in range(3, int(math.pow(2, len(colors) - 1))):
        if i not in (4, 8, 16, 32, 64, 128):
            one_hot_colors[i] = (0, 0, 0)  # TODO compute colors for overlapping classes

    palette = list(sum(one_hot_colors, ()))  # flatten

    return palette if not complementary else complementary_palette(palette)


def complementary_palette(palette):
    """Creates a PIL complementary colors palette based on an initial PIL palette."""

    comp_palette = []
    colors = [palette[i : i + 3] for i in range(0, len(palette), 3)]

    for color in colors:
        r, g, b = [v for v in color]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        comp_palette.extend(map(int, colorsys.hsv_to_rgb((h + 0.5) % 1, s, v)))

    return comp_palette


def check_color(color):
    """Check if an input color is or not valid (i.e CSS3 color name or #RRGGBB)."""

    hex_color = webcolors.CSS3_NAMES_TO_HEX[color.lower()] if color[0] != "#" else color
    return bool(re.match(r"^#([0-9a-fA-F]){6}$", hex_color))


#
# Web UI
#
def web_ui(out, base_url, coverage_tiles, selected_tiles, ext, template, union_tiles=True):

    out = os.path.expanduser(out)
    template = os.path.expanduser(template)

    templates = glob.glob(os.path.join(Path(__file__).parent, "web_ui", "*"))
    if os.path.isfile(template):
        templates.append(template)
    if os.path.lexists(os.path.join(out, "index.html")):
        os.remove(os.path.join(out, "index.html"))  # if already existing output dir, as symlink can't be overwriten
    os.symlink(os.path.basename(template), os.path.join(out, "index.html"))

    def process_template(template):
        web_ui = open(template, "r").read()
        web_ui = re.sub("{{base_url}}", base_url, web_ui)
        web_ui = re.sub("{{ext}}", ext, web_ui)
        web_ui = re.sub("{{tiles}}", "tiles.json" if selected_tiles else "''", web_ui)

        if coverage_tiles:
            tile = list(coverage_tiles)[0]  # Could surely be improved, but for now, took the first tile to center on
            x, y, z = map(int, [tile.x, tile.y, tile.z])
            web_ui = re.sub("{{zoom}}", str(z), web_ui)
            web_ui = re.sub("{{center}}", str(list(tile_pixel_to_location(tile, 0.5, 0.5))[::-1]), web_ui)

        with open(os.path.join(out, os.path.basename(template)), "w", encoding="utf-8") as fp:
            fp.write(web_ui)

    for template in templates:
        process_template(template)

    if selected_tiles:
        with open(os.path.join(out, "tiles.json"), "w", encoding="utf-8") as fp:
            fp.write(tiles_to_geojson(selected_tiles, union_tiles))
