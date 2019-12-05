import os
import sys
import csv
import json
import math
import psycopg2
import collections

from tqdm import tqdm
from random import shuffle
from mercantile import tiles, xy_bounds
from rasterio import open as rasterio_open
from rasterio.warp import transform_bounds

from robosat_pink.tiles import tiles_from_dir, tiles_from_csv
from robosat_pink.geojson import geojson_srid, geojson_parse_feature


def add_parser(subparser, formatter_class):

    help = "Generate a tiles covering list (i.e either X,Y,Z or relative path excluding filename extension)"
    parser = subparser.add_parser("cover", help=help, formatter_class=formatter_class)

    inp = parser.add_argument_group("Input [one among the following is required]")
    inp.add_argument("--dir", type=str, help="plain tiles dir path")
    inp.add_argument("--bbox", type=str, help="a lat/lon bbox: xmin,ymin,xmax,ymax or a bbox: xmin,xmin,xmax,xmax,EPSG:xxxx")
    inp.add_argument("--geojson", type=str, nargs="+", help="path to GeoJSON features files")
    inp.add_argument("--cover", type=str, help="a cover file path")
    inp.add_argument("--raster", type=str, help="a raster file path")
    inp.add_argument("--sql", type=str, help="SQL to retrieve geometry features [e.g SELECT geom FROM a_table]")

    db = parser.add_argument_group("Spatial DataBase [required with --sql input]")
    db.add_argument("--pg", type=str, help="PostgreSQL dsn using psycopg2 syntax (e.g 'dbname=db user=postgres')")

    tile = parser.add_argument_group("Tiles")
    tile.add_argument("--no_xyz", action="store_true", help="if set, tiles are not expected to be XYZ based.")

    out = parser.add_argument_group("Outputs")
    out.add_argument("--zoom", type=int, help="zoom level of tiles [required with --geojson or --bbox]")
    out.add_argument("--extent", action="store_true", help="if set, rather than a cover, output a bbox extent")
    out.add_argument("--splits", type=str, help="if set, shuffle and split in several cover subpieces. [e.g 50/15/35]")
    out.add_argument("out", type=str, nargs="*", help="cover csv output paths [required except for extent]")

    parser.set_defaults(func=main)


def main(args):

    assert not (args.extent and args.splits), "--splits and --extent are mutually exclusive options."
    assert not (args.extent and len(args.out) > 1), "--extent option imply a single output."
    assert not (args.sql and not args.pg), "--sql option imply --pg"
    assert (
        int(args.bbox is not None)
        + int(args.geojson is not None)
        + int(args.sql is not None)
        + int(args.dir is not None)
        + int(args.raster is not None)
        + int(args.cover is not None)
        == 1
    ), "One, and only one, input type must be provided, among: --dir, --bbox, --cover, --raster, --geojson or --sql"

    if args.bbox:
        try:
            w, s, e, n, crs = args.bbox.split(",")
            w, s, e, n = map(float, (w, s, e, n))
        except:
            crs = None
            w, s, e, n = map(float, args.bbox.split(","))
        assert isinstance(w, float) and isinstance(s, float) and w < e and s < n, "Invalid bbox parameter."

    if args.splits:
        splits = [int(split) for split in args.splits.split("/")]
        assert len(splits) == len(args.out) and 0 < sum(splits) <= 100, "Invalid split value or incoherent with out paths."

    assert not (not args.zoom and (args.geojson or args.bbox or args.raster)), "Zoom parameter is required."

    args.out = [os.path.expanduser(out) for out in args.out]

    cover = []

    if args.raster:
        print("RoboSat.pink - cover from {} at zoom {}".format(args.raster, args.zoom), file=sys.stderr, flush=True)
        with rasterio_open(os.path.expanduser(args.raster)) as r:
            w, s, e, n = transform_bounds(r.crs, "EPSG:4326", *r.bounds)
            assert isinstance(w, float) and isinstance(s, float), "Unable to deal with raster projection"

            cover = [tile for tile in tiles(w, s, e, n, args.zoom)]

    if args.geojson:
        print("RoboSat.pink - cover from {} at zoom {}".format(args.geojson, args.zoom), file=sys.stderr, flush=True)
        feature_map = collections.defaultdict(list)
        for geojson_file in args.geojson:
            with open(os.path.expanduser(geojson_file)) as f:
                feature_collection = json.load(f)
                srid = geojson_srid(feature_collection)

                for feature in tqdm(feature_collection["features"], ascii=True, unit="feature"):
                    feature_map = geojson_parse_feature(args.zoom, srid, feature_map, feature)

        cover = feature_map.keys()

    if args.sql:
        print("RoboSat.pink - cover from {} {} at zoom {}".format(args.sql, args.pg, args.zoom), file=sys.stderr, flush=True)
        conn = psycopg2.connect(args.pg)
        assert conn, "Unable to connect to PostgreSQL database."
        db = conn.cursor()

        query = """
            WITH
              sql  AS ({}),
              geom AS (SELECT "1" AS geom FROM sql AS t("1"))
              SELECT '{{"type": "Feature", "geometry": '
                     || ST_AsGeoJSON((ST_Dump(ST_Transform(ST_Force2D(geom.geom), 4326))).geom, 6)
                     || '}}' AS features
              FROM geom
            """.format(
            args.sql
        )

        db.execute(query)
        assert db.rowcount is not None and db.rowcount != -1, "SQL Query return no result."

        feature_map = collections.defaultdict(list)

        for feature in tqdm(db.fetchall(), ascii=True, unit="feature"):  # FIXME: fetchall will not always fit in memory...
            feature_map = geojson_parse_feature(args.zoom, 4326, feature_map, json.loads(feature[0]))

        cover = feature_map.keys()

    if args.bbox:
        print("RoboSat.pink - cover from {} at zoom {}".format(args.bbox, args.zoom), file=sys.stderr, flush=True)
        if crs:
            w, s, e, n = transform_bounds(crs, "EPSG:4326", w, s, e, n)
            assert isinstance(w, float) and isinstance(s, float), "Unable to deal with raster projection"

        cover = [tile for tile in tiles(w, s, e, n, args.zoom)]

    if args.cover:
        print("RoboSat.pink - cover from {}".format(args.cover), file=sys.stderr, flush=True)
        cover = [tile for tile in tiles_from_csv(args.cover)]

    if args.dir:
        print("RoboSat.pink - cover from {}".format(args.dir), file=sys.stderr, flush=True)
        cover = [tile for tile in tiles_from_dir(args.dir, xyz=not (args.no_xyz))]

    _cover = []
    extent_w, extent_s, extent_n, extent_e = (180.0, 90.0, -180.0, -90.0)
    for tile in tqdm(cover, ascii=True, unit="tile"):
        if args.zoom and tile.z != args.zoom:
            w, s, n, e = transform_bounds("EPSG:3857", "EPSG:4326", *xy_bounds(tile))
            for t in tiles(w, s, n, e, args.zoom):
                unique = True
                for _t in _cover:
                    if _t == t:
                        unique = False
                if unique:
                    _cover.append(t)
        else:
            if args.extent:
                w, s, n, e = transform_bounds("EPSG:3857", "EPSG:4326", *xy_bounds(tile))
            _cover.append(tile)

        if args.extent:
            extent_w, extent_s, extent_n, extent_e = (min(extent_w, w), min(extent_s, s), max(extent_n, n), max(extent_e, e))

    cover = _cover

    if args.splits:
        shuffle(cover)  # in-place
        cover_splits = [math.floor(len(cover) * split / 100) for i, split in enumerate(splits, 1)]
        if len(splits) > 1 and sum(map(int, splits)) == 100 and len(cover) > sum(map(int, splits)):
            cover_splits[0] = len(cover) - sum(map(int, cover_splits[1:]))  # no tile waste
        s = 0
        covers = []
        for e in cover_splits:
            covers.append(cover[s : s + e])
            s += e
    else:
        covers = [cover]

    if args.extent:
        if args.out and os.path.dirname(args.out[0]) and not os.path.isdir(os.path.dirname(args.out[0])):
            os.makedirs(os.path.dirname(args.out[0]), exist_ok=True)

        extent = "{:.8f},{:.8f},{:.8f},{:.8f}".format(extent_w, extent_s, extent_n, extent_e)

        if args.out:
            with open(args.out[0], "w") as fp:
                fp.write(extent)
        else:
            print(extent)
    else:
        for i, cover in enumerate(covers):

            if os.path.dirname(args.out[i]) and not os.path.isdir(os.path.dirname(args.out[i])):
                os.makedirs(os.path.dirname(args.out[i]), exist_ok=True)

            with open(args.out[i], "w") as fp:
                csv.writer(fp).writerows(cover)
