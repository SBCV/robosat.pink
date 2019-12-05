# Feature Extraction


Additional tools used, in this tutorial:
---------------------------------------
```bash
sudo apt install -y gdal-bin postgresql-11-postgis-2.5
```


```
sudo su postgres -c "createdb tanzania"
sudo su postgres -c "psql -c 'CREATE EXTENSION postgis' tanzania"
sudo su postgres -c "psql -c \"CREATE USER rsp WITH PASSWORD 'pass' \""
```

GeoJSON Cleanup and aggregation
-------------------------------

```bash
ogr2ogr -f "PostgreSQL" PG:"user=rsp host=localhost dbname=tanzania password=pass" predict/building.json -t_srs EPSG:3857 -nlt PROMOTE_TO_MULTI -nln masks -lco GEOMETRY_NAME=geom
```

```bash
echo """
UPDATE masks SET geom=ST_MakeValid(geom) WHERE NOT ST_IsValid(geom);     -- clean it up

CREATE TABLE aggregate_masks AS (                                        -- aggegate tiled features

WITH a AS (SELECT array_agg(b.ogc_fid) AS tuples FROM masks a, masks b WHERE ST_DWithin(a.geom, b.geom, 0.01) GROUP BY a.ogc_fid),
     b AS (SELECT DISTINCT tuples::int[] AS ogc_fids FROM a WHERE array_length(tuples, 1) > 1),
     c AS (SELECT tuples[1] AS ogc_fid FROM a WHERE array_length(tuples, 1) = 1),
     d AS (SELECT (ST_Dump(ST_Union(geom))).geom AS geom FROM masks, b WHERE masks.ogc_fid = ANY(b.ogc_fids)
           UNION
           SELECT geom FROM masks, c WHERE masks.ogc_fid = c.ogc_fid)

SELECT row_number() OVER() AS gid, geom FROM d WHERE ST_Area(geom) > 5.0 -- remove artefacts
);


CREATE INDEX ON aggregate_masks USING GIST(geom);                        -- spatial index
""" > SQL
sudo su postgres -c "psql tanzania < SQL"
```




OSM Features Dedupe
-------------------
TODO



Export back to GeoJSON
----------------------

```bash
echo """
SELECT '{\"type\": \"FeatureCollection\", \"features\": ['
        || string_agg('{\"type\": \"Feature\", \"properties\": {\"id\":\"' || gid
        || '\"},\"geometry\":' || ST_AsGeoJSON(ST_Transform(ST_Multi(geom), 4326), 6) || '}', ',')
        || ']}'
FROM aggregate_masks
""" > SQL
sudo su postgres -c "psql -t tanzania < SQL > /tmp/masks.geojson"
```



FAIR Rapid
----------
TODO
