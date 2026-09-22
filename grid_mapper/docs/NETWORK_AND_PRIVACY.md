# Network services, privacy and credentials

Grid Mapper has no telemetry and does not send project data to the plugin author.
Some tools intentionally use third-party network services. What leaves the computer depends on the tool and options you choose.

## OpenStreetMap

`Map the grid` uses:

- OpenStreetMap Nominatim for named-place lookup;
- public Overpass API instances for grid features such as power lines and substations.

The requested place/bounding box and query are sent to those services. Results are cached for seven days in the QGIS profile under `grid_mapper/osm_cache` to reduce repeated public-server load.

Transmission towers and distribution poles are optional. They are fetched separately in small cells because large point queries commonly time out on public Overpass servers. A failed optional cell is skipped; it does not invalidate the core grid result.

Self-hosted endpoints can be selected with `GRIDMAPPER_NOMINATIM_URL` and `GRIDMAPPER_OVERPASS_URL`.

## Imagery

When AI scanning is enabled, QGIS requests imagery from the selected imagery layer/provider. Use imagery only under terms that permit your intended analysis, storage and derived products. For commercial or large-scale work, prefer imagery you are licensed to analyse, or your own aerial/drone imagery.

## Local models

ONNX and TFLite inference runs locally. The plugin does not upload imagery when a local model is selected.

Optional external runtimes are not bundled:

- ONNX: `onnxruntime`
- TFLite: `ai-edge-litert`, `tflite-runtime`, or `tensorflow`

Install them into the Python environment used by QGIS. Model weights are also not bundled.

## Roboflow

Roboflow inference uploads each rendered image tile to the configured Roboflow endpoint. Do not use Roboflow mode for imagery or locations that you are not authorised to send to that service.

Prefer supplying the API key through the `ROBOFLOW_API_KEY` environment variable. Processing frameworks may include parameter values in logs, so entering secrets directly into ordinary processing-string parameters can expose them in local logs.

## QGIS network stack

Inside QGIS, Grid Mapper uses QGIS network classes so proxy, SSL and authentication settings are respected. It does not bypass QGIS networking with a direct Python HTTP client.
