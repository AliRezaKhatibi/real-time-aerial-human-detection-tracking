# Using the Current Existing Dataset

Aerial Person Studio can open an existing workspace even if that workspace was moved to a new Windows path.

The recommended recovery path is **not** to trust old absolute paths inside an old manifest. Instead use the current CVAT ZIP packages as the authoritative prepared image set:

1. Open the existing workspace.
2. Dashboard -> Import Existing CVAT ZIPs into Editor.
3. Studio extracts the ZIPs into `work/studio/images/<split>`.
4. Edits happen on the Studio copy.
5. Final export is generated from the Studio copy.

This is particularly useful after intentionally removing black startup frames from a prepared ZIP.
