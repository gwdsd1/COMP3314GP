# FERET Grayscale Fa/Fb data download instructions

## Complete results The

source is the original grayscale FERET included in the Color FERET archive provided under license from NIST, not a color photo converted to grayscale. Using HTTP byte ranges to extract the specified file, the complete 8,579,819,520-byte tarball was not saved.

Our group selects to retain all relevant positive candidates and additionally organizes the official standard matching list:

| project| quantity|
|---|---:|
| Original grayscale TIFF image| 3,680 |
| Fa Candidate: `fa` / `ba`| 1,962 |
| Fb candidate: `fb` / `bj`| 1,718 |
| official text `.gnd` label| 3,680 |
| official XML `.gnd.xml` annotation| 3,680 |
| official gallery list| 1,196 photos / 1,196 people|
| Official Fb probe list| 1,195 photos / 1,195 people|
| Two official roster confirmed Fa/Fb pairings| **1,195 people, 2,390 tickets**|
| Standard paired image has valid binocular coordinates| **2,390 / 2,390** |

The selected raw data and official documentation total 371,284,046 bytes (~371 MB, 354 MiB); the directory also contains a download index, manifests, inspection records, and scripts.

## file location and usage

- `images/`: 3,680 decompressed raw TIFFs, all width 256, height 384; NumPy image array shape `(384, 256)`.
- `official_fa_fb_pairs.csv`: **This table is preferred when reproduction standard Fa/Fb**. Each line has the identity number, gallery/probe path and respective official annotation path, retaining the relative order of the official gallery list. `official_gallery_order` is the filtered pairing number 1–1,195, not the line number of the original list.
- `fa_fb_candidates.csv`: Identity, expression, collection date, eye coordinates, annotation path and coordinate validity of all candidates. Eye coordinates for standard pairs can be found by file path.
- `metadata/gray_feret_cd1/`, `metadata/gray_feret_cd2/`: Keep the official label, related instructions and list as they are.
- `file_manifest.json`: Source archive path, byte offset, size, source payload SHA-256 and local file SHA-256 of the selected file.
- `download_report.json`: Download statistics and matching verification results.
- `verification.json`: Read back the independent check results of all files after the download is completed.
- `archive_index.json`, `index_checkpoint.jsonl`, `download_checkpoint.jsonl`: remote indexing and resuming records.
- `download_subset.py`: Download/resume script, credentials are only entered at runtime, not written to the file. The

standard pairing is generated from these two original lists:

```
metadata/gray_feret_cd2/partitions/by_previously_reported/feret/gallery.names
metadata/gray_feret_cd2/partitions/by_previously_reported/feret/probe_fafb_expression.names
```

is matched according to the official identity number, one gallery and one probe per person; there is no guessing the earliest date or selecting photos in dictionary order of file names. Gallery's unique identity `00989` does not appear in the official Fb probe list and therefore does not enter the 1,195 pair matching list, and its original image remains.

## naming, eye coordinates and original image

1. Some images from 1996 are named `ba` / `bj`. The official annotation confirms that the expressions are FA / FB, both frontal and inside_feret. You cannot just filter out `fa` / `fb` in the file name. No `bk`, other poses or thumbnails were selected this time. All candidate identities and expressions of
2. are consistent with the official `.gnd` / XML annotation. The original annotations of 317 of the total candidates lacked eye coordinates, and the CSV was left blank and marked `eyes_in_bounds=False`; there were no reconstructed coordinates. None of these images are part of the official collection of 1,195 pairs. All supplied eye coordinates are within the image range.
3. `left_eye_coords` / `right_eye_coords` retains the official human left and right eye semantics, and the coordinates are `(x, y)`. Do not directly regard left as the left side of the screen; when you need to sort the left and right sides of the screen, you should process it according to the x value. Among
4. TIFF, 2,319 pictures are in Pillow `L` mode, and 1,361 pictures are in `P` mode; all the colors in the latter's palette satisfy R=G=B, and they are still grayscale images. `Image.open(path).convert("L")` can be used when reading, and the original TIFF bytes will remain unchanged when saving the file.
5. Only decompressed and saved this time, no alignment, scaling, normalization or additional histogram equalization was performed. Some of the archived images themselves have been annotated with `histogram_adjusted`, and the original historical processing records are retained in `.gnd`.
6. Official Fa/Fb pairing and data integrity verified; this does not mean that the entire experimental protocol of the CSA has been verified. Agreements such as training identity sequence, eye position target, preprocessing, training/test split, etc. still need to be recorded in the experimental code.

## verification scope

- Checks HTTP 206, Content-Range, return length and archive ETag on every network request.
- All 3,680 compressed images pass bzip2 CRC and are fully decoded on download.
- completed rechecking SHA-256 of 11,058 selected files, fully decoding all TIFFs again, and parsing all 3,680 XMLs.
- checked 1,195 unique identities, 2,390 images and binocular coordinates from the official matching table, and all passed.
- SHA-256 is used to log download payloads and inspect local files, and is not a file-by-file signature published by NIST.
- The full archive MD5 was not verified because the complete tar was not downloaded; the full archive MD5 verification cannot be claimed.

The `metadata/doc/partitions/fa.txt` / `fb.txt` of the outer Color FERET belongs to the color library; `metadata/gray_feret_cd1/test/gallery.names` / `probe.names` belongs to the old software test sample. These small files were retained for records during early verification and were not used to generate standard grayscale pairings for this project.

## When the download needs to be resumed

Open a terminal in this directory and use the existing Python and Pillow environments: The

```powershell
python .\download_subset.py candidates
```

script will ask for a NIST username and password, and the password will not be echoed; files that have been downloaded and have correct SHA-256 will be skipped. The account validity period is subject to NIST email. Images should be used under the terms of the FERET license, and the public code repository provides access instructions instead of uploading the original database.

Official description: [Color FERET](https://www.nist.gov/itl/products-and-services/color-feret-database), [FERET Project](https://www.nist.gov/programs-projects/face-recognition-technology-feret).
