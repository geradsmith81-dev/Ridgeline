# Ridgeline

Find where a landscape photo was taken: Ridgeline reads the skyline in a colour or black-and-white
photo and compares it with the skylines computed from open elevation data around a place you
suggest. It shows the best-fitting places on a map with the camera direction, tilt and lens angle,
and draws the model skyline over your photo. Normal photos, stitched panoramas (cylindrical or
equirectangular, with Google GPano data read automatically) and 360° photos are supported. Results can be saved as a CSV list and as a map picture (PNG), and each best place can be shown as a downloadable 3-D oblique view
with the photo's view cone and the predicted skyline ridge.

Made in the style of [Afterglow](https://github.com/geradsmith81-dev/afterglow-planner).

## Run on your computer
    pip install -r requirements.txt
    streamlit run app.py

## Deploy on Streamlit Community Cloud
1. Upload all files to a new GitHub repository (e.g. `ridgeline`).
2. On github.com choose *Add file > Create new file*, type `.streamlit/config.toml` as the name and
   paste the **contents** of `config.toml for GitHub (paste).txt`.
3. On share.streamlit.io choose *Create app*, pick the repository, branch `main`, file `app.py`.

The online version is slower than a PC (Community Cloud gives one shared CPU), so use a smaller
search radius there.

## Accuracy
In testing every "Strong" result lay within 0.33 km of the true camera position (7 of 8 within
15 m); a real iPhone photo from Yosemite's Tunnel View was located 46-49 m from its GPS position.
Photos whose skyline is close (valleys, gentle hills) are harder, and the app says "Unsure" when
several places fit about equally.


### Hazy and faint skylines (trained model)
The skyline finder includes a small neural network (`skymodel.py`, weights in
`assets/sky_model.npz`, plain numpy at run time) that tells sky from land pixel by pixel using
haze-tolerant measures (dark channel, contrast-normalised vertical edges at four scales, colour,
texture, and comparisons with the rest of each column). Its map is blended with the photo's own
sky/land classifier, and the line-finding step follows the place where the learned map drops
from sky to land rather than the strongest raw edge, so a faint far ridge is not passed over for
a crisp near one. Turn it off under "Skyline finder settings".

Training data were rendered from the same elevation tiles and Esri World Imagery: eye-level and
summit views at 46 areas, each shown in clear air, haze (visibility 3-160 km), overcast, white
sky, sunset, dusk, black-and-white, shaded cumulus and stratus, and altitude-dependent valley
haze. On 96 images of 4 held-out areas (Great Smoky Mountains, Tatra, Caucasus, Taiwan) the share
of skyline columns traced within 3 px and kept rose from 50 % to 71 % (haze: 49 to 68 %; shaded
clouds: 45 to 66 %; valley haze: 48 to 70 %) and kept-but-wrong columns fell from 40 % to 19 %.
On real photos the gain is smaller: with the camera held at the GPS position, the share of the
skyline matching the elevation model rose from 37 % to 43 % on 8 photos (Clingmans Dome,
Newfound Gap, Tunnel View), and Tunnel View's match became "Strong" (36 m) instead of
"Possible" (45 m). Scripts are in `training/`.


### Drone photographs
DJI originals carry the gimbal heading, the camera tilt and the height above take-off in XMP
(`skyline.read_drone`). The app shows them, fills in the camera height, and can hold the search
within 15 degrees of the recorded heading (`search.locate(heading_hint=..., heading_window=...)`).
Copies re-saved by photo software usually lose these tags.

### Forested skylines: a negative result
The matcher can add a uniform canopy height to the elevation model
(`search.locate(canopy_try=[0, 10, 20])`, off by default). Tested on 9 geotagged photographs from
Interior Alaska (the 13 pilot searches include duplicate copies of the same pictures) it made
matters worse: median error rose from 706 m to 2,130 m, 6 of the 9 got worse and 3 better, and
every search still graded "Unsure". The limitation in forest is not a constant offset — the skyline is formed by
nearby trees whose angular height depends on their distance, which a bare-earth model cannot
reproduce. The option is kept for experiment, not recommended.


### Reporting a region, and checking it
`search.good_region(coarse, tol)` returns the tested places whose score is within `tol` of the best,
merged into rectangles with their total area; the app draws them on the Leaflet map and in the
downloadable map image, and reports the area in square kilometres. `search.start_agreement(points)`
measures how far apart the best places from several starting points fall; the app can re-run the
search from two further starting points and warns when they disagree by more than a kilometre.
Together these answer a weakness found in testing: in gentle terrain a score ratio can look
respectable while the answer still moves kilometres with the starting point.

Elevation: AWS Terrain Tiles. Maps: OpenTopoMap, OpenStreetMap, Esri. Place search: Open-Meteo.
Leaflet (BSD-2) is bundled in `assets/`. Licence: MIT.
