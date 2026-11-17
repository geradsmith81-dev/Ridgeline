# Ridgeline

Find where a landscape photo was taken: Ridgeline reads the skyline in a colour or black-and-white
photo and compares it with the skylines computed from open elevation data around a place you
suggest. It shows the best-fitting places on a map with the camera direction, tilt and lens angle,
and draws the model skyline over your photo. Normal photos, stitched panoramas (cylindrical or
equirectangular, with Google GPano data read automatically) and 360° photos are supported. Results can be saved as a CSV list and as a map picture (PNG).

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

Elevation: AWS Terrain Tiles. Maps: OpenTopoMap, OpenStreetMap, Esri. Place search: Open-Meteo.
Leaflet (BSD-2) is bundled in `assets/`. Licence: MIT.
