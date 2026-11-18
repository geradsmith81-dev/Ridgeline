# Training the hazy-skyline model

These scripts made `assets/sky_model.npz`. They are not needed to run the app.

1. `synth.py` renders eye-level views from AWS Terrain Tiles and Esri World Imagery (true skyline,
   distance per pixel) and dresses them in weather: haze `I = J t + A (1 - t)`, overcast, white
   sky, sunset, dusk, clouds, sun glow, black-and-white, blur, noise, JPEG.
2. `clouds.py` adds shaded cumulus / stratus / storm cloud to the sky of existing views.
3. `valleyhaze.py` adds altitude-dependent haze (valleys hazier than ridge tops), integrated along
   each line of sight with Earth curvature.
4. `summits.py` adds views from summits and overlooks over layered lower ridges.
5. `train.py` samples pixels (half within 30 px of the visible skyline, half at random, smeared
   ground within 150 m and land hidden by haze left out), computes the 27 measures of
   `skymodel.pixel_features`, trains a 27-48-24-1 network (scikit-learn) and saves the weights.
6. `evalsky.py` scores skyline tracing on held-out areas; `realtest.py` checks real photos with GPS.

Held-out areas (never trained on): Great Smoky Mountains, Tatra, Caucasus, central Taiwan.
Generated images are not included (about 1,300 images); re-run the scripts to rebuild them.
