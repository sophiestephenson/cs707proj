# how to make a scene

make sure you have [processing](https://processing.org/download) downloaded.

## set scene configs

in `config.pde`, set: 
1. `FOLDER`, the desired output folder 
2. `SPEED`, the desired speed of the cube (less than 5 is probably best)
3. `CAM_PERSPECTIVE`, the desired perspective you want. set this to the index of the camera in cameraPositions() that you want to see the perspective of. set to null if you want an overview of the scene.
4. `cameraPositions()`, which returns the list of camera positions for the scene. it's currently four cameras arranged in a circle around the box's path. 
    - keep the cam_y the same and just change the x and z positions of the cameras.
    - TODO: i'm going to make it either generate these positions randomly or make it easier to generate the scenes, stay tuned.

## run the scene

then, go to `cubes.pde` in processing and hit run. this should save the distances at each frame to your desired folder. to get a video of the scene, i have just been doing screen recording of the scene from each camera's perspective.


