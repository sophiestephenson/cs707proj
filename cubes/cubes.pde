
// variables
ArrayList<Camera> cameras = generateCameras(); // the cameras in the scene
int xPos = Config.BOX_X_START;				   // the box x position
boolean running = false;					   // whether the box is moving
ArrayList<Float[]> frames = new ArrayList<>(); // the distances we collect along the way


void setup() {
  size(800, 500, P3D);

  if (Config.CAM_PERSPECTIVE != null) {
	  cameras.get(Config.CAM_PERSPECTIVE).setPerspective()
  }
}

void draw() {
  // make the background and the lighting
  background(215, 238, 250); // light blue
  lights();
  
  // draw each camera
  for (Camera cam : cameras) cam.draw();
  
  // draw the box
  pushMatrix();
  stroke(0);
  fill(255);
  translate(xPos, Config.BOX_Y, Config.BOX_Z);
  box(Config.BOX_SIZE);
  popMatrix();
  
  // start moving if ENTER is pressed
  if (keyPressed) {
    if (key == ENTER && !running) {
      running = true;
    }
  }
  
  if (running) {
	// calc the distances every time we loop 
    calcDistances();
    
    // increment xPos to move the box
    xPos += Config.SPEED;
    
    // at the end: stop, print frames, and save to CSV
    if (xPos >= Config.BOX_X_END) {
      running = false;
      calcDistances();
      printFrames();
      saveFramesToCSV();
    }
  }
}

/**
 * generate a list of cameras we want to use in the scene
 * TODO: make this easy to change the cams used
 *
 * @returns: an arraylist of cameras that we will use in the scene
 */
ArrayList<Camera> generateCameras() {
  ArrayList<int[]> cameraPositions = Config.cameraPositions();
  
  ArrayList<Camera> cameras = new ArrayList<>();
  for (int[] position : cameraPositions) {
    cameras.add(new Camera(position[0], position[1], position[2]));
  }
  
  return cameras;
}

/**
 * calculates the current distance of the box from each camera
 * and adds the list of distances to our frame data
 */
void calcDistances() {
  Float[] distances = new Float[cameras.size()];
  
  int i = 0;
  for (Camera cam : cameras) {
    distances[i] = cam.dist(xPos);
    i++;
  }
  
  frames.add(distances);
}

/**
 * prints the list of distances found at each frame
 */
void printFrames() {
      int frameNo = 0;
      for (Float[] frame : frames) {

        print("frame " + frameNo + ": ");

        for (int camNo = 0; camNo < frame.length; camNo++) {
          print("d" + camNo + " = " + frame[camNo] + "   ");
        }

        print("\n");
        frameNo++;
      }
}

/**
 * saves the distances at each frame to CSV files. 
 * 	(1) a file where each row is a frame and each column is a camera
 *  (2) a file where each row is a camera and each column is a frame
 */
void saveFramesToCSV() {
  // --------------------------------
  // (1) each frame in its own row
  // --------------------------------
  Table table = new Table();
  
  // add columns for all the frames
  table.addColumn("frameID");
  for (int camNo = 0; camNo < frames.get(0).length; camNo++) {
    table.addColumn("d" + str(camNo));
  }
  
  // for each frame, add a new row and put in the distances
  //  of each camera
  for (Float[] frame: frames) {
    TableRow newRow = table.addRow();
    newRow.setInt("frameID", table.lastRowIndex());
    for (int camNo = 0; camNo < frame.length; camNo++) {
      newRow.setFloat("d" + camNo, frame[camNo]);
    }
  }
 
  saveTable(table, Config.FOLDER + "/row_per_frame.csv");
  
  // --------------------------------
  // (2) each camera in its own row
  // --------------------------------
  table = new Table();
  table.addColumn("camera", Table.STRING);
 
  // make rows for each camera
  for (int camNo = 0; camNo < frames.get(0).length; camNo++) {
    table.addRow();
    table.setString(camNo, "camera", "camera" + camNo);
  }
  
  // for each frame, make a new column for the frame and add
  // values in each camera's row
  int frameNo = 0;
  for (Float[] frame: frames) {
    String colName = "frame" + frameNo;
    table.addColumn(colName, Table.FLOAT);
    for (int camNo = 0; camNo < frame.length; camNo++) {
      table.setFloat(camNo, colName, frame[camNo]);
    }
    frameNo++;
  }
  
  saveTable(table, Config.FOLDER + "/row_per_cam.csv");
}
