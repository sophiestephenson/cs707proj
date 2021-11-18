// files
String folder = "two_cameras";

// sizes
int boxsize = 50;
int camsize = 25;
int offset = 75; // space between cam and cube when starting/stopping

// camera sizes
int camsY = 500 - camsize;
int cam1X = 0;
int cam1Z = -400;
int cam2X = 400; 
int cam2Z = -800;
int cam3X = 800;
int cam3Z = -400;
int cam4X = 400;
int cam4Z = 0;

// cameras
Camera cam1 = new Camera(cam1X, camsY, cam1Z, camsize);
//Camera cam2 = new Camera(cam2X, camsY, cam2Z, camsize);
Camera cam2 = new Camera(cam3X, camsY, cam3Z, camsize);
//Camera cam4 = new Camera(cam4X, camsY, cam4Z, camsize);

// box starting position
int xPos = cam1X + offset;
int yPos = 500 - (boxsize/2);
int zPos = -400;

// box stop position
int stopX = cam3X - offset;

// movement
float speed = 1;
boolean running = false;

// data collection
ArrayList<Float[]> frames = new ArrayList<>();


void setup() {
  size(800, 500, P3D);
  cam2.set_camera();
}



void draw() {
  // scene
  background(215, 238, 250); // light blue
  lights();
  
  cam1.draw();
  cam2.draw();
  //cam3.draw();
  //cam4.draw();
  
  pushMatrix();
  stroke(0);
  fill(255);
  translate(xPos, yPos, zPos);
  box(boxsize);
  popMatrix();
  
  
  if (keyPressed) {
    if (key == ENTER && !running) {
      running = true;
    }
  }
  
  if (running) {
    calcDistances();
    
    // increment xPos and yPos
    xPos += speed;
    if (xPos >= stopX) {
      running = false;
      endRecord();
      calcDistances();
      printFrames(frames);
      saveFramesToCSV(frames);
    }
  }
  
}

void calcDistances() {
    // note distances
    Float[] distances = {cam1.dist(xPos, zPos),
                         cam2.dist(xPos, zPos),
                        // cam3.dist(xPos, zPos),
                        // cam4.dist(xPos, zPos)
                        };
    frames.add(distances);
}

void printFrames(ArrayList<Float[]> frames) {
      int i = 0;
      for (Float[] frame : frames) {
        print("frame " + i + ": ");
        print("d1 = " + frame[0] + "   ");
        print("d2 = " + frame[1] + "   ");
       // print("d3 = " + frame[2] + "   ");
       // print("d4 = " + frame[3] + "   ");
       print("\n");
        i++;
      }
}

void saveFramesToCSV(ArrayList<Float[]> frames) {
  // each frame in its own row
  Table table = new Table();
  
  table.addColumn("frameID");
  table.addColumn("d1");
  table.addColumn("d2");
  //table.addColumn("d3");
  //table.addColumn("d4");
  
  for (Float[] frame: frames) {
    TableRow newRow = table.addRow();
    newRow.setInt("frameID", table.lastRowIndex());
    newRow.setFloat("d1", frame[0]);
    newRow.setFloat("d2", frame[1]);
    //newRow.setFloat("d3", frame[2]);
    //newRow.setFloat("d4", frame[3]);
  }

  saveTable(table, folder + "/row_per_frame.csv");
  
  // each camera in its own row
  table = new Table();
  table.addColumn("camera", Table.STRING);
  table.addRow();
  table.setString(0, "camera", "camera1");
  table.addRow();
  table.setString(1, "camera", "camera2");
  //table.addRow();
  //table.setString(2, "camera", "camera3");
  //table.addRow();
  //table.setString(3, "camera", "camera4");
  
  int frameNo = 0;
  for (Float[] frame: frames) {
    String colName = "frame" + frameNo;
    table.addColumn(colName, Table.FLOAT);
    table.setFloat(0, colName, frame[0]);
    table.setFloat(1, colName, frame[1]);
    //table.setFloat(2, colName, frame[2]);
    //table.setFloat(3, colName, frame[3]);
    frameNo++;
  }
  
  saveTable(table, folder + "/row_per_cam.csv");
  
  
}
