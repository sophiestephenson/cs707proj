static class Config {
  // change for the desired scenario
  public static final String FOLDER = "test"; 
  public static final float SPEED = 5;
  public static final float CAM_PERSPECTIVE = null;
  
  // scene params
  public static final int SCENE_HEIGHT = 500;
  public static final int SCENE_WIDTH = 800;
  
  // positions for the cameras
  public static ArrayList<int[]> cameraPositions() {
      ArrayList<int[]> positions = new ArrayList<>();
      
      // TO DO: generate a bunch of possible positions (x, y, z)
      positions.add(new int[] {0, CAMS_Y, -400}); 
      positions.add(new int[] {400, CAMS_Y, -800}); 
      positions.add(new int[] {800, CAMS_Y, -400}); 
      positions.add(new int[] {400, CAMS_Y, 0}); 
      
      return positions;
  }

  // -----------------------------------
  //	DON'T CHANGE BELOW THIS LINE
  // -----------------------------------
  
  // the best offset between the cube and the cams 
  public static final int OFFSET = 100;
  
  // box moves from x = 0 to x = 800 (with offset so it doesn't hit the cameras)
  // the y coord stays at the bottom of the scene, the z coord stays at -400 so we can see the movement
  public static final int BOX_SIZE = 50;
  public static final int BOX_X_START = OFFSET;
  public static final int BOX_X_END = 800 - OFFSET;
  public static final int BOX_Y = SCENE_HEIGHT - BOX_SIZE/2;
  public static final int BOX_Z = -400;
  
  // size and y coord for the cams
  public static final int CAM_SIZE = 25;
  public static final int CAMS_Y = SCENE_HEIGHT - CAM_SIZE;
  
}
