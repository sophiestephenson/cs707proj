class Camera {
  static final int VIEW_CENTER_X = 400;
  static final int VIEW_CENTER_Y = 500;
  static final int VIEW_CENTER_Z = -400;
  static final int SIZE = Config.CAM_SIZE;
  
  int x;
  int y;
  int z;
  boolean hidden;
  
  public Camera(int x, int y, int z) {
    this.x = x;
    this.y = y;
    this.z = z;
    hidden = false;
  }
  
  void draw() {
    if (!hidden) {
      fill(198, 1, 16);
      noStroke();
      
      pushMatrix();
      translate(x, y, z);
      sphere(SIZE);
      popMatrix();
    }
  }
  
  void setPerspective() {
    hidden = true;
    camera(x, y, z, VIEW_CENTER_X, VIEW_CENTER_Y, VIEW_CENTER_Z, 0, 1, 0);
  }
  
  float dist(int boxX) {
    float xdist = abs(boxX - x);
    float zdist = abs(Config.BOX_Z - z);
    
    float d = sqrt(pow(xdist, 2) + pow(zdist, 2)) * 100.0;
    return round(d) / 100.0;
  }
}
