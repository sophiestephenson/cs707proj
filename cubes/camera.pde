class Camera {
  int x;
  int y;
  int z;
  int size;
  
  public Camera(int x, int y, int z, int size) {
    this.x = x;
    this.y = y;
    this.z = z;
    this.size = size;
  }
  
  void draw() {
    fill(198, 1, 16);
    noStroke();
    
    pushMatrix();
    translate(x, y, z);
    sphere(camsize);
    popMatrix();
  }
  
  float dist(int boxX, int boxZ) {
    float xdist = abs(boxX - x);
    float zdist = abs(boxZ - z);
    
    float d = sqrt(pow(xdist, 2) + pow(zdist, 2)) * 100.0;
    return round(d) / 100.0;
  }
}
