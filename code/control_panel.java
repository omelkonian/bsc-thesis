// Ros topics
RosTopic<LaserScan> LASER = new RosTopic<>("/scan");
RosTopic<Image> CAM = new RosTopic<>("/camera/rgb");
RosTopic<Image> DEPTH = new RosTopic<>("/camera/depth");
RosTopic<TFMessage> TF = new RosTopic<>("/tf");

Stream<LaserScan> laser = Stream.from(LASER);
Stream<Mat> image = Stream.<Image>from(CAM)
                          .map(CvImage::toCvCopy)          
                          .sample(100, TimeUnit.MILLISECONDS)
                          .map(ControlPanel::faceDetect);
// Embed laser
Stream.combineLatest(laser, image, ControlPanel::embedLaser)
      .subscribe(viz::displayRGB);
// TF       
Stream.from(TF)
  .take(50)
  .collect(HashMap::new, (m, msg) -> {
    List<Transform> transforms = msg.getTransforms();
    for (Transform transform : transforms) {
      String parent = transform.getHeader().getFrameId();
      String child = transform.getChildFrameId();
      if (!m.containsKey(parent)) {
        Set<String> init = new HashSet<>();
        init.add(child);
        m.put(parent, init);
      }
      else m.get(parent).add(child);
    }})
  .subscribe(viz::displayTF);
// Depth
Stream.<Image>from(DEPTH)
      .map(ControlPanel::toGray)
      .sample(100, TimeUnit.MILLISECONDS)
      .subscribe(viz::displayDepth);
// Battery
Stream.interval(2, TimeUnit.SECONDS)
      .map(v -> (100 - v) / 100.0)
      .subscribe(viz::displayBattery);  

static Mat faceDetect(Mat im) {
  if (Visualizer.faceDetection) {        
    Mat temp = new Mat();
    Imgproc.cvtColor(im, temp, Imgproc.COLOR_BGR2GRAY, 3);
    FaceDetector.detectMultiScale(temp, faces);
    for (Rect r : faces.toArray())
      Core.rectangle(im, r);
  }
  return im;
}

static Mat embedLaser(LaserScan l, Mat im) {
  int width = im.cols(), height = im.rows();
  Point center = new Point(width / 2, height);
  float curAngle = l.getAngleMin();
  float[] ranges = l.getRanges();
  for (float range : ranges) {
    double x = center.x + (width / 2 * range * Math.cos(curAngle + Math.PI / 2));
    double y = center.y - (width / l.getRangeMax() * range * Math.sin(curAngle + Math.PI / 2));
    if (Math.abs(curAngle) < 0.3)
      Core.line(im, center, new Point(x, y), Colors.RED);
      curAngle += l.getAngleIncrement();
  }
  Core.circle(im, center, 2, Colors.BLACK, -1);
  return im;
}

static Mat toGray(Image im) {
  im.setEncoding(ImageEncodings.RGBA8);
  Mat mat = CvImage.toCvCopy(im).image;
  Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2GRAY);
  Imgproc.threshold(mat, mat, 150, 255, 0);
  return mat;
}
