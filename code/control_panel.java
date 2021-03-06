Stream<LaserScan> laser = 
    Stream.from(new RosTopic<>("/scan"));
Stream<Mat> image = 
    Stream.<Image>from(new RosTopic<>("/camera/rgb"))
        .map(CvImage::toCvCopy)          
        .sample(100, TimeUnit.MILLISECONDS)
        .map(this::faceDetect);
Stream.zip(laser, image, this::embedLaser)
      .subscribe(viz::displayRGB);
Stream.from(new RosTopic<>("/tf"))
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
Stream.<Image>from(new RosTopic<>("/camera/depth"))
      .map(this::toGray)
      .sample(100, TimeUnit.MILLISECONDS)
      .subscribe(viz::displayDepth);
Stream.interval(2, TimeUnit.SECONDS)
      .map(v -> (100 - v) / 100.0)
      .subscribe(viz::displayBattery);  