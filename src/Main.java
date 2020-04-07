import org.opencv.core.Core;

public class Main {
    private static final String calibrationPath = "./res/calibration/";

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Rectification rectification = new Rectification();
        rectification.setUp();

        Calibration calibration = new Calibration();

        calibration.takeImages();

//        Mat calibration_image_1 = Imgcodecs.imread(calibrationPath + "calib0.jpg");
//        Mat calibration_image_2 = Imgcodecs.imread(calibrationPath + "calib1.jpg");
//        MatchingPointsDetector detector = new MatchingPointsDetector();
//        detector.matchImages(calibration_image_1, calibration_image_2);

    }
}
