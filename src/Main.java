import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

public class Main {
    private static final String calibrationPath = "./res/calibration/";
    private static final String imagePath = "./res/images/";

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Rectification rectification = new Rectification();
        rectification.setUp();

        Calibration calibration = new Calibration();

        calibration.takeImages();

        Mat calibration_image_1 = Imgcodecs.imread(calibrationPath + "recti1.jpg");
        Mat calibration_image_2 = Imgcodecs.imread(calibrationPath + "recti2.jpg");
        MatchingPointsDetector detector = new MatchingPointsDetector(calibration_image_1, calibration_image_2);
        detector.matchImages();

    }
}
