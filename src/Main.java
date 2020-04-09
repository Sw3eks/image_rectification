import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import utils.Utils;

public class Main {
    private static final String calibrationPath = "./res/calibration/";
    private static final String imagePath = "./res/images/";

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Rectification rectification = new Rectification();
        rectification.setUp();

        Calibration calibration = new Calibration();

        calibration.takeImages();

        Mat calibration_image_1 = Imgcodecs.imread(imagePath + "rect_image_1.jpg");
        Mat calibration_image_2 = Imgcodecs.imread(imagePath + "rect_image_2.jpg");

//        Mat calibration_image_1 = Imgcodecs.imread(calibrationPath + "recti1.jpg");
//        Mat calibration_image_2 = Imgcodecs.imread(calibrationPath + "recti2.jpg");

//        Matcher matcher = new OFMatcher(calibration_image_1, calibration_image_2);
//        matcher.match();
//
//        Mat outImg = matcher.drawMatchesAndKeyPoints("./res/calibration/test1.jpg");
        Utils utils = new Utils();

        utils.computeEpiLines(calibration_image_1, calibration_image_2);

//        MatchingPointsDetector detector = new MatchingPointsDetector(calibration_image_1, calibration_image_2);
//        detector.matchImages();

        // test loading
//        Mat cameraMatrix = new Mat();
//        Mat distCoeffs = new Mat();
//        calibration.loadCameraCalibration(cameraMatrix, distCoeffs);
    }
}
