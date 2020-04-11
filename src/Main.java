import models.RectificationModel;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import utils.CalibrationUtils;
import utils.Utils;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

public class Main {
    private static final String calibrationPath = "./res/output/";
    private static final String imagePath = "./res/images/";

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Rectification rectification = new Rectification();

        Calibration calibration = new Calibration();
        calibration.init();
        try {
            calibration.cameraCalibration(loadImages());
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        Mat PPM1 = new Mat();
        Mat PPM2 = new Mat();
        List<Mat> result = CalibrationUtils.loadPPM(PPM1, PPM2);

        PPM1 = result.get(0);
        PPM2 = result.get(1);

        Mat good_matches_1;
        Mat good_matches_2;
        Mat calibration_image_1 = Imgcodecs.imread(imagePath + "testbilder0.jpg");
        Mat calibration_image_2 = Imgcodecs.imread(imagePath + "testbilder1.jpg");
        Utils utils = new Utils();
        result = utils.computeEpiLines(calibration_image_1, calibration_image_2, null, null);
        good_matches_1 = result.get(0);
        good_matches_2 = result.get(1);

        RectificationModel rectiResults = rectification.doRectification(PPM1, PPM2, good_matches_1, good_matches_2);

        Mat result_image_1 = Imgcodecs.imread(calibrationPath + "recti1.jpg");
        Mat result_image_2 = Imgcodecs.imread(calibrationPath + "recti2.jpg");

        utils.computeEpiLines(result_image_1, result_image_2, rectiResults.getRectifiedImagePoints1(), rectiResults.getRectifiedImagePoints2());
//
//        calibration.takeImages();

//        Mat calibration_image_1 = Imgcodecs.imread(calibrationPath + "recti1.jpg");
//        Mat calibration_image_2 = Imgcodecs.imread(calibrationPath + "recti2.jpg");

//        Matcher matcher = new OFMatcher(calibration_image_1, calibration_image_2);
//        matcher.match();
//
//        Mat outImg = matcher.drawMatchesAndKeyPoints("./res/calibration/test1.jpg");

//        MatchingPointsDetector detector = new MatchingPointsDetector(calibration_image_1, calibration_image_2);
//        detector.matchImages();
    }

    private static List<Mat> loadImages() throws FileNotFoundException {
        List<Mat> images = new ArrayList<>();
        for (int i = 0; i < 15; i++) {
            String currImageName = "calib" + i + ".jpg";

            // Load image
            Mat image = Imgcodecs.imread(calibrationPath + "images/" + currImageName);
            if (image.empty()) {
                System.out.println(currImageName + " Error: File empty.");
                throw new FileNotFoundException(calibrationPath + "images/" + currImageName);
            }
            images.add(image);
        }
        return images;
    }
}
