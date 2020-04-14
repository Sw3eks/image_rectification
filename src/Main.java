import models.RectificationModel;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import utils.CalibrationUtils;
import utils.Utils;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.opencv.calib3d.Calib3d.*;

public class Main {
    private static final String calibrationPath = "./res/output/";
    private static final String imagePath = "./res/images/";

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Rectification rectification = new Rectification();
        Utils utils = new Utils();

        Calibration calibration = new Calibration();
        calibration.init();
//        calibration.takeImages();
//        try {
//            calibration.cameraCalibration(loadImages());
//        } catch (FileNotFoundException e) {
//            e.printStackTrace();
//        }
        Mat calibration_image_1 = Imgcodecs.imread(imagePath + "testbilder0.jpg");
        Mat calibration_image_2 = Imgcodecs.imread(imagePath + "testbilder1.jpg");
        Mat good_matches_1;
        Mat good_matches_2;
        List<Mat> result = utils.computeEpiLines(calibration_image_1, calibration_image_2, null, null);
        good_matches_1 = result.get(0);
        good_matches_2 = result.get(1);

        Mat intrinsic = new Mat();
        Mat distCoeffs = new Mat();
        List<Mat> resultCamera = CalibrationUtils.loadCameraCalibration(intrinsic, distCoeffs);

        intrinsic = resultCamera.get(0);
        distCoeffs = resultCamera.get(1);

        Mat rVector1 = new Mat();
        Mat tVector1 = new Mat();
        MatOfPoint2f imagePoints = new MatOfPoint2f();
        imagePoints.push_back(new MatOfPoint2f(new Point(275, 312)));
        imagePoints.push_back(new MatOfPoint2f(new Point(348, 312)));
        imagePoints.push_back(new MatOfPoint2f(new Point(275, 370)));
        imagePoints.push_back(new MatOfPoint2f(new Point(348, 370)));
        for (int i = 0; i < good_matches_1.rows(); i++) {
            if (good_matches_1.get(i, 0)[0] > 275 && good_matches_1.get(i, 0)[0] < 348 &&
                    good_matches_1.get(i, 0)[1] > 312 && good_matches_1.get(i, 0)[0] < 370) {
                imagePoints.push_back(new MatOfPoint2f(new Point(good_matches_1.get(i, 0)[0], good_matches_1.get(i, 0)[1])));
            }
        }
        MatOfPoint3f objectPoints = new MatOfPoint3f();
//        objectPoints.push_back(new MatOfPoint3f(new Point3(275 * 0.2645833333, 312 * 0.2645833333, 0)));
//        objectPoints.push_back(new MatOfPoint3f(new Point3(348 * 0.2645833333, 312 * 0.2645833333, 0)));
//        objectPoints.push_back(new MatOfPoint3f(new Point3(275 * 0.2645833333, 370 * 0.2645833333, 0)));
//        objectPoints.push_back(new MatOfPoint3f(new Point3(348 * 0.2645833333, 370 * 0.2645833333, 0)));
        for (int i = 0; i < imagePoints.rows(); i++) {
            objectPoints.push_back(new MatOfPoint3f(new Point3(imagePoints.get(i, 0)[0] * 0.2645833333, imagePoints.get(i, 0)[1] * 0.2645833333, 0)));
        }
        solvePnP(objectPoints, imagePoints, intrinsic, new MatOfDouble(distCoeffs), rVector1, tVector1);

        Mat rVector2 = new Mat();
        Mat tVector2 = new Mat();
        imagePoints = new MatOfPoint2f();
//        imagePoints.push_back(new MatOfPoint2f(new Point(310, 324)));
//        imagePoints.push_back(new MatOfPoint2f(new Point(380, 324)));
//        imagePoints.push_back(new MatOfPoint2f(new Point(310, 370)));
//        imagePoints.push_back(new MatOfPoint2f(new Point(380, 370)));
        for (int i = 0; i < good_matches_2.rows(); i++) {
            if (good_matches_2.get(i, 0)[0] > 310 && good_matches_2.get(i, 0)[0] < 380 &&
                    good_matches_2.get(i, 0)[1] > 324 && good_matches_2.get(i, 0)[0] < 370) {
                imagePoints.push_back(new MatOfPoint2f(new Point(good_matches_2.get(i, 0)[0], good_matches_2.get(i, 0)[1])));
            }
        }
        objectPoints = new MatOfPoint3f();
//        objectPoints.push_back(new MatOfPoint3f(new Point3(310 * 0.2645833333, 324 * 0.2645833333, 0)));
//        objectPoints.push_back(new MatOfPoint3f(new Point3(380 * 0.2645833333, 324 * 0.2645833333, 0)));
//        objectPoints.push_back(new MatOfPoint3f(new Point3(310 * 0.2645833333, 370 * 0.2645833333, 0)));
//        objectPoints.push_back(new MatOfPoint3f(new Point3(380 * 0.2645833333, 370 * 0.2645833333, 0)));
        for (int i = 0; i < imagePoints.rows(); i++) {
            objectPoints.push_back(new MatOfPoint3f(new Point3(imagePoints.get(i, 0)[0] * 0.2645833333, imagePoints.get(i, 0)[1] * 0.2645833333, 0)));
        }
        solvePnP(objectPoints, imagePoints, intrinsic, new MatOfDouble(distCoeffs), rVector2, tVector2);

        utils.calculatePPM(Arrays.asList(rVector1, rVector2), Arrays.asList(tVector1, tVector2), intrinsic);

        Mat PPM1 = new Mat();
        Mat PPM2 = new Mat();
        result = CalibrationUtils.loadPPM(PPM1, PPM2);

        PPM1 = result.get(0);
        PPM2 = result.get(1);

        RectificationModel rectiResults = rectification.doRectification(PPM1, PPM2, good_matches_1, good_matches_2);

        Mat result_image_1 = Imgcodecs.imread(calibrationPath + "recti1.jpg");
        Mat result_image_2 = Imgcodecs.imread(calibrationPath + "recti2.jpg");

        utils.computeEpiLines(result_image_1, result_image_2, rectiResults.getRectifiedImagePoints1(), rectiResults.getRectifiedImagePoints2());
    }

    private static List<Mat> loadImages() throws FileNotFoundException {
        List<Mat> images = new ArrayList<>();
        for (int i = 0; i < 16; i++) {
            String currImageName = "calib" + i + ".jpg";

            // Load image
            Mat image = Imgcodecs.imread(calibrationPath + "calibration/" + currImageName);
            if (image.empty()) {
                System.out.println(currImageName + " Error: File empty.");
                throw new FileNotFoundException(calibrationPath + "calibration/" + currImageName);
            }
            images.add(image);
        }
        return images;
    }
}
