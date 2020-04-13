import models.RectificationModel;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import utils.CalibrationUtils;
import utils.Utils;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.opencv.calib3d.Calib3d.solvePnP;

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

        Mat intrinsic = new Mat();
        Mat distCoeffs = new Mat();
        List<Mat> resultCamera = CalibrationUtils.loadCameraCalibration(intrinsic, distCoeffs);

        intrinsic = resultCamera.get(0);
        distCoeffs = resultCamera.get(1);

        Mat rVector1 = new Mat();
        Mat tVector1 = new Mat();
        MatOfPoint3f objectPoints = new MatOfPoint3f();
        objectPoints.push_back(new MatOfPoint3f(new Point3(0.0f, 0.0f, 0.0f)));
        objectPoints.push_back(new MatOfPoint3f(new Point3(calibration_image_1.cols(), 0.0f, 0.0f)));
        objectPoints.push_back(new MatOfPoint3f(new Point3(calibration_image_1.cols(), calibration_image_1.rows(), 0.0f)));
        objectPoints.push_back(new MatOfPoint3f(new Point3(0.0f, calibration_image_1.rows(), 0.0f)));
        MatOfPoint2f imagePoints = new MatOfPoint2f();
        imagePoints.push_back(new MatOfPoint2f(new Point(0.0f, 0.0f)));
        imagePoints.push_back(new MatOfPoint2f(new Point(calibration_image_1.cols(), 0.0f)));
        imagePoints.push_back(new MatOfPoint2f(new Point(calibration_image_1.cols(), calibration_image_1.rows())));
        imagePoints.push_back(new MatOfPoint2f(new Point(0.0f, calibration_image_1.rows())));
        solvePnP(objectPoints, imagePoints, intrinsic, new MatOfDouble(distCoeffs), rVector1, tVector1);

        Mat rVector2 = new Mat();
        Mat tVector2 = new Mat();
        objectPoints = new MatOfPoint3f();
        objectPoints.push_back(new MatOfPoint3f(new Point3(0.0f, 0.0f, 0.0f)));
        objectPoints.push_back(new MatOfPoint3f(new Point3(calibration_image_2.cols(), 0.0f, 0.0f)));
        objectPoints.push_back(new MatOfPoint3f(new Point3(calibration_image_2.cols(), calibration_image_2.rows(), 0.0f)));
        objectPoints.push_back(new MatOfPoint3f(new Point3(0.0f, calibration_image_2.rows(), 0.0f)));
        imagePoints = new MatOfPoint2f();
        imagePoints.push_back(new MatOfPoint2f(new Point(0.0f, 0.0f)));
        imagePoints.push_back(new MatOfPoint2f(new Point(calibration_image_2.cols(), 0.0f)));
        imagePoints.push_back(new MatOfPoint2f(new Point(calibration_image_2.cols(), calibration_image_2.rows())));
        imagePoints.push_back(new MatOfPoint2f(new Point(0.0f, calibration_image_2.rows())));
        solvePnP(objectPoints, imagePoints, intrinsic, new MatOfDouble(distCoeffs), rVector2, tVector2);

        utils.calculatePPM(Arrays.asList(rVector1, rVector2), Arrays.asList(tVector1, tVector2), intrinsic);

        Mat PPM1 = new Mat();
        Mat PPM2 = new Mat();
        List<Mat> result = CalibrationUtils.loadPPM(PPM1, PPM2);

        PPM1 = result.get(0);
        PPM2 = result.get(1);
        Mat good_matches_1;
        Mat good_matches_2;
        result = utils.computeEpiLines(calibration_image_1, calibration_image_2, null, null);
        good_matches_1 = result.get(0);
        good_matches_2 = result.get(1);

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
