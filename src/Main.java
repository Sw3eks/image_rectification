import matcher.MatchingPointsDetector;
import models.CalibrationModel;
import models.RectificationModel;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import utils.CalibrationUtils;
import utils.Utils;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.opencv.calib3d.Calib3d.solvePnPRansac;
import static org.opencv.calib3d.Calib3d.undistortPoints;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;

public class Main {
    private static final String OUTPUT_PATH = "./res/output/";
    private static final String IMAGE_PATH = "./res/images/";

    private static final String CAMERA_PARAMS_FILENAME = "cameraParams";
    private static final String PROJECTION_MATRICES_FILENAME = "projectionMatrices";

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // setup
        Rectification rectification = new Rectification();
        Utils utils = new Utils();
        Calibration calibration = new Calibration();

        // inits variables used in the calibration process
        calibration.init();

        // used to calibrate a connected/embedded webcam by taking images
//        calibration.takeImages();

        // which images in folder /res/calibration shall be rectified
        int index_image_1 = 0;
        int index_image_2 = 1;
        CalibrationModel calibrationModel = new CalibrationModel(null, null);
        // used to calibrate with a given set of images saved in the /res folder
        try {
            calibrationModel = calibration.cameraCalibration(loadImages(), index_image_1, index_image_2);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        // loads the 2 images for rectification
//        Mat calibration_image_1 = Imgcodecs.imread(IMAGE_PATH + "toolkit_image1.jpg");
//        Mat calibration_image_2 = Imgcodecs.imread(IMAGE_PATH + "toolkit_image2.jpg");


        Mat calibration_image_1 = Imgcodecs.imread(OUTPUT_PATH + "calibration/calib" + index_image_1 + ".jpg");
        Mat calibration_image_2 = Imgcodecs.imread(OUTPUT_PATH + "calibration/calib" + index_image_2 + ".jpg");
        // load camera params to undistort images
        Mat intrinsic = new Mat();
        Mat distCoeffs = new Mat();
        List<Mat> resultCamera = CalibrationUtils.loadCameraCalibration(CAMERA_PARAMS_FILENAME, intrinsic, distCoeffs);
        intrinsic = resultCamera.get(0);
        distCoeffs = resultCamera.get(1);
        utils.undistortImages(calibration_image_1, intrinsic, distCoeffs, 1);
        utils.undistortImages(calibration_image_2, intrinsic, distCoeffs, 2);
        MatOfPoint2f undistortedPoints1 = new MatOfPoint2f();
        MatOfPoint2f undistortedPoints2 = new MatOfPoint2f();
        undistortPoints(new MatOfPoint2f(calibrationModel.getCalibrationImagePoints1()), undistortedPoints1, intrinsic, distCoeffs);
        undistortPoints(new MatOfPoint2f(calibrationModel.getCalibrationImagePoints2()), undistortedPoints2, intrinsic, distCoeffs);

        List<Mat> result;
        // detects & computes matching feature points in the 2 given images and draws epilines
//        Mat good_matches_1;
//        Mat good_matches_2;
//        result = utils.computeEpiLines(calibration_image_1, calibration_image_2, calibrationModel.getCalibrationImagePoints1(), calibrationModel.getCalibrationImagePoints2());
//        imwrite("./res/output/epipolar/epipolar_output_3.jpg", result.get(2));
//        imwrite("./res/output/epipolar/epipolar_output_4.jpg", result.get(3));
//        good_matches_1 = result.get(0);
//        good_matches_2 = result.get(1);

        // load projection matrices used for rectification
        Mat PPM1 = new Mat();
        Mat PPM2 = new Mat();
        result = CalibrationUtils.loadPPM(PROJECTION_MATRICES_FILENAME, PPM1, PPM2);
        PPM1 = result.get(0);
        PPM2 = result.get(1);

        // rectification process
        RectificationModel rectiResults = rectification.doRectification(PPM1, PPM2,
                calibration_image_1,
                calibration_image_2,
                calibrationModel.getCalibrationImagePoints1(),
                calibrationModel.getCalibrationImagePoints2());
        imwrite(OUTPUT_PATH + "rectification/rectified_image_3.jpg", rectiResults.getRectifiedImage1());
        imwrite(OUTPUT_PATH + "rectification/rectified_image_4.jpg", rectiResults.getRectifiedImage2());

        // loads the rectified images and draws epilines
        result = utils.computeEpiLines(
                rectiResults.getRectifiedImage1(),
                rectiResults.getRectifiedImage2(),
                rectiResults.getRectifiedImagePoints1(),
                rectiResults.getRectifiedImagePoints2());
        imwrite("./res/output/epipolar/epipolar_output_3.jpg", result.get(2));
        imwrite("./res/output/epipolar/epipolar_output_4.jpg", result.get(3));
        // detects and matches keyPoints and draws epiLines in 1 combined image
//       MatchingPointsDetector detector = new MatchingPointsDetector(rectiResults.getRectifiedImage1(), rectiResults.getRectifiedImage2());
//       detector.matchImages(calibrationModel.getCalibrationImagePoints1(), calibrationModel.getCalibrationImagePoints1());

        utils.mergeImagesAndDrawLine(rectiResults.getRectifiedImage1(), rectiResults.getRectifiedImage2(), rectiResults.getRectifiedImagePoints1(), rectiResults.getRectifiedImagePoints2());
    }

    /**
     * Util function to load saved images from the /res folder
     *
     * @return list of loaded image matrices
     * @throws FileNotFoundException throws exception if file with the given name cant be found
     */
    private static List<Mat> loadImages() throws FileNotFoundException {
        List<Mat> images = new ArrayList<>();
        for (int i = 0; i < 32; i++) {
            String currImageName = "calib" + i + ".jpg";

            // Load image
            Mat image = Imgcodecs.imread(OUTPUT_PATH + "calibration/" + currImageName);
            if (image.empty()) {
                System.out.println(currImageName + " Error: File empty.");
                throw new FileNotFoundException(OUTPUT_PATH + "calibration/" + currImageName);
            }
            images.add(image);
        }
        return images;
    }

    /**
     * Experimental function for calculating translation and rotation vectors
     * (not working properly due to missing real world coordinates)
     *
     * @param good_matches_1 input for matching feature points in image 1
     * @param good_matches_2 input for matching feature points in image 2
     */
    private static void loadAndComputePPM(Mat good_matches_1, Mat good_matches_2) {
        Mat intrinsic = new Mat();
        Mat distCoeffs = new Mat();
        List<Mat> resultCamera = CalibrationUtils.loadCameraCalibration(CAMERA_PARAMS_FILENAME, intrinsic, distCoeffs);

        intrinsic = resultCamera.get(0);
        distCoeffs = resultCamera.get(1);

        Mat rVector1 = new Mat();
        Mat tVector1 = new Mat();
        MatOfPoint2f imagePoints = new MatOfPoint2f();
        imagePoints.push_back(new MatOfPoint2f(new Point(219, 167)));
        imagePoints.push_back(new MatOfPoint2f(new Point(482, 167)));
        imagePoints.push_back(new MatOfPoint2f(new Point(482, 385)));
        imagePoints.push_back(new MatOfPoint2f(new Point(219, 385)));
//        for (int i = 0; i < good_matches_1.rows(); i++) {
//            if (good_matches_1.get(i, 0)[0] > 219 && good_matches_1.get(i, 0)[0] < 482 &&
//                    good_matches_1.get(i, 0)[1] > 167 && good_matches_1.get(i, 0)[0] < 385) {
//                imagePoints.push_back(new MatOfPoint2f(new Point(good_matches_1.get(i, 0)[0], good_matches_1.get(i, 0)[1])));
//            }
//        }
        MatOfPoint3f objectPoints = new MatOfPoint3f();
        objectPoints.push_back(new MatOfPoint3f(new Point3(0, 0, 0)));
        objectPoints.push_back(new MatOfPoint3f(new Point3(1, 0, 0)));
        objectPoints.push_back(new MatOfPoint3f(new Point3(1, 1, 0)));
        objectPoints.push_back(new MatOfPoint3f(new Point3(0, 1, 0)));
//        for (int i = 0; i < imagePoints.rows(); i++) {
//            objectPoints.push_back(new MatOfPoint3f(new Point3(imagePoints.get(i, 0)[0] * 0.2645833333, imagePoints.get(i, 0)[1] * 0.2645833333, 0)));
//        }
        solvePnPRansac(objectPoints, imagePoints, intrinsic, new MatOfDouble(distCoeffs), rVector1, tVector1, false, 100);

        Mat rVector2 = new Mat();
        Mat tVector2 = new Mat();
        imagePoints = new MatOfPoint2f();
        imagePoints.push_back(new MatOfPoint2f(new Point(210, 148)));
        imagePoints.push_back(new MatOfPoint2f(new Point(463, 148)));
        imagePoints.push_back(new MatOfPoint2f(new Point(463, 389)));
        imagePoints.push_back(new MatOfPoint2f(new Point(210, 389)));
//        for (int i = 0; i < good_matches_2.rows(); i++) {
//            if (good_matches_2.get(i, 0)[0] > 210 && good_matches_2.get(i, 0)[0] < 463 &&
//                    good_matches_2.get(i, 0)[1] > 148 && good_matches_2.get(i, 0)[0] < 389) {
//                imagePoints.push_back(new MatOfPoint2f(new Point(good_matches_2.get(i, 0)[0], good_matches_2.get(i, 0)[1])));
//            }
//        }
        objectPoints = new MatOfPoint3f();
        objectPoints.push_back(new MatOfPoint3f(new Point3(0, 0, 0)));
        objectPoints.push_back(new MatOfPoint3f(new Point3(1, 0, 0)));
        objectPoints.push_back(new MatOfPoint3f(new Point3(1, 1, 0)));
        objectPoints.push_back(new MatOfPoint3f(new Point3(0, 1, 0)));
//        for (int i = 0; i < imagePoints.rows(); i++) {
//            objectPoints.push_back(new MatOfPoint3f(new Point3(imagePoints.get(i, 0)[0] * 0.2645833333, imagePoints.get(i, 0)[1] * 0.2645833333, 0)));
//        }
        solvePnPRansac(objectPoints, imagePoints, intrinsic, new MatOfDouble(distCoeffs), rVector2, tVector2, false, 100);

        Utils utils = new Utils();
        utils.calculatePPM(PROJECTION_MATRICES_FILENAME, Arrays.asList(rVector1, rVector2), Arrays.asList(tVector1, tVector2), intrinsic);

    }
}
