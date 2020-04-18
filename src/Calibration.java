import models.CalibrationModel;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import utils.Utils;

import java.util.ArrayList;
import java.util.List;

import static org.opencv.calib3d.Calib3d.*;
import static org.opencv.core.CvType.CV_64F;
import static org.opencv.highgui.HighGui.*;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;
import static utils.CalibrationUtils.saveCameraCalibration;

public class Calibration {
    private final float calibrationSquareDimension = 0.0245f; // meters
    private final Size chessboardDimensions = new Size(9, 6);

    private VideoCapture capture;
    private List<Mat> imagePoints;
    private List<Mat> objectPoints;
    private MatOfPoint3f obj;
    private MatOfPoint2f imageCorners;
    private Mat intrinsic;
    private Mat distCoeffs;

    /**
     * Init all the (global) variables needed in the controller
     */
    protected void init() {
        this.capture = new VideoCapture();
        this.obj = new MatOfPoint3f();
        this.imageCorners = new MatOfPoint2f();
        this.imagePoints = new ArrayList<>();
        this.objectPoints = new ArrayList<>();
        this.intrinsic = new Mat(3, 3, CV_64F);
        this.distCoeffs = new Mat();
    }

    /**
     * starts the webcam and detects chessboard corners in a pattern
     * take images by pressing 'space' till the threshold (at least 10) to calibrate camera
     */
    public void takeImages() {
        init();
        Mat frame = new Mat();
        Mat drawToFrame = new Mat();

        capture.open(0);
        if (!capture.isOpened()) {
            return;
        }
        int framesPerSecond = 20;
        namedWindow("Webcam", WINDOW_AUTOSIZE); // 640 * 480

        while (capture.read(frame)) {

            capture.read(frame);

            if (!frame.empty()) {
                boolean found = findChessboardCorners(frame, chessboardDimensions, imageCorners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
                frame.copyTo(drawToFrame);

                drawChessboardCorners(drawToFrame, chessboardDimensions, imageCorners, found);

                if (found) {
                    imshow("Webcam", drawToFrame);
                } else {
                    imshow("Webcam", frame);
                }
                int character = waitKey(1000 / framesPerSecond);

                switch (character) {
                    case 32: // 32 = space key event
                        if (found) {
                            imwrite("./res/output/calibration/calib" + objectPoints.size() + ".jpg", frame);
                            imwrite("./res/output/calibration/calibWithChess" + objectPoints.size() + ".jpg", drawToFrame);
                            System.out.println("found " + objectPoints.size());
                            this.imagePoints.add(imageCorners);
                            imageCorners = new MatOfPoint2f();
                            this.objectPoints.add(obj);
                        }
                        if (objectPoints.size() > 31) {
                            cameraCalibration(null, 0, 1);
                        }
                        break;
                    case 27: // 27 = esc key event
                        System.out.println("Esc");
                        return;
                    default:
                        break;
                }
            }
        }

    }

    /**
     * Calibrates the camera for the given images
     *
     * @param calibrationImages images for calibration
     * @param index_1           index for image 1
     * @param index_2           index for image 2
     * @return image points of the images with the given index
     */
    public CalibrationModel cameraCalibration(List<Mat> calibrationImages, int index_1, int index_2) {

        if (calibrationImages != null) {

            getChessBoardCorners(calibrationImages);
        } else {

            createKnownBoardPosition();
        }

        List<Mat> rVectors = new ArrayList<>();
        List<Mat> tVectors = new ArrayList<>();

        intrinsic.put(0, 0, 1);
        intrinsic.put(1, 1, 1);

        distCoeffs = Mat.zeros(5, 1, CV_64F);

        double result = calibrateCamera(objectPoints, imagePoints, chessboardDimensions, intrinsic, distCoeffs, rVectors, tVectors);
        calculatePPM(rVectors, tVectors);
        boolean isCalibrated = saveCameraCalibration("cameraParams", intrinsic, distCoeffs);
        if (isCalibrated) {
            System.out.println("Done");
        }
        System.out.println("Result: " + result);
        return new CalibrationModel(imagePoints.get(index_1), imagePoints.get(index_2));
    }

    public void createKnownBoardPosition() {
        for (int i = 0; i < chessboardDimensions.height; i++) {
            for (int j = 0; j < chessboardDimensions.width; j++) {
                obj.push_back(new MatOfPoint3f(new Point3(j * calibrationSquareDimension, i * calibrationSquareDimension, 0.0f)));
            }
        }
    }

    public void getChessBoardCorners(List<Mat> images) {
        for (Mat image : images) {
            obj = new MatOfPoint3f();
            createKnownBoardPosition();
            Mat grayimg = new Mat();
            Imgproc.cvtColor(image, grayimg, Imgproc.COLOR_BGR2GRAY);
            MatOfPoint2f pointBuf = new MatOfPoint2f();
            boolean found = findChessboardCorners(grayimg, chessboardDimensions, pointBuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
            if (found) {
                TermCriteria term = new TermCriteria(TermCriteria.EPS | TermCriteria.MAX_ITER, 30, 0.1);
                Imgproc.cornerSubPix(grayimg, pointBuf, new Size(11, 11), new Size(-1, -1), term);

                this.imagePoints.add(pointBuf);
                this.objectPoints.add(obj);
            }

        }
    }

    public void calculatePPM(List<Mat> rVectors, List<Mat> tVectors) {
        Utils utils = new Utils();
        utils.calculatePPM("projectionMatrices", rVectors, tVectors, intrinsic);
    }
}
