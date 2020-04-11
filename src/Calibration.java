import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import utils.CalibrationUtils;

import java.util.ArrayList;
import java.util.List;

import static org.opencv.calib3d.Calib3d.*;
import static org.opencv.core.CvType.CV_64F;
import static org.opencv.highgui.HighGui.*;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;
import static utils.CalibrationUtils.saveCameraCalibration;

public class Calibration {
    private final float calibrationSquareDimension = 0.024f; // meters
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
                boolean found = findChessboardCorners(frame, chessboardDimensions, imageCorners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
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
                        imwrite("./res/output/calibration/calib" + objectPoints.size() + ".jpg", frame);
                        System.out.println("found " + objectPoints.size());
                        if (found) {
                            this.imagePoints.add(imageCorners);
                            imageCorners = new MatOfPoint2f();
                            this.objectPoints.add(obj);
                        }
                        if (objectPoints.size() > 1) {
                            cameraCalibration(null);
                        }
                        break;
                    case 13: // 13 = enter key event
                        System.out.println("Enter");
                        if (objectPoints.size() > 15) {
                            cameraCalibration(null);
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

    public void cameraCalibration(List<Mat> calibrationImages) {

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
        boolean isCalibrated = saveCameraCalibration(intrinsic, distCoeffs);
        if (isCalibrated) {
            System.out.println("Done");
        }
        System.out.println("Result: " + result);
    }

    public void calculatePPM(List<Mat> rVectors, List<Mat> tVectors) {
        Mat r1 = rVectors.get(0);
        Mat r2 = rVectors.get(1);
        Mat R1 = new Mat();
        Mat R2 = new Mat();
        Calib3d.Rodrigues(r1, R1);
        Calib3d.Rodrigues(r2, R2);

        Mat t1 = tVectors.get(0);
        Mat t2 = tVectors.get(1);
        Mat Rt1 = new Mat();
        Mat Rt2 = new Mat();
        Core.hconcat(List.of(R1, t1), Rt1);
        Core.hconcat(List.of(R2, t2), Rt2);
        System.out.println("Rt1: " + Rt1.dump());

        Mat PPM1 = new Mat();
        Mat PPM2 = new Mat();
        Core.gemm(intrinsic, Rt1, 1, new Mat(), 0, PPM1, 0);
        Core.gemm(intrinsic, Rt2, 1, new Mat(), 0, PPM2, 0);

        boolean result = CalibrationUtils.savePPM(PPM1, PPM2);
        if (result) {
            System.out.println("Saved PPM!");
            System.out.println("PPM1: " + PPM1.dump());
            System.out.println("PPM2: " + PPM2.dump());
        }

        Mat calibration_image_1 = Imgcodecs.imread("./res/output/testbilder0.jpg");
        Mat calibration_image_2 = Imgcodecs.imread("./res/output/testbilder1.jpg");
        Mat undistortedImage1 = new Mat();
        Mat undistortedImage2 = new Mat();
        Calib3d.undistort(calibration_image_1, undistortedImage1, intrinsic, distCoeffs);
        Calib3d.undistort(calibration_image_2, undistortedImage2, intrinsic, distCoeffs);
        imwrite("./res/output/undistort1.jpg", undistortedImage1);
        imwrite("./res/output/undistort2.jpg", undistortedImage2);
        MatOfPoint2f undistortedPoints1 = new MatOfPoint2f();
        MatOfPoint2f undistortedPoints2 = new MatOfPoint2f();
        Calib3d.undistortPoints((MatOfPoint2f) imagePoints.get(0), undistortedPoints1, intrinsic, distCoeffs, new Mat(), intrinsic);
        Calib3d.undistortPoints((MatOfPoint2f) imagePoints.get(1), undistortedPoints2, intrinsic, distCoeffs, new Mat(), intrinsic);
    }
}
