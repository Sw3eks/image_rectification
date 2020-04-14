import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import utils.CalibrationUtils;
import utils.Utils;

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
        int i = 2;
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
                        imwrite("./res/output/calibration/testbilder" + i + ".jpg", frame);
                        System.out.println("found " + objectPoints.size());
                        if (found) {
                            this.imagePoints.add(imageCorners);
                            imageCorners = new MatOfPoint2f();
                            this.objectPoints.add(obj);
                        }
                        if (objectPoints.size() > 15) {
                            cameraCalibration(null);
                        }
                        i++;
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
        Utils utils = new Utils();
        utils.calculatePPM(rVectors, tVectors, intrinsic);
    }
}
