import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.videoio.VideoCapture;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.opencv.calib3d.Calib3d.*;
import static org.opencv.core.CvType.CV_64F;
import static org.opencv.highgui.HighGui.*;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;

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

    public void getChessBoardCorners(List<Mat> images, boolean showResults) {
        for (Mat image : images) {
            MatOfPoint2f pointBuf = new MatOfPoint2f();
            boolean found = findChessboardCorners(image, chessboardDimensions, pointBuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
            if (found) {
                imageCorners.push_back(pointBuf);
            }
            if (showResults) {
                drawChessboardCorners(image, chessboardDimensions, pointBuf, found);
                imshow("Looking for Corners", image);
                waitKey(0);
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
                        if (found) {
                            //imwrite("./res/calibration/calib" + objectPoints.size() + ".jpg", frame);
                            System.out.println("found");
                            this.imagePoints.add(imageCorners);
                            imageCorners = new MatOfPoint2f();
                            this.objectPoints.add(obj);
                        }
                        if (objectPoints.size() > 1) {
                            cameraCalibration();
                            boolean result = saveCameraCalibration();
                            if (result) {
                                System.out.println("Done");
                            }
                        }
                        break;
                    case 13: // 13 = enter key event
                        System.out.println("Enter");
                        if (objectPoints.size() > 15) {
                            cameraCalibration();
                            boolean result = saveCameraCalibration();
                            if (result) {
                                System.out.println("Done");
                            }
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

    public void cameraCalibration() {

        // getChessBoardCorners(calibrationImages, checkerBoardImageSpacePoints, false);
        // imwrite("./res/calibration/calib1.jpg", obj);

        createKnownBoardPosition();

        List<Mat> rVectors = new ArrayList<>();
        List<Mat> tVectors = new ArrayList<>();

        intrinsic.put(0, 0, 1);
        intrinsic.put(1, 1, 1);

        distCoeffs = Mat.zeros(8, 1, CV_64F);

        double result = calibrateCamera(objectPoints, imagePoints, chessboardDimensions, intrinsic, distCoeffs, rVectors, tVectors);
        calculatePPM(rVectors, tVectors);
        System.out.println("Result: " + result);
    }

    public boolean saveCameraCalibration() {
        FileWriter fStream = null;
        try {
            fStream = new FileWriter("out.txt", true);
            BufferedWriter out = new BufferedWriter(fStream);

            out.write("Matrix: " + intrinsic.dump());
            out.write("\nDist: " + distCoeffs.dump());

            //Close the output stream
            out.close();
            return true;
        } catch (IOException e) {
            System.out.println("Exception: " + e.getMessage());
            Logger.getLogger(getClass().getName()).log(Level.SEVERE, null, e);
        } finally {
            try {
                if (fStream != null) {
                    fStream.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return false;
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

        System.out.println("PPM1: " + PPM1.dump());
        System.out.println("PPM2: " + PPM2.dump());

        Rectification rectification = new Rectification();
        rectification.doRectification(PPM1, PPM2);
    }
}
