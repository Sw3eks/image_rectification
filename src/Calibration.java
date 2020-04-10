import models.RectificationModel;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
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
                        imwrite("./res/calibration/testbilder" + objectPoints.size() + ".jpg", frame);
                        System.out.println("found");
                        if (found) {
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

        createKnownBoardPosition();

        List<Mat> rVectors = new ArrayList<>();
        List<Mat> tVectors = new ArrayList<>();

        intrinsic.put(0, 0, 1);
        intrinsic.put(1, 1, 1);

        distCoeffs = Mat.zeros(5, 1, CV_64F);

        double result = calibrateCamera(objectPoints, imagePoints, chessboardDimensions, intrinsic, distCoeffs, rVectors, tVectors);
        calculatePPM(rVectors, tVectors);
        System.out.println("Result: " + result);
    }

    public boolean saveCameraCalibration() {
        FileWriter fStream = null;
        try {
            fStream = new FileWriter("out_1.txt", true);
            BufferedWriter out = new BufferedWriter(fStream);

            out.write(intrinsic.rows() + "\n");
            out.write(intrinsic.cols() + "\n");

            for (int r = 0; r < intrinsic.rows(); r++) {
                for (int c = 0; c < intrinsic.cols(); c++) {
                    out.write(intrinsic.get(r, c)[0] + "\n");
                }
            }

            out.write("\n" + distCoeffs.rows() + "\n");
            out.write(distCoeffs.cols() + "\n");

            for (int r = 0; r < distCoeffs.rows(); r++) {
                for (int c = 0; c < distCoeffs.cols(); c++) {
                    out.write(distCoeffs.get(r, c)[0] + "\n");
                }
            }
//            out.write("Matrix: " + intrinsic.dump());
//            out.write("\nDist: " + distCoeffs.dump());
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

    public List<Mat> loadCameraCalibration(Mat cameraMatrix, Mat distCoeffs) {
        FileReader reader;
        try {
            reader = new FileReader("out.txt");
            BufferedReader in = new BufferedReader(reader);
            int rows = Integer.parseInt(in.readLine());
            int columns = Integer.parseInt(in.readLine());

            cameraMatrix = new Mat(new Size(rows, columns), CV_64F);

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    double read;
                    read = Double.parseDouble(in.readLine());
                    cameraMatrix.put(r, c, read);
                }
            }
            System.out.println("CameraMatrix: " + cameraMatrix.dump());

            in.readLine(); // read the empty line

            rows = Integer.parseInt(in.readLine());
            columns = Integer.parseInt(in.readLine());

            distCoeffs = Mat.zeros(new Size(rows, columns), CV_64F);

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    double read;
                    read = Double.parseDouble(in.readLine());
                    distCoeffs.put(r, c, read);
                }
            }
            System.out.println("DistCoeffs: " + distCoeffs.dump());

        } catch (Exception e) {
            System.out.println("Exception: " + e.getMessage());
            Logger.getLogger(getClass().getName()).log(Level.SEVERE, null, e);
        }

        return Arrays.asList(cameraMatrix, distCoeffs);
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

        boolean result = savePPM(PPM1, PPM2);
        if (result) {
            System.out.println("Saved PPM!");
            System.out.println("PPM1: " + PPM1.dump());
            System.out.println("PPM2: " + PPM2.dump());
        }

        Mat calibration_image_1 = Imgcodecs.imread("./res/calibration/calib0.jpg");
        Mat calibration_image_2 = Imgcodecs.imread("./res/calibration/calib1.jpg");
        Mat undistortedImage1 = new Mat();
        Mat undistortedImage2 = new Mat();
        Calib3d.undistort(calibration_image_1, undistortedImage1, intrinsic, distCoeffs);
        Calib3d.undistort(calibration_image_2, undistortedImage2, intrinsic, distCoeffs);
        imwrite("./res/calibration/undistort1.jpg", undistortedImage1);
        imwrite("./res/calibration/undistort2.jpg", undistortedImage2);
        MatOfPoint2f undistortedPoints1 = new MatOfPoint2f();
        MatOfPoint2f undistortedPoints2 = new MatOfPoint2f();
        Calib3d.undistortPoints((MatOfPoint2f) imagePoints.get(0), undistortedPoints1, intrinsic, distCoeffs, new Mat(), intrinsic);
        Calib3d.undistortPoints((MatOfPoint2f) imagePoints.get(1), undistortedPoints2, intrinsic, distCoeffs, new Mat(), intrinsic);

        Rectification rectification = new Rectification();
        RectificationModel rectificationModel = rectification.doRectification(PPM1, PPM2, imagePoints);

        rectification.drawEpipolarLines(
                rectificationModel.getRectifiedImage1(),
                rectificationModel.getRectifiedImage2(),
                rectificationModel.getRectifiedImagePoints1(),
                rectificationModel.getRectifiedImagePoints2());
    }

    public boolean savePPM(Mat PPM1, Mat PPM2) {
        FileWriter fStream = null;
        try {
            fStream = new FileWriter("ppm_1.txt", true);
            BufferedWriter out = new BufferedWriter(fStream);


            out.write(PPM1.rows() + "\n");
            out.write(PPM1.cols() + "\n");

            for (int r = 0; r < PPM1.rows(); r++) {
                for (int c = 0; c < PPM1.cols(); c++) {
                    out.write(PPM1.get(r, c)[0] + "\n");
                }
            }

            out.write("\n" + PPM2.rows() + "\n");
            out.write(PPM2.cols() + "\n");

            for (int r = 0; r < PPM2.rows(); r++) {
                for (int c = 0; c < PPM2.cols(); c++) {
                    out.write(PPM2.get(r, c)[0] + "\n");
                }
            }
            //out.write("PPM1: " + PPM1.dump());
            //out.write("\nPPM2: " + PPM2.dump());

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
}
