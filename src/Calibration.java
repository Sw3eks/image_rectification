import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.calib3d.Calib3d.*;
import static org.opencv.core.CvType.CV_64F;
import static org.opencv.highgui.HighGui.*;

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
        namedWindow("Webcam", WINDOW_AUTOSIZE);

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
                    case ' ':
                        if (found) {
                            this.imagePoints.add(imageCorners);
                            imageCorners = new MatOfPoint2f();
                            this.objectPoints.add(obj);
                        }
                        break;
                    case 13:
                        if (objectPoints.size() > 15) {
                            cameraCalibration();
                        }
                        break;
                    case 27:
                        return;
                    default:
                        break;
                }
            }
        }

    }

    public void cameraCalibration() {

        //getChessBoardCorners(calibrationImages, checkerBoardImageSpacePoints, false);

        createKnownBoardPosition();

        List<Mat> rVectors = new ArrayList<>();
        List<Mat> tVectors = new ArrayList<>();

        intrinsic.put(0, 0, 1);
        intrinsic.put(1, 1, 1);

        distCoeffs = Mat.zeros(8, 1, CV_64F);

        calibrateCamera(objectPoints, imagePoints, chessboardDimensions, intrinsic, distCoeffs, rVectors, tVectors);
    }

    public boolean saveCameraCalibration(String name, Mat cameraMatrix, Mat distanceCoefficients) {
        // ofstream outStream(name);
//TBD
        return false;
    }

    // Method to showImages by A. Siebert -> Skript
    private void showImage(Mat matrix) {
        MatOfByte matOfByte = new MatOfByte();  // subclass of org.opencv.core.Mat
        Imgcodecs.imencode(".png", matrix, matOfByte);
        byte[] byteArray = matOfByte.toArray();
        BufferedImage bufImage = null;  // subclass of java.awt.image
        try {
            InputStream inStream = new ByteArrayInputStream(byteArray);
            bufImage = ImageIO.read(inStream);
        } catch (Exception exc) {
            exc.printStackTrace();
        }
        JFrame frame = new JFrame("Looking for Corners");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        JLabel imageLabel = null;
        if (bufImage != null) {
            imageLabel = new JLabel(new ImageIcon(bufImage));
        }
        frame.getContentPane().add(imageLabel);
        frame.setLocationRelativeTo(null);
        frame.pack();
        frame.setVisible(true);
    }
}
