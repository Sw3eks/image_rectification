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

    public void createKnownBoardPosition(Size boardSize, float squareEdgeLength, MatOfPoint3f corners) {
        for (int i = 0; i < boardSize.height; i++) {
            for (int j = 0; j < boardSize.width; j++) {
                corners.push_back(new MatOfPoint3f(new Point3(j * squareEdgeLength, i * squareEdgeLength, 0.0f)));
            }
        }
    }

    public void getChessBoardCorners(List<Mat> images, MatOfPoint2f allFoundCorners, boolean showResults) {
        for (Mat image : images) {
            MatOfPoint2f pointBuf = new MatOfPoint2f();
            boolean found = findChessboardCorners(image, chessboardDimensions, pointBuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
            if (found) {
                allFoundCorners.push_back(pointBuf);
            }
            if (showResults) {
                drawChessboardCorners(image, chessboardDimensions, pointBuf, found);
                imshow("Looking for Corners", image);
                waitKey(0);
                //showImage(image);
            }
        }
    }

    public int calibrate() {
        Mat frame = new Mat();
        Mat drawToFrame = new Mat();

        Mat cameraMatrix = Mat.eye(3, 3, CV_64F);

        Mat distanceCoefficients;

        List<Mat> savedImages = new ArrayList<>();

        List<MatOfPoint2f> markerCorners, rejectedCandidates = new ArrayList<>();

        VideoCapture vid = new VideoCapture();
        if (!vid.isOpened()) {
            return 0;
        }
        int framesPerSecond = 20;
        namedWindow("Webcam", WINDOW_AUTOSIZE);

        while (vid.read(frame)) {
            MatOfPoint2f foundPoints = new MatOfPoint2f();
            boolean found;

            found = findChessboardCorners(frame, chessboardDimensions, foundPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
            frame.copyTo(drawToFrame);
            drawChessboardCorners(drawToFrame, chessboardDimensions, foundPoints, found);
            if (found) {
                imshow("Webcam", drawToFrame);
            } else {
                imshow("Webcam", frame);
            }
            int character = waitKey(1000 / framesPerSecond);
        }

        return 0;
    }

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
