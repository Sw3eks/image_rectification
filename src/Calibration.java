import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.List;

import static org.opencv.calib3d.Calib3d.*;

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
                showImage(image);
            }
        }
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
