package utils;

import org.opencv.core.Mat;
import org.opencv.core.Size;

import java.io.*;
import java.util.Arrays;
import java.util.List;

import static org.opencv.core.CvType.CV_64F;

public class CalibrationUtils {

    /**
     * Util function to save Camera Projection Matrices
     *
     * @param fileName name of the file with the saved values
     * @param PPM1     input projection matrix 1
     * @param PPM2     input projection matrix 2
     * @return result whether saving was successful
     */
    public static boolean savePPM(String fileName, Mat PPM1, Mat PPM2) {
        FileWriter fStream = null;
        try {
            fStream = new FileWriter(fileName);
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

            //Close the output stream
            out.close();
            return true;
        } catch (IOException e) {
            System.out.println("Exception: " + e.getMessage());
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

    /**
     * Util function to load Camera Projection Matrices
     *
     * @param fileName name of the file with the saved values
     * @param PPM1     projection matrix 1 to be loaded
     * @param PPM2     projection matrix 2 to be loaded
     * @return List of the loaded projection matrices
     */
    public static List<Mat> loadPPM(String fileName, Mat PPM1, Mat PPM2) {
        FileReader reader;
        try {
            reader = new FileReader(fileName);
            BufferedReader in = new BufferedReader(reader);
            int rows = Integer.parseInt(in.readLine());
            int columns = Integer.parseInt(in.readLine());

            PPM1 = Mat.zeros(rows, columns, CV_64F);

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    double read;
                    read = Double.parseDouble(in.readLine());
                    PPM1.put(r, c, read);
                }
            }
            System.out.println("PPM1: " + PPM1.dump());

            in.readLine(); // read the empty line

            rows = Integer.parseInt(in.readLine());
            columns = Integer.parseInt(in.readLine());

            PPM2 = Mat.zeros(rows, columns, CV_64F);

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    double read;
                    read = Double.parseDouble(in.readLine());
                    PPM2.put(r, c, read);
                }
            }
            System.out.println("PPM2: " + PPM2.dump());

        } catch (Exception e) {
            System.out.println("Exception: " + e.getMessage());
        }

        return Arrays.asList(PPM1, PPM2);
    }

    /**
     * Util function to save camera calibration parameters
     *
     * @param fileName     name of the file with the saved values
     * @param cameraMatrix input camera matrix
     * @param distCoeffs   input distortion coefficients
     * @return result whether saving was successful
     */
    public static boolean saveCameraCalibration(String fileName, Mat cameraMatrix, Mat distCoeffs) {
        FileWriter fStream = null;
        try {
            fStream = new FileWriter(fileName);
            BufferedWriter out = new BufferedWriter(fStream);

            out.write(cameraMatrix.rows() + "\n");
            out.write(cameraMatrix.cols() + "\n");

            for (int r = 0; r < cameraMatrix.rows(); r++) {
                for (int c = 0; c < cameraMatrix.cols(); c++) {
                    out.write(cameraMatrix.get(r, c)[0] + "\n");
                }
            }

            out.write("\n" + distCoeffs.rows() + "\n");
            out.write(distCoeffs.cols() + "\n");

            for (int r = 0; r < distCoeffs.rows(); r++) {
                for (int c = 0; c < distCoeffs.cols(); c++) {
                    out.write(distCoeffs.get(r, c)[0] + "\n");
                }
            }
            //Close the output stream
            out.close();
            return true;
        } catch (IOException e) {
            System.out.println("Exception: " + e.getMessage());
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

    /**
     * Util function to load camera calibration parameters
     *
     * @param fileName     name of the file with the saved values
     * @param cameraMatrix camera matrix to be loaded
     * @param distCoeffs   distortion coefficients to be loaded
     * @return List of the loaded camera calibration parameters
     */
    public static List<Mat> loadCameraCalibration(String fileName, Mat cameraMatrix, Mat distCoeffs) {
        FileReader reader;
        try {
            reader = new FileReader(fileName);
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

            distCoeffs = Mat.zeros(rows, columns, CV_64F);

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
        }

        return Arrays.asList(cameraMatrix, distCoeffs);
    }
}
