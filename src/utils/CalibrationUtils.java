package utils;

import org.opencv.core.Mat;
import org.opencv.core.Size;

import java.io.*;
import java.util.Arrays;
import java.util.List;

import static org.opencv.core.CvType.CV_64F;

public class CalibrationUtils {

    public static boolean savePPM(Mat PPM1, Mat PPM2) {
        FileWriter fStream = null;
        try {
            fStream = new FileWriter("ppm_1.txt");
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

    public static List<Mat> loadPPM(Mat PPM1, Mat PPM2) {
        FileReader reader;
        try {
            reader = new FileReader("ppm_2.txt");
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

    public static boolean saveCameraCalibration(Mat cameraMatrix, Mat distCoeffs) {
        FileWriter fStream = null;
        try {
            fStream = new FileWriter("out_1.txt");
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

    public static List<Mat> loadCameraCalibration(Mat cameraMatrix, Mat distCoeffs) {
        FileReader reader;
        try {
            reader = new FileReader("out_2.txt");
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
