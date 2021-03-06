package utils;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.opencv.calib3d.Calib3d.*;
import static org.opencv.core.Core.NORM_HAMMING;
import static org.opencv.core.CvType.CV_32F;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;
import static org.opencv.imgproc.Imgproc.FILLED;

public class Utils {

    private MatOfPoint2f good_matches_1;
    private MatOfPoint2f good_matches_2;
    private Mat imageOne;
    private Mat imageTwo;

    /**
     * Draws computed Epilines on the given images
     * Detects feature Points if point Inputs are null
     *
     * @param firstImage        image 1 for calculation
     * @param secondImage       image 2 for calculation
     * @param firstImagePoints  points for image 1
     * @param secondImagePoints points for image 2
     * @return List of matches and images with epiLines
     */
    public List<Mat> computeEpiLines(Mat firstImage, Mat secondImage, Mat firstImagePoints, Mat secondImagePoints) {
        this.imageOne = firstImage;
        this.imageTwo = secondImage;

        if (firstImagePoints != null && secondImagePoints != null) {
            good_matches_1 = new MatOfPoint2f(firstImagePoints);
            good_matches_2 = new MatOfPoint2f(secondImagePoints);
        } else {
            good_matches_1 = new MatOfPoint2f();
            good_matches_2 = new MatOfPoint2f();
        }
        Mat fund_mat = fundamentalMat();

        Mat lines_1 = new Mat();
        Mat lines_2 = new Mat();
        computeCorrespondEpilines(good_matches_1, 1, fund_mat, lines_2);
        computeCorrespondEpilines(good_matches_2, 2, fund_mat, lines_1);

        drawEpilines(lines_1, lines_2);

        return Arrays.asList(good_matches_1, good_matches_2, imageOne, imageTwo);
    }

    /**
     * Computes the fundamental Matrix with the 8Point algorithm
     * if matches are empty, detects matches and then computes with ransac
     *
     * @return computed fundamental matrix
     */
    private Mat fundamentalMat() {

        List<MatOfPoint2f> matches;
        Mat F;
        Mat mask = new Mat();

        Mat first;
        Mat second;
        int flag = FM_8POINT;

        if (good_matches_1.empty() && good_matches_2.empty()) {
            matches = match(DescriptorMatcher.BRUTEFORCE);
            first = matches.get(0);
            second = matches.get(1);
            flag |= FM_RANSAC;
        } else {
            first = good_matches_1;
            second = good_matches_2;
        }


        F = Calib3d.findFundamentalMat(new MatOfPoint2f(first), new MatOfPoint2f(second), flag, 1., 0.99, mask);

        Mat final_1 = new Mat();
        Mat final_2 = new Mat();

        for (int row = 0; row < mask.rows(); row++) {
            if (mask.get(row, 0)[0] == 1.0) {
                final_1.push_back(first);
                final_2.push_back(second);
            }
        }

        good_matches_1 = new MatOfPoint2f(final_1);
        good_matches_2 = new MatOfPoint2f(final_2);

        return F;
    }

    private List<MatOfPoint2f> match(int descriptor) {
        // 1 - Get keypoints and its descriptors in both images
        MatOfKeyPoint keyPoints1 = new MatOfKeyPoint();
        MatOfKeyPoint keyPoints2 = new MatOfKeyPoint();
        Mat descriptors1;
        Mat descriptors2;

        descriptors1 = detectFeatures(imageOne, "ORB", keyPoints1);
        descriptors2 = detectFeatures(imageTwo, "ORB", keyPoints2);

        // 2 - Match both descriptors using required detector
        // Declare the matcher
        DescriptorMatcher matcher;

        // Define the matcher
        if (descriptor == DescriptorMatcher.BRUTEFORCE) {
            // For ORB and BRISK descriptors, NORM_HAMMING should be used.
            // See http://sl.ugr.es/norm_ORB_BRISK
            matcher = BFMatcher.create(NORM_HAMMING, true);
        } else {
            matcher = FlannBasedMatcher.create();
            // FlannBased Matcher needs CV_32F descriptors
            // See http://sl.ugr.es/FlannBase_32F
            if (descriptors1.type() != CV_32F || descriptors2.type() != CV_32F) {
                descriptors1.convertTo(descriptors1, CV_32F);
                descriptors2.convertTo(descriptors2, CV_32F);
            }
        }

        // Match!
        MatOfDMatch matches = new MatOfDMatch();
        matcher.match(descriptors1, descriptors2, matches);

        DMatch[] dm = matches.toArray();
        // 3 - Create MatOfKeyPoint following obtained matches
        MatOfPoint2f good_matches_1 = new MatOfPoint2f();
        MatOfPoint2f good_matches_2 = new MatOfPoint2f();

        KeyPoint[] srcArray = keyPoints1.toArray();
        KeyPoint[] dstArray = keyPoints2.toArray();

        List<Point> srcFilteredPoints = new ArrayList<>(dm.length);
        List<Point> dstFilteredPoints = new ArrayList<>(dm.length);

        for (int i = 0; i < dm.length; i++) {
            DMatch dmO = dm[i];
            srcFilteredPoints.add(srcArray[dmO.queryIdx].pt);
            dstFilteredPoints.add(dstArray[dmO.trainIdx].pt);

            i++;
        }

        good_matches_1.fromList(srcFilteredPoints);
        good_matches_2.fromList(dstFilteredPoints);

        return Arrays.asList(good_matches_1, good_matches_2);
    }

    private Mat detectFeatures(Mat image, String det_id, MatOfKeyPoint keyPoints1) {
        // Declare detector
        Feature2D detector;
        Mat descriptors = new Mat();

        // Define FAST detector
        detector = FastFeatureDetector.create();
        // Detect and compute with ORB (FAST cannot compute)!
        detector.detect(image, keyPoints1);
        detector = ORB.create(500, 1.2f, 4, 21, 0, 2, ORB.HARRIS_SCORE, 21, 20);
        detector.compute(image, keyPoints1, descriptors);
        if (det_id.equals("BRISK")) {
            // Declare BRISK and BRISK detectors
            detector = BRISK.create(
                    30,   // thresh = 30
                    3,    // octaves = 3
                    1.0f  // patternScale = 1.0f
            );
            detector.detect(image, keyPoints1);
            detector.compute(image, keyPoints1, descriptors);
        }
        return descriptors;
    }

    /**
     * Draws the given lines on corresponding images
     *
     * @param lines_1 lines for image 1
     * @param lines_2 lines for image 2
     */
    void drawEpilines(Mat lines_1, Mat lines_2) {

        int epiLinesCount = lines_1.rows();

        for (int line = 0; line < epiLinesCount; line++) {
            // dont draw too many lines
            if (line % (epiLinesCount / 150) == 0) {
                Scalar color = new Scalar(Math.random() * 255,
                        Math.random() * 255,
                        Math.random() * 255);
                Imgproc.line(imageOne,
                        new Point(0,
                                -lines_1.get(line, 0)[2] / lines_1.get(line, 0)[1]),
                        new Point(imageOne.cols(),
                                -(lines_1.get(line, 0)[2] + lines_1.get(line, 0)[0] * imageOne.cols()) / lines_1.get(line, 0)[1]),
                        color, 1
                );
                Imgproc.circle(imageOne,
                        new Point(good_matches_1.toArray()[line].x, good_matches_1.toArray()[line].y),
                        4,
                        color,
                        FILLED);

                Imgproc.line(imageTwo,
                        new Point(0,
                                -lines_2.get(line, 0)[2] / lines_2.get(line, 0)[1]),
                        new Point(imageTwo.cols(),
                                -(lines_2.get(line, 0)[2] + lines_2.get(line, 0)[0] * imageTwo.cols()) / lines_2.get(line, 0)[1]),
                        color, 1
                );

                Imgproc.circle(imageTwo,
                        new Point(good_matches_2.toArray()[line].x, good_matches_2.toArray()[line].y),
                        4,
                        color,
                        FILLED);
            }
        }
    }

    /**
     * Calculates projection matrices for the given indices
     * Results are saved in a txt file with the given fileName
     *
     * @param fileName  name for the saved file
     * @param rVectors  rotation vectors
     * @param tVectors  translation vectors
     * @param intrinsic camera matrix
     * @param index1    index for image 1
     * @param index2    index for image 2
     */
    public void calculatePPM(String fileName, List<Mat> rVectors, List<Mat> tVectors, Mat intrinsic, int index1, int index2) {
        Mat r1 = rVectors.get(index1);
        Mat r2 = rVectors.get(index2);
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

        Mat PPM1 = new Mat();
        Mat PPM2 = new Mat();
        Core.gemm(intrinsic, Rt1, 1, new Mat(), 0, PPM1, 0);
        Core.gemm(intrinsic, Rt2, 1, new Mat(), 0, PPM2, 0);

        boolean result = CalibrationUtils.savePPM(fileName, PPM1, PPM2);
        if (result) {
            System.out.println("Saved PPM!");
            System.out.println("PPM1: " + PPM1.dump());
            System.out.println("PPM2: " + PPM2.dump());
        }
    }

    public void undistortImages(Mat image, Mat intrinsic, Mat distCoeffs, int index) {
        Mat undistortedImage = new Mat();
        Calib3d.undistort(image, undistortedImage, intrinsic, distCoeffs);
        imwrite("./res/output/undistorted/undistorted" + index + ".jpg", undistortedImage);
    }


    /**
     * Utility function which draws 3 lines on the dst image connecting
     * points of image 1 and image 2
     * result is saved in /res/output/epipolar
     *
     * @param imageOne     image 1 for drawing
     * @param imageTwo     image 2 for drawing
     * @param imagePoints1 points for image 1
     * @param imagePoints2 points for image 2
     */
    public void mergeImagesAndDrawLine(Mat imageOne, Mat imageTwo, Mat imagePoints1, Mat imagePoints2) {
        Mat dst = new Mat();
        List<Mat> src = Arrays.asList(imageOne, imageTwo);
        Core.hconcat(src, dst);

        Scalar color = new Scalar(Math.random() * 255,
                Math.random() * 255,
                Math.random() * 255);
        Imgproc.line(dst,
                new Point(imagePoints1.get(0, 0)[0],
                        imagePoints1.get(0, 0)[1]),
                new Point(1000 + imagePoints2.get(0, 0)[0],
                        imagePoints2.get(0, 0)[1]),
                color, 2
        );
        Imgproc.circle(dst,
                new Point(imagePoints1.get(0, 0)[0], imagePoints1.get(0, 0)[1]),
                4,
                color,
                FILLED);
        Imgproc.circle(dst,
                new Point(1000 + imagePoints2.get(0, 0)[0], imagePoints2.get(0, 0)[1]),
                4,
                color,
                FILLED);

        Imgproc.line(dst,
                new Point(imagePoints1.get(35, 0)[0],
                        imagePoints1.get(35, 0)[1]),
                new Point(1000 + imagePoints2.get(35, 0)[0],
                        imagePoints2.get(35, 0)[1]),
                color, 2
        );
        Imgproc.circle(dst,
                new Point(imagePoints1.get(35, 0)[0], imagePoints1.get(35, 0)[1]),
                4,
                color,
                FILLED);
        Imgproc.circle(dst,
                new Point(1000 + imagePoints2.get(35, 0)[0], imagePoints2.get(35, 0)[1]),
                4,
                color,
                FILLED);

        Imgproc.line(dst,
                new Point(imagePoints1.get(45, 0)[0],
                        imagePoints1.get(45, 0)[1]),
                new Point(1000 + imagePoints2.get(45, 0)[0],
                        imagePoints2.get(45, 0)[1]),
                color, 2
        );

        Imgproc.circle(dst,
                new Point(imagePoints1.get(45, 0)[0], imagePoints1.get(45, 0)[1]),
                4,
                color,
                FILLED);
        Imgproc.circle(dst,
                new Point(1000 + imagePoints2.get(45, 0)[0], imagePoints2.get(45, 0)[1]),
                4,
                color,
                FILLED);
        imwrite("./res/output/epipolar/combined_epipolar.jpg", dst);
    }
}
