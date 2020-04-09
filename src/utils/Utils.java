package utils;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.opencv.calib3d.Calib3d.*;
import static org.opencv.core.Core.NORM_HAMMING;
import static org.opencv.core.CvType.CV_32F;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;
import static org.opencv.imgproc.Imgproc.FILLED;
import static org.opencv.imgproc.Imgproc.line;

public class Utils {

    private MatOfPoint2f good_matches_1;
    private MatOfPoint2f good_matches_2;
    private Mat imageOne;
    private Mat imageTwo;

    public List<Mat> computeEpiLines(Mat firstImage, Mat secondImage) {
        this.imageOne = firstImage;
        this.imageTwo = secondImage;

        good_matches_1 = new MatOfPoint2f();
        good_matches_2 = new MatOfPoint2f();
        Mat fund_mat = fundamentalMat();

        Mat lines_1 = new Mat();
        Mat lines_2 = new Mat();
        computeCorrespondEpilines(good_matches_1, 1, fund_mat, lines_2);
        computeCorrespondEpilines(good_matches_2, 2, fund_mat, lines_1);


//        Mat img1WithLines = drawLines(imageOne, lines_1);
//        Mat img2WithLines = drawLines(imageTwo, lines_2);
        drawEpilines(lines_1, lines_2);

        imwrite("./res/calibration/test1.jpg", imageOne);
        imwrite("./res/calibration/test2.jpg", imageTwo);

        // The epipole is the left-null vector of F
        Mat epi_mat = new Mat();
        Mat dest_mat = new Mat();
        //Core.solve(fund_mat, epi_mat, dest_mat, DECOMP_SVD);

        return Arrays.asList(lines_1, lines_2);
    }

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

        // Define detector
        if (det_id.equals("ORB")) {
            // Declare ORB detector
            detector = ORB.create(
                    500,
                    1.2f,
                    4,
                    21,
                    0,
                    2,
                    ORB.HARRIS_SCORE,
                    21,
                    20
            );
        } else {
            // Declare BRISK and BRISK detectors
            detector = BRISK.create(
                    30,   // thresh = 30
                    3,    // octaves = 3
                    1.0f  // patternScale = 1.0f
            );
        }

        // Declare array for storing the descriptors
        Mat descriptors = new Mat();

        // Detect and compute!
        detector.detect(image, keyPoints1);
        detector.compute(image, keyPoints1, descriptors);

        return descriptors;
    }

    private static Mat drawLines(Mat image1, Mat lines1) {
        Mat resultImg = new Mat();
        image1.copyTo(resultImg);
        //Imgproc.cvtColor(image1, resultImg, Imgproc.COLOR_BGR2GRAY);
        int epiLinesCount = lines1.rows();

        double a, b, c;

        for (int line = 0; line < epiLinesCount; line++) {
            a = lines1.get(line, 0)[0];
            b = lines1.get(line, 0)[1];
            c = lines1.get(line, 0)[2];

            int x0 = 0;
            int y0 = (int) (-(c + a * x0) / b);
            int x1 = resultImg.cols() / 2;
            int y1 = (int) (-(c + a * x1) / b);

            Point p1 = new Point(x0, y0);
            Point p2 = new Point(x1, y1);
            Scalar color = new Scalar(255, 0, 0);
            line(resultImg, p1, p2, color);

        }
        for (int line = 0; line < epiLinesCount; line++) {
            a = lines1.get(line, 0)[0];
            b = lines1.get(line, 0)[1];
            c = lines1.get(line, 0)[2];

            int x0 = resultImg.cols() / 2;
            int y0 = (int) (-(c + a * x0) / b);
            int x1 = resultImg.cols();
            int y1 = (int) (-(c + a * x1) / b);

            Point p1 = new Point(x0, y0);
            Point p2 = new Point(x1, y1);
            Scalar color = new Scalar(255, 0, 0);
            line(resultImg, p1, p2, color);

        }
        return resultImg;
    }

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
                        color
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
                        color
                );

                Imgproc.circle(imageTwo,
                        new Point(good_matches_2.toArray()[line].x, good_matches_2.toArray()[line].y),
                        4,
                        color,
                        FILLED);
            }
        }
    }
}
