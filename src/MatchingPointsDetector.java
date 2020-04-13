import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.ORB;

import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;

public class MatchingPointsDetector {

    MatOfKeyPoint srcKeyPoints, dstKeyPoints;
    MatOfPoint2f srcSortedGoodPoints, dstSortedGoodPoints;
    MatOfDMatch matches;

    private Mat img1, img2;

    public MatchingPointsDetector(Mat img1, Mat img2) {
        this.img1 = img1;
        this.img2 = img2;
    }

    public void matchImages() {
        MatOfKeyPoint keyPoints1 = new MatOfKeyPoint();
        MatOfKeyPoint keyPoints2 = new MatOfKeyPoint();
        Mat descriptors1 = new Mat();
        Mat descriptors2 = new Mat();
        detectKeyPoints(keyPoints1, keyPoints2, descriptors1, descriptors2);

        this.srcKeyPoints = keyPoints1;
        this.dstKeyPoints = keyPoints2;

        descriptors1.convertTo(descriptors1, CV_32F);
        descriptors2.convertTo(descriptors2, CV_32F);
        DescriptorMatcher descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        List<MatOfDMatch> matches = new ArrayList<>();
        descriptorMatcher.knnMatch(descriptors1, descriptors2, matches, 1);

        MatOfPoint2f srcPoints = new MatOfPoint2f();
        MatOfPoint2f dstPoints = new MatOfPoint2f();
        convertUnsortedKeyPointsIntoPoint2f(keyPoints1, keyPoints2, matches, srcPoints, dstPoints);

        Mat mask = new Mat();

        if (srcPoints.toList().isEmpty() || dstPoints.toList().isEmpty()) {
            System.out.println("No matches found at all.....");
            return;
        }
        Calib3d.findHomography(srcPoints, dstPoints, Calib3d.RANSAC, 5, mask);

        MatOfDMatch goodMatches = getInliers(mask, matches);
        this.matches = goodMatches;

        MatOfPoint2f srcGoodSortedPoints = new MatOfPoint2f();
        MatOfPoint2f dstGoodSortedPoints = new MatOfPoint2f();
        extractAndSortGoodMatchPoints(keyPoints1, keyPoints2, goodMatches, srcGoodSortedPoints, dstGoodSortedPoints);
        this.srcSortedGoodPoints = srcGoodSortedPoints;
        this.dstSortedGoodPoints = dstGoodSortedPoints;

        Mat outImg = new Mat(img1.rows(), img1.cols() * 2, img1.type());
        Features2d.drawMatches(img1, keyPoints1, img2, keyPoints2, goodMatches, outImg);

        imwrite("./res/output/detect1.jpg", img1);
        imwrite("./res/output/detect2.jpg", img2);
        imwrite("./res/output/detect3.jpg", outImg);

    }

    private void extractAndSortGoodMatchPoints(MatOfKeyPoint srcPoints, MatOfKeyPoint dstPoints, MatOfDMatch goodMatches,
                                               MatOfPoint2f srcFilteredMat, MatOfPoint2f dstFilteredMat) {
        DMatch[] dm = goodMatches.toArray();
        KeyPoint[] srcArray = srcPoints.toArray();
        KeyPoint[] dstArray = dstPoints.toArray();
        System.out.println(dstPoints);

        List<Point> srcFilteredPoints = new ArrayList<>(dm.length);
        List<Point> dstFilteredPoints = new ArrayList<>(dm.length);

        System.out.println(dm.length);
        for (int i = 0; i < dm.length; i++) {
            DMatch dmO = dm[i];
            srcFilteredPoints.add(srcArray[dmO.queryIdx].pt);
            dstFilteredPoints.add(dstArray[dmO.trainIdx].pt);

            i++;
        }
        srcFilteredMat.fromList(srcFilteredPoints);
        dstFilteredMat.fromList(dstFilteredPoints);
    }


    private void detectKeyPoints(MatOfKeyPoint keyPoints1, MatOfKeyPoint keyPoints2, Mat descriptors1, Mat descriptors2) {
        FastFeatureDetector detector = FastFeatureDetector.create();
        Mat mask1 = Mat.zeros(img1.size(), CV_8U); // or new Mask()
        Mat mask2 = Mat.zeros(img1.size(), CV_8U);
        detector.detect(img1, keyPoints1);
        detector.detect(img2, keyPoints2);

        ORB extractor = ORB.create();
        extractor.compute(img1, keyPoints1, descriptors1);
        extractor.compute(img2, keyPoints2, descriptors2);
    }

    private void convertUnsortedKeyPointsIntoPoint2f(MatOfKeyPoint keyPoints1, MatOfKeyPoint keyPoints2, List<MatOfDMatch> matches, MatOfPoint2f srcPoints, MatOfPoint2f dstPoints) {
        List<KeyPoint> kplist1 = keyPoints1.toList();
        List<KeyPoint> kplist2 = keyPoints2.toList();

        ArrayList<Point> pointsList1 = new ArrayList<>();
        ArrayList<Point> pointsList2 = new ArrayList<>();

        for (MatOfDMatch match : matches) {
            pointsList1.add(kplist1.get((int) (match.get(0, 0)[0])).pt);
            pointsList2.add(kplist2.get((int) (match.get(0, 0)[1])).pt);
        }


        srcPoints.fromList(pointsList1);
        dstPoints.fromList(pointsList2);

    }

    private MatOfDMatch getInliers(Mat mask, List<MatOfDMatch> matches) {
        MatOfDMatch goodMatches = new MatOfDMatch();

        for (int row = 0; row < mask.rows(); row++) {
            if (mask.get(row, 0)[0] == 1.0) {
                goodMatches.push_back(matches.get(row));
            }
        }
        return goodMatches;
    }
}
