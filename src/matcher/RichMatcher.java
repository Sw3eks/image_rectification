package matcher;

import org.opencv.core.*;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.features2d.ORB;

import java.util.ArrayList;
import java.util.List;

public class RichMatcher extends Matcher {

    public RichMatcher(Mat img1, Mat img2) {
        super(img1, img2);
        setDetector(FastFeatureDetector.create());
        setDescriptor(ORB.create());
        setMatcher(DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING));
        detect();
        matchDescriptors();
    }

    @Override
    public void match() {
        List<DMatch> goodMatches = new ArrayList<>();

        for (MatOfDMatch matOfDMatch : matches1) {
            DMatch match = matOfDMatch.toArray()[0];
            goodMatches.add(match);
        }
        MatOfPoint2f points1 = new MatOfPoint2f();
        MatOfPoint2f points2 = new MatOfPoint2f();
        sortedKeyPointsToMatOfPoint2f(getKeyPoints1(), getKeyPoints2(), goodMatches, points1, points2);
        this.goodMatches = ransacTest(points1, points2, goodMatches);
        sortedKeyPointsToMatOfPoint2f(getKeyPoints1(), getKeyPoints2(), this.goodMatches, points1, points2);
        setMatchPoints1(points1);
        setMatchPoints2(points2);
        setGoodMatches(this.goodMatches);
    }

    public void sortedKeyPointsToMatOfPoint2f(MatOfKeyPoint srcPoints, MatOfKeyPoint dstPoints,
                                              List<DMatch> matches, MatOfPoint2f srcFilteredMat, MatOfPoint2f dstFilteredMat) {
        KeyPoint[] srcArray = srcPoints.toArray();
        KeyPoint[] dstArray = dstPoints.toArray();
        ArrayList<Point> srcFilteredPoints = new ArrayList<>();
        ArrayList<Point> dstFilteredPoints = new ArrayList<>();

        for (DMatch match : matches) {
            srcFilteredPoints.add(srcArray[match.queryIdx].pt);
            dstFilteredPoints.add(dstArray[match.trainIdx].pt);
        }

        srcFilteredMat.fromList(srcFilteredPoints);
        dstFilteredMat.fromList(dstFilteredPoints);
    }

}
