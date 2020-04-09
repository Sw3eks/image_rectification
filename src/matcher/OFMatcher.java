package matcher;

import org.opencv.core.*;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.features2d.ORB;
import org.opencv.utils.Converters;
import org.opencv.video.Video;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class OFMatcher extends Matcher {

    public OFMatcher(Mat img1, Mat img2) {
        super(img1, img2);
    }

    public void match() {

        setDetector(FastFeatureDetector.create());
        setDescriptor(ORB.create());
        detect();

        MatOfPoint2f leftPointsMat = keyPointsToMatOfPoint2f(getKeyPoints1());
        MatOfPoint2f rightPointsMat = new MatOfPoint2f();

        //calculate leftPoints movement
        MatOfByte status = new MatOfByte();
        MatOfFloat err = new MatOfFloat();
        Video.calcOpticalFlowPyrLK(getImg1(), getImg2(), leftPointsMat, rightPointsMat, status, err);

        List<Point> leftPoints = new ArrayList<>();
        List<Point> rightPoints = new ArrayList<>();
        Converters.Mat_to_vector_Point2f(leftPointsMat, leftPoints);
        Converters.Mat_to_vector_Point2f(rightPointsMat, rightPoints);

        //filter high error points and keep original index
        List<Point> rightPointsToFind = new ArrayList<>();
        List<Integer> rightPointsToFindBackIndex = new ArrayList<>();
        for (int i = 0; i < status.rows(); i++) {
            if (status.get(i, 0)[0] == 1 && err.get(i, 0)[0] < 12) {
                rightPointsToFindBackIndex.add(i);
                rightPointsToFind.add(rightPoints.get(i));
            }
        }

        //match rightPoints found by OF to its features
        Mat rightPointsToFindMat = Converters.vector_Point2f_to_Mat(rightPointsToFind).reshape(1, rightPointsToFind.size());
        Mat rightPointsFeatures = keyPointsToMatOfPoint2f(getKeyPoints2()).reshape(1, getKeyPoints2().rows());

        DescriptorMatcher bfMatcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        List<MatOfDMatch> bfMatches = new ArrayList<>();
        bfMatcher.radiusMatch(rightPointsToFindMat, rightPointsFeatures, bfMatches, 2.0f);

        this.goodMatches = ofRatioTest(bfMatches, rightPointsToFindBackIndex);
        MatOfPoint2f points1 = new MatOfPoint2f();
        MatOfPoint2f points2 = new MatOfPoint2f();


        sortedKeyPointsToMatOfPoint2f(getKeyPoints1(), getKeyPoints2(), getGoodMatches(), points1, points2);
        setMatchPoints1(points1);
        setMatchPoints2(points2);

        this.goodMatches = ransacTest(points1, points2, this.goodMatches);
        sortedKeyPointsToMatOfPoint2f(getKeyPoints1(), getKeyPoints2(), getGoodMatches(), points1, points2);
        setMatchPoints1(points1);
        setMatchPoints2(points2);
    }

    private List<DMatch> ofRatioTest(List<MatOfDMatch> matches, List<Integer> originalIndexes) {
        List<DMatch> goodMatchesWithOriginalIndexes = new ArrayList<>();
        MatOfDMatch match;
        Iterator<MatOfDMatch> it = matches.iterator();
        double ratio;

        int queryIdx, trainIdx, oldQueryIdx;
        while (it.hasNext()) {
            match = it.next();
            if (match.rows() == 1) {
                queryIdx = (int) getQueryIdxFromMatOfDMatch(match);
                oldQueryIdx = originalIndexes.get(queryIdx);
                trainIdx = (int) getTrainIdxFromMatOfDMatch(match);

                goodMatchesWithOriginalIndexes.add(new DMatch(
                        oldQueryIdx,
                        trainIdx,
                        (float) distanceFromKNNMatch(match, 0)
                ));
            } else if (match.rows() > 1) {
                ratio = distanceFromKNNMatch(match, 0) / distanceFromKNNMatch(match, 1);
                if (ratio < 0.7) {
                    queryIdx = (int) getQueryIdxFromMatOfDMatch(match);
                    oldQueryIdx = originalIndexes.get(queryIdx);
                    trainIdx = (int) getTrainIdxFromMatOfDMatch(match);

                    goodMatchesWithOriginalIndexes.add(new DMatch(
                            oldQueryIdx,
                            trainIdx,
                            (float) distanceFromKNNMatch(match, 0)
                    ));
                }
            }
        }
        System.out.println("Before ratio-test: " + matches.size() + " After: " + goodMatchesWithOriginalIndexes.size());
        return goodMatchesWithOriginalIndexes;

    }

    public static MatOfPoint2f keyPointsToMatOfPoint2f(MatOfKeyPoint keyPoints) {


        List<Point> pointsList = new ArrayList<>();
        KeyPoint[] kpArr = keyPoints.toArray();
        for (KeyPoint keyPoint : kpArr) {
            pointsList.add(keyPoint.pt);
        }
        MatOfPoint2f points = new MatOfPoint2f();
        points.fromList(pointsList);

        return points;
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

    public static double getQueryIdxFromMatOfDMatch(MatOfDMatch matOfDMatch) {
        return matOfDMatch.get(0, 0)[0];
    }

    public static double getTrainIdxFromMatOfDMatch(MatOfDMatch matOfDMatch) {
        return matOfDMatch.get(0, 0)[1];
    }

    public static double distanceFromKNNMatch(MatOfDMatch matOfDMatch, int neighbourIndex) {
        return matOfDMatch.get(neighbourIndex, 0)[3];
    }

}
