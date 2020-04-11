package matcher;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.*;

import java.util.ArrayList;
import java.util.List;

import static org.opencv.imgcodecs.Imgcodecs.imwrite;

public abstract class Matcher {
    private final Mat img1;
    private final Mat img2;
    //detector, descriptor, matcher
    private FastFeatureDetector detector;
    private ORB descriptor;
    private DescriptorMatcher matcher;
    //keypoints
    private MatOfKeyPoint keyPoints1;
    private MatOfKeyPoint keyPoints2;
    private MatOfPoint2f matchPoints1;
    private MatOfPoint2f matchPoints2;
    //descriptors
    private Mat descriptors1;
    private Mat descriptors2;
    //matches
    List<MatOfDMatch> matches1;
    List<MatOfDMatch> matches2;
    List<DMatch> goodMatches;
    double epipolarDistance = 1.0;
    double ransacConfidence = 0.99;

    public Matcher(Mat img1, Mat img2) {
        this.img1 = img1;
        this.img2 = img2;

    }

    public abstract List<MatOfPoint2f> match();


    public Mat drawMatchesAndKeyPoints(String path) {
        Mat outImg = new Mat(img1.rows(), img1.cols() * 2, img1.type());
        MatOfDMatch gm = new MatOfDMatch();
        gm.fromList(getGoodMatches());
        Features2d.drawMatches(getImg1(), getKeyPoints1(), img2, getKeyPoints2(), gm, outImg);
        imwrite(path, outImg);
        return outImg;
    }

    protected void matchDescriptors() {
        List<MatOfDMatch> matches1 = new ArrayList<>();
        List<MatOfDMatch> matches2 = new ArrayList<>();

        getMatcher().knnMatch(getDescriptors1(), getDescriptors2(), matches1, 2);
        getMatcher().knnMatch(getDescriptors2(), getDescriptors1(), matches2, 2);

        this.matches1 = matches1;
        this.matches2 = matches2;
    }


    protected List<DMatch> ransacTest(MatOfPoint2f points1, MatOfPoint2f points2, List<DMatch> matches) {
        Mat mask = new Mat();

        Calib3d.findFundamentalMat(points1, points2, Calib3d.RANSAC, 1, ransacConfidence, mask);
        List<DMatch> goodMatches = filterGoodMatches(matches, mask);

        System.out.println("Before ransac: " + matches.size() + " After: " + goodMatches.size());
        return goodMatches;
    }

    public static List<DMatch> filterGoodMatches(List<DMatch> matches, Mat mask) {
        List<DMatch> goodMatches = new ArrayList<>();
        for (int matchIndex = 0; matchIndex < mask.rows(); matchIndex++) {
            if (mask.get(matchIndex, 0)[0] == 1) {
                goodMatches.add(matches.get(matchIndex));
            }
        }

        return goodMatches;
    }

    protected void detect() {
        MatOfKeyPoint keyPoints1 = new MatOfKeyPoint();
        MatOfKeyPoint keyPoints2 = new MatOfKeyPoint();

        Mat descriptors1 = new Mat();
        Mat descriptors2 = new Mat();

        detector.detect(getImg1(), keyPoints1);
        detector.detect(getImg2(), keyPoints2);
        this.keyPoints1 = keyPoints1;
        this.keyPoints2 = keyPoints2;

        descriptor.compute(getImg1(), getKeyPoints1(), descriptors1);
        descriptor.compute(getImg2(), getKeyPoints2(), descriptors2);
        this.descriptors1 = descriptors1;
        this.descriptors2 = descriptors2;

    }

    public Mat getImg1() {
        return img1;
    }

    public Mat getImg2() {
        return img2;
    }

    public DescriptorMatcher getMatcher() {
        return matcher;
    }

    public MatOfKeyPoint getKeyPoints1() {
        return keyPoints1;
    }

    public MatOfKeyPoint getKeyPoints2() {
        return keyPoints2;
    }

    public Mat getDescriptors1() {
        return descriptors1;
    }

    public Mat getDescriptors2() {
        return descriptors2;
    }

    public List<DMatch> getGoodMatches() {
        return goodMatches;
    }

    public void setGoodMatches(List<DMatch> goodMatches) {
        this.goodMatches = goodMatches;
    }

    public void setDetector(FastFeatureDetector detector) {
        this.detector = detector;
    }

    public void setDescriptor(ORB descriptor) {
        this.descriptor = descriptor;
    }

    public void setMatcher(DescriptorMatcher matcher) {
        this.matcher = matcher;
    }

    public void setMatchPoints1(MatOfPoint2f matchPoints1) {
        this.matchPoints1 = matchPoints1;
    }

    public void setMatchPoints2(MatOfPoint2f matchPoints2) {
        this.matchPoints2 = matchPoints2;
    }

    public MatOfPoint2f getMatchPoints1() {
        return matchPoints1;
    }

    public MatOfPoint2f getMatchPoints2() {
        return matchPoints2;
    }
}