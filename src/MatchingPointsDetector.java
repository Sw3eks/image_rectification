import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.features2d.Features2d;

import java.util.ArrayList;
import java.util.List;

import static org.opencv.imgcodecs.Imgcodecs.imwrite;

public class MatchingPointsDetector {

    public MatchingPointsDetector() {
    }

    public void matchImages(Mat img1, Mat img2) {

        List<Mat> images = new ArrayList<>();
        images.add(img1);
        images.add(img2);
        FastFeatureDetector detector = FastFeatureDetector.create();
        List<MatOfKeyPoint> keypoints = new ArrayList<>();
        detector.detect(images, keypoints);

        //-- Draw keypoints
        Features2d.drawKeypoints(images.get(0), keypoints.get(0), images.get(0));
        Features2d.drawKeypoints(images.get(1), keypoints.get(1), images.get(1));

        imwrite("./res/calibration/detect1.jpg", images.get(0));
        imwrite("./res/calibration/detect2.jpg", images.get(1));
    }
}
