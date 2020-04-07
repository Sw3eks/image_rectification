import org.opencv.calib3d.Calib3d;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.ArrayList;
import java.util.List;

public class Rectification {

    private String imagePath = "/res/images/";

    public void setUp() {
        List<Mat> images = new ArrayList<>();

        Mat image_1 = Imgcodecs.imread(imagePath + "rect_image_1.jpg");
        Mat image_2 = Imgcodecs.imread(imagePath + "rect_image_2.jpg");
        images.add(image_1);
        images.add(image_2);
    }

    public void doRectification(Mat Po1, Mat Po2) {
        Mat A1 = new Mat();
        Mat A2 = new Mat();
        Mat R1 = new Mat();
        Mat R2 = new Mat();
        Mat t1_new = new Mat();
        Mat t2_new = new Mat();
        Calib3d.decomposeProjectionMatrix(Po1, A1, R1, t1_new);
        Calib3d.decomposeProjectionMatrix(Po2, A2, R2, t2_new);

        System.out.println("A1: " + A1.dump());
        System.out.println("A2: " + A2.dump());
    }

}
