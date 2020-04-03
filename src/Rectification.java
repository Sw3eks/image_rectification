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

    public void doRectification() {

    }

}
