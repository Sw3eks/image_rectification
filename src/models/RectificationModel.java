package models;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;

public class RectificationModel {
    private Mat rectifiedImage1;
    private Mat rectifiedImage2;

    private MatOfPoint2f rectifiedImagePoints1;
    private MatOfPoint2f rectifiedImagePoints2;

    public RectificationModel(Mat rectifiedImage1, Mat rectifiedImage2, MatOfPoint2f rectifiedImagePoints1, MatOfPoint2f rectifiedImagePoints2) {
        this.rectifiedImage1 = rectifiedImage1;
        this.rectifiedImage2 = rectifiedImage2;
        this.rectifiedImagePoints1 = rectifiedImagePoints1;
        this.rectifiedImagePoints2 = rectifiedImagePoints2;
    }

    public Mat getRectifiedImage1() {
        return rectifiedImage1;
    }

    public void setRectifiedImage1(Mat rectifiedImage1) {
        this.rectifiedImage1 = rectifiedImage1;
    }

    public Mat getRectifiedImage2() {
        return rectifiedImage2;
    }

    public void setRectifiedImage2(Mat rectifiedImage2) {
        this.rectifiedImage2 = rectifiedImage2;
    }

    public MatOfPoint2f getRectifiedImagePoints1() {
        return rectifiedImagePoints1;
    }

    public void setRectifiedImagePoints1(MatOfPoint2f rectifiedImagePoints1) {
        this.rectifiedImagePoints1 = rectifiedImagePoints1;
    }

    public MatOfPoint2f getRectifiedImagePoints2() {
        return rectifiedImagePoints2;
    }

    public void setRectifiedImagePoints2(MatOfPoint2f rectifiedImagePoints2) {
        this.rectifiedImagePoints2 = rectifiedImagePoints2;
    }
}
