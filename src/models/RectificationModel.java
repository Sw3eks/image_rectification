package models;

import org.opencv.core.Mat;

public class RectificationModel {
    private Mat T1;
    private Mat T2;
    private Mat Pn1;
    private Mat Pn2;

    public RectificationModel(Mat t1, Mat t2, Mat pn1, Mat pn2) {
        T1 = t1;
        T2 = t2;
        Pn1 = pn1;
        Pn2 = pn2;
    }

    public Mat getT1() {
        return T1;
    }

    public void setT1(Mat t1) {
        T1 = t1;
    }

    public Mat getT2() {
        return T2;
    }

    public void setT2(Mat t2) {
        T2 = t2;
    }

    public Mat getPn1() {
        return Pn1;
    }

    public void setPn1(Mat pn1) {
        Pn1 = pn1;
    }

    public Mat getPn2() {
        return Pn2;
    }

    public void setPn2(Mat pn2) {
        Pn2 = pn2;
    }
}
