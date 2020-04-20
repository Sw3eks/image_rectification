package models;

import org.opencv.core.Mat;

public class CalibrationModel {
    private Mat calibrationImagePoints1;
    private Mat calibrationImagePoints2;

    public CalibrationModel(Mat calibrationImagePoints1, Mat calibrationImagePoints2) {
        this.calibrationImagePoints1 = calibrationImagePoints1;
        this.calibrationImagePoints2 = calibrationImagePoints2;
    }

    public Mat getCalibrationImagePoints1() {
        return calibrationImagePoints1;
    }

    public void setCalibrationImagePoints1(Mat calibrationImagePoints1) {
        this.calibrationImagePoints1 = calibrationImagePoints1;
    }

    public Mat getCalibrationImagePoints2() {
        return calibrationImagePoints2;
    }

    public void setCalibrationImagePoints2(Mat calibrationImagePoints2) {
        this.calibrationImagePoints2 = calibrationImagePoints2;
    }
}
