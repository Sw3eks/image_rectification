import models.RectificationModel;
import models.RectifyModel;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.List;

import static org.opencv.imgcodecs.Imgcodecs.imwrite;

public class Rectification {

    private static final String imagePath = "./res/images/";
    private static final String calibrationPath = "./res/output/";

    public RectificationModel doRectification(Mat ppm1, Mat ppm2, Mat imagePoints1, Mat imagePoints2) {
        Mat calibration_image_1 = Imgcodecs.imread(calibrationPath + "testbilder0.jpg");
        Mat calibration_image_2 = Imgcodecs.imread(calibrationPath + "testbilder1.jpg");

        RectifyModel rectificationModel = rectify(ppm1, ppm2);

        Mat rectifiedImage1 = new Mat();
        Mat rectifiedImage2 = new Mat();
        Imgproc.warpPerspective(calibration_image_1, rectifiedImage1, rectificationModel.getT1(), calibration_image_1.size());
        Imgproc.warpPerspective(calibration_image_2, rectifiedImage2, rectificationModel.getT2(), calibration_image_2.size());
        imwrite("./res/calibration/recti1.jpg", rectifiedImage1);
        imwrite("./res/calibration/recti2.jpg", rectifiedImage2);

        MatOfPoint2f imagePointsTransformed1 = new MatOfPoint2f();
        MatOfPoint2f imagePointsTransformed2 = new MatOfPoint2f();

        Core.perspectiveTransform(imagePoints1, imagePointsTransformed1, rectificationModel.getT1());
        Core.perspectiveTransform(imagePoints2, imagePointsTransformed2, rectificationModel.getT2());

        return new RectificationModel(rectifiedImage1, rectifiedImage2, imagePointsTransformed1, imagePointsTransformed2);
    }

    public RectifyModel rectify(Mat Po1, Mat Po2) {
        Mat A1 = new Mat();
        Mat A2 = new Mat();
        Mat R1 = new Mat();
        Mat R2 = new Mat();
        Mat t1_new = new Mat();
        Mat t2_new = new Mat();
        Calib3d.decomposeProjectionMatrix(Po1, A1, R1, t1_new);
        Calib3d.decomposeProjectionMatrix(Po2, A2, R2, t2_new);

        Mat c1 = new Mat();
        Mat c2 = new Mat();
        Core.gemm(Po1.colRange(new Range(0, 3)).inv(), Po1.col(3), -1, new Mat(), 0, c1);
        Core.gemm(Po2.colRange(new Range(0, 3)).inv(), Po2.col(3), -1, new Mat(), 0, c2);

        Mat v1 = new Mat();
        Core.subtract(c1, c2, v1);

        Mat R1_row2_transposed = new Mat();
        Core.transpose(R1.row(2), R1_row2_transposed);
        Mat v2 = R1_row2_transposed.cross(v1);
        Mat v3 = v1.cross(v2);

        Mat v1_transposed = new Mat();
        Mat v2_transposed = new Mat();
        Mat v3_transposed = new Mat();
        Core.transpose(v1, v1_transposed);
        Core.transpose(v2, v2_transposed);
        Core.transpose(v3, v3_transposed);
        double v1_norm = Core.norm(v1);
        double v2_norm = Core.norm(v2);
        double v3_norm = Core.norm(v3);
        Mat row1 = new Mat();
        Mat row2 = new Mat();
        Mat row3 = new Mat();
        v1_transposed.convertTo(row1, v1_transposed.type(), 1 / v1_norm);
        v2_transposed.convertTo(row2, v2_transposed.type(), 1 / v2_norm);
        v3_transposed.convertTo(row3, v3_transposed.type(), 1 / v3_norm);
        Mat R = new Mat();
        Core.vconcat(List.of(row1, row2, row3), R);

        Mat A_sum = new Mat();
        Core.add(A1, A2, A_sum);
        Mat A = new Mat();
        A_sum.convertTo(A, A_sum.type(), 0.5);

        Mat R_times_c1_neg = new Mat();
        Mat R_times_c2_neg = new Mat();
        Core.gemm(R, c1, -1, new Mat(), 0, R_times_c1_neg, 0);
        Core.gemm(R, c2, -1, new Mat(), 0, R_times_c2_neg, 0);
        Mat bracket1 = new Mat();
        Mat bracket2 = new Mat();
        Core.hconcat(List.of(R, R_times_c1_neg), bracket1);
        Core.hconcat(List.of(R, R_times_c2_neg), bracket2);

        Mat Pn1 = new Mat();
        Mat Pn2 = new Mat();
        Core.gemm(A, bracket1, 1, new Mat(), 0, Pn1, 0);
        Core.gemm(A, bracket2, 1, new Mat(), 0, Pn2, 0);

        Mat PPM1_sub_col = Po1.colRange(new Range(0, 3));
        Mat PPM2_sub_col = Po2.colRange(new Range(0, 3));

        Mat PPM1_sub = PPM1_sub_col.rowRange(new Range(0, 3));
        Mat PPM2_sub = PPM2_sub_col.rowRange(new Range(0, 3));

        Mat Pn1_sub_col = Pn1.colRange(new Range(0, 3));
        Mat Pn2_sub_col = Pn2.colRange(new Range(0, 3));
        Mat Pn1_sub = Pn1_sub_col.rowRange(new Range(0, 3));
        Mat Pn2_sub = Pn2_sub_col.rowRange(new Range(0, 3));
        Mat T1 = new Mat();
        Mat T2 = new Mat();
        Core.gemm(Pn1_sub, PPM1_sub.inv(), 1, new Mat(), 0, T1, 0);
        Core.gemm(Pn2_sub, PPM2_sub.inv(), 1, new Mat(), 0, T2, 0);

        return new RectifyModel(T1, T2, Pn1, Pn2);
    }

    public void drawEpipolarLines(Mat image1, Mat image2, MatOfPoint2f imagePoints1, MatOfPoint2f imagePoints2) {
        Mat F = Calib3d.findFundamentalMat(
                new MatOfPoint2f(imagePoints1.submat(0, 8, 0, 1)),
                new MatOfPoint2f(imagePoints2.submat(0, 8, 0, 1)));
        System.out.println("Fundamental Matrix F: \n" + F.dump());

        // Find epipolar lines
        Mat epipolarLines1 = new Mat();
        Mat epipolarLines2 = new Mat();
        Calib3d.computeCorrespondEpilines(imagePoints2.submat(0, 8, 0, 1), 2, F, epipolarLines1);
        Calib3d.computeCorrespondEpilines(imagePoints1.submat(0, 8, 0, 1), 1, F, epipolarLines2);

        Mat img1WithLines = drawLines(image1, epipolarLines1);
        Mat img2WithLines = drawLines(image2, epipolarLines2);

        imwrite("./res/calibration/epipol1.jpg", img1WithLines);
        imwrite("./res/calibration/epipol2.jpg", img2WithLines);
    }

    private Mat drawLines(Mat image1, Mat epipolarLines1) {
        Mat resultImg = new Mat();
        image1.copyTo(resultImg);
        int epiLinesCount = epipolarLines1.rows();

        double a, b, c;

        for (int line = 0; line < epiLinesCount; line++) {
            a = epipolarLines1.get(line, 0)[0];
            b = epipolarLines1.get(line, 0)[1];
            c = epipolarLines1.get(line, 0)[2];

            int x0 = 0;
            int y0 = (int) (-(c + a * x0) / b);
            int x1 = resultImg.cols() / 2;
            int y1 = (int) (-(c + a * x1) / b);

            Point p1 = new Point(x0, y0);
            Point p2 = new Point(x1, y1);
            Scalar color = new Scalar(255, 0, 0);
            Imgproc.line(resultImg, p1, p2, color);

        }
        for (int line = 0; line < epiLinesCount; line++) {
            a = epipolarLines1.get(line, 0)[0];
            b = epipolarLines1.get(line, 0)[1];
            c = epipolarLines1.get(line, 0)[2];

            int x0 = resultImg.cols() / 2;
            int y0 = (int) (-(c + a * x0) / b);
            int x1 = resultImg.cols();
            int y1 = (int) (-(c + a * x1) / b);

            Point p1 = new Point(x0, y0);
            Point p2 = new Point(x1, y1);
            Scalar color = new Scalar(255, 0, 0);
            Imgproc.line(resultImg, p1, p2, color);

        }
        return resultImg;
    }
}
