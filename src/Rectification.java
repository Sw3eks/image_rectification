import models.RectificationModel;
import models.RectifyModel;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Range;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.List;

import static org.opencv.calib3d.Calib3d.decomposeProjectionMatrix;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;

public class Rectification {

    private static final String imagePath = "./res/images/";
    private static final String calibrationPath = "./res/output/";

    public RectificationModel doRectification(Mat ppm1, Mat ppm2, Mat imagePoints1, Mat imagePoints2) {
        Mat calibration_image_1 = Imgcodecs.imread(calibrationPath + "testbilder0.jpg");
        Mat calibration_image_2 = Imgcodecs.imread(calibrationPath + "testbilder1.jpg");

        RectifyModel rectificationModel = rectify(ppm1, ppm2);
//        Mat cameraMatrix = new Mat();
//        Mat rVec = new Mat();
//        Mat tVec = new Mat();
//        decomposeProjectionMatrix(rectificationModel.getPn1(), cameraMatrix, rVec, tVec);
//        System.out.println("Matrix: " + cameraMatrix.dump());
//        System.out.println("R: " + rVec.dump());
//        System.out.println("T: " + tVec.dump());

        Mat rectifiedImage1 = new Mat();
        Mat rectifiedImage2 = new Mat();
        Imgproc.warpPerspective(calibration_image_1, rectifiedImage1, rectificationModel.getT1(), calibration_image_1.size());
        Imgproc.warpPerspective(calibration_image_2, rectifiedImage2, rectificationModel.getT2(), calibration_image_2.size());
        imwrite("./res/output/recti1.jpg", rectifiedImage1);
        imwrite("./res/output/recti2.jpg", rectifiedImage2);

        MatOfPoint2f imagePointsTransformed1 = new MatOfPoint2f();
        MatOfPoint2f imagePointsTransformed2 = new MatOfPoint2f();

        //Core.perspectiveTransform(imagePoints1, imagePointsTransformed1, rectificationModel.getT1());
        //Core.perspectiveTransform(imagePoints2, imagePointsTransformed2, rectificationModel.getT2());

        return new RectificationModel(
                rectifiedImage1,
                rectifiedImage2,
                imagePointsTransformed1,
                imagePointsTransformed2);
    }

    public RectifyModel rectify(Mat Po1, Mat Po2) {
        Mat A1 = new Mat();
        Mat A2 = new Mat();
        Mat R1 = new Mat();
        Mat R2 = new Mat();
        Mat t1_new = new Mat();
        Mat t2_new = new Mat();
        decomposeProjectionMatrix(Po1, A1, R1, t1_new);
        decomposeProjectionMatrix(Po2, A2, R2, t2_new);

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
        A.put(0, 1, 0); // set skew to zero
        //A.put(0, 2, A.get(0, 2)[0] - 160);

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

        System.out.println(Pn1.dump());
        System.out.println(Pn2.dump());
        System.out.println(T1.dump());
        System.out.println(T2.dump());
        return new RectifyModel(T1, T2, Pn1, Pn2);
    }

}
