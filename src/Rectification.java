import models.RectificationModel;
import models.RectifyModel;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Range;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.List;

import static org.opencv.calib3d.Calib3d.decomposeProjectionMatrix;

public class Rectification {

    /**
     * Uses the implemented algorithm to rectify 2 given images
     *
     * @param ppm1         projection matrix 1
     * @param ppm2         projection matrix 2
     * @param image1       image to be rectified 1
     * @param image2       image to be rectified 2
     * @param imagePoints1 feature Points for image 1
     * @param imagePoints2 feature Points for image 2
     * @return RectificationModel with the rectified images and rectified image points
     */
    public RectificationModel doRectification(Mat ppm1, Mat ppm2, Mat image1, Mat image2, Mat imagePoints1, Mat imagePoints2) {

        RectifyModel rectificationModel = rectify(ppm1, ppm2);

        // applies the transformation matrices calculated by 'rectify' to the given images
        Mat rectifiedImage1 = new Mat();
        Mat rectifiedImage2 = new Mat();
        Imgproc.warpPerspective(image1, rectifiedImage1, rectificationModel.getT1(), new Size(1000, 500));
        Imgproc.warpPerspective(image2, rectifiedImage2, rectificationModel.getT2(), new Size(1000, 500));

        // Transform the image points:
        Mat rectifiedImagePoints1 = new Mat();
        Mat rectifiedImagePoints2 = new Mat();

        Core.perspectiveTransform(imagePoints1, rectifiedImagePoints1, rectificationModel.getT1());
        Core.perspectiveTransform(imagePoints2, rectifiedImagePoints2, rectificationModel.getT2());

        return new RectificationModel(
                rectifiedImage1, rectifiedImage2,
                rectifiedImagePoints1, rectifiedImagePoints2);
    }

    /**
     * Rectifies a stereo pair with known camera calibration using a simple algorithm described in
     * A. Fusiello, E. Trucco, and A. Verri, "A Compact Algorithm for Rectification of Stereo Pairs"
     * Machine Vision and Applications, 2000
     *
     * @param Po1 projection matrix of image/camera 1
     * @param Po2 projection matrix of image/camera 2
     * @return resulting projection & transformation matrices
     */
    public RectifyModel rectify(Mat Po1, Mat Po2) {
        Mat A1 = new Mat();
        Mat A2 = new Mat();
        Mat R1 = new Mat();
        Mat R2 = new Mat();
        Mat t1 = new Mat();
        Mat t2 = new Mat();
        decomposeProjectionMatrix(Po1, A1, R1, t1);
        decomposeProjectionMatrix(Po2, A2, R2, t2);

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
        Mat row1 = new Mat();
        Mat row2 = new Mat();
        Mat row3 = new Mat();
        v1_transposed.convertTo(row1, v1_transposed.type(), 1 / Core.norm(v1));
        v2_transposed.convertTo(row2, v2_transposed.type(), 1 / Core.norm(v2));
        v3_transposed.convertTo(row3, v3_transposed.type(), 1 / Core.norm(v3));
        Mat R = new Mat();
        Core.vconcat(List.of(row1, row2, row3), R);

        Mat A_sum = new Mat();
        Core.add(A1, A2, A_sum);
        Mat A = new Mat();
        A_sum.convertTo(A, A_sum.type(), 0.5);
        A.put(0, 1, 0); // set skew to zero
//        A.put(0, 2, A.get(0, 2)[0] - 500); // to recenter, not always needed

        Mat R_times_c1_neg = new Mat();
        Mat R_times_c2_neg = new Mat();
        Core.gemm(R, c1, -1, new Mat(), 0, R_times_c1_neg, 0);
        Core.gemm(R, c2, -1, new Mat(), 0, R_times_c2_neg, 0);
        Mat R_times_R_times_c1_neg = new Mat();
        Mat R_times_R_times_c2_neg = new Mat();
        Core.hconcat(List.of(R, R_times_c1_neg), R_times_R_times_c1_neg);
        Core.hconcat(List.of(R, R_times_c2_neg), R_times_R_times_c2_neg);

        Mat Pn1 = new Mat();
        Mat Pn2 = new Mat();
        Core.gemm(A, R_times_R_times_c1_neg, 1, new Mat(), 0, Pn1, 0);
        Core.gemm(A, R_times_R_times_c2_neg, 1, new Mat(), 0, Pn2, 0);

        Mat Po1_sub_col = Po1.colRange(new Range(0, 3));
        Mat Po2_sub_col = Po2.colRange(new Range(0, 3));

        Mat Po1_sub = Po1_sub_col.rowRange(new Range(0, 3));
        Mat Po2_sub = Po2_sub_col.rowRange(new Range(0, 3));

        Mat Pn1_sub_col = Pn1.colRange(new Range(0, 3));
        Mat Pn2_sub_col = Pn2.colRange(new Range(0, 3));
        Mat Pn1_sub = Pn1_sub_col.rowRange(new Range(0, 3));
        Mat Pn2_sub = Pn2_sub_col.rowRange(new Range(0, 3));
        Mat T1 = new Mat();
        Mat T2 = new Mat();
        Core.gemm(Pn1_sub, Po1_sub.inv(), 1, new Mat(), 0, T1, 0);
        Core.gemm(Pn2_sub, Po2_sub.inv(), 1, new Mat(), 0, T2, 0);
        return new RectifyModel(T1, T2, Pn1, Pn2);
    }

}
