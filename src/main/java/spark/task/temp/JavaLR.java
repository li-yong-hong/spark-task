
package spark.task.temp;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;

/**
 * Logistic regression based classification using ML Lib.
 */
public final class JavaLR {

  public static void main(String[] args) {

    SparkConf sparkConf = new SparkConf().setAppName("JavaLR");
    JavaSparkContext sc = new JavaSparkContext(sparkConf);
    RDD<LabeledPoint> points =MLUtils.loadLibSVMFile(sc.sc(), "/ntes_weblog/sparkData/mllib/sample_libsvm_data.txt").cache();
//    double stepSize = Double.parseDouble(args[1]);
//    int iterations = Integer.parseInt(args[2]);

    // Another way to configure LogisticRegression
    //
    // LogisticRegressionWithSGD lr = new LogisticRegressionWithSGD();
    // lr.optimizer().setNumIterations(iterations)
    //               .setStepSize(stepSize)
    //               .setMiniBatchFraction(1.0);
    // lr.setIntercept(true);
    // LogisticRegressionModel model = lr.train(points.rdd());
     RDD<LabeledPoint>[] splits =points.randomSplit(new double[]{0.6,0.4},11L);
     RDD<LabeledPoint> training = splits[0].cache();
     RDD<LabeledPoint> test = splits[1];
     
    LogisticRegressionModel model = LogisticRegressionWithSGD.train(training,10, 0.1);
    

    System.out.print("Final w: " + model.weights());

    sc.stop();
  }
}
