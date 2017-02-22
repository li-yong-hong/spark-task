
package spark.task.temp;

import org.apache.hadoop.io.Text;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.optimization.SquaredL2Updater;
//import org.apache.spark.mllib.optimization.SquaredL2Updater;
import org.apache.spark.mllib.regression.LabeledPoint;

import scala.Tuple2;


/**
 * Logistic regression based classification using ML Lib.
 */
public final class JavaSVMSave {

  public static void main(String[] args) {

    SparkConf sparkConf = new SparkConf().setAppName("JavaLRSave");
    JavaSparkContext sc = new JavaSparkContext(sparkConf);
    JavaPairRDD<Text,Text>  webInfoResultPercent = sc.sequenceFile("/ntes_weblog/portalUserAnalysis/statistics/userProfile/weeklyWebInfoRandomResult/20160324/", Text.class, Text.class);
    JavaRDD<LabeledPoint> points =  webInfoResultPercent.map(new Function<Tuple2<Text,Text>,LabeledPoint>(){

		public LabeledPoint call(Tuple2<Text, Text> data) throws Exception {
			String  sex = data._1().toString().split("\t")[1];
			String[] sfeature = data._2().toString().split("\t");
			double[] feature = new double[sfeature.length];
			
			int label = 0;
			if (sex.equals("male")){
				label =1;
			}
			for (int i =0;i<sfeature.length;i++){
				feature[i]=Double.parseDouble(sfeature[i]);
			}
			DenseVector  v = new DenseVector(feature);
			return new LabeledPoint(label,v);
		}
     });
    points.cache();
//    double stepSize = Double.parseDouble(args[1]);
//    int iterations = Integer.parseInt(args[2]);

    // Another way to configure LogisticRegression
    //
//     LogisticRegressionWithSGD lr = new LogisticRegressionWithSGD();
//     lr.optimizer().setUpdater(new SquaredL2Updater()).setNumIterations(10)
//                   .setStepSize(0.1)
//                   .setMiniBatchFraction(1.0);
//     lr.setIntercept(true);
//     LogisticRegressionModel model = lr.train(points.rdd());

 //    RDD<LabeledPoint> test = splits[1];
     
	//    LogisticRegressionModel model = LogisticRegressionWithSGD.train(points.rdd(),10, 0.1);
    SVMWithSGD svmAlg = new SVMWithSGD() ;
    svmAlg.optimizer().
        setRegParam(0.1).
        setUpdater(new SquaredL2Updater());
    SVMModel  model = SVMWithSGD.train(points.rdd(), 1000);
    model.save(sc.sc(), "/ntes_weblog/sparkModel/SVMModel");

    sc.stop();
  }
}
