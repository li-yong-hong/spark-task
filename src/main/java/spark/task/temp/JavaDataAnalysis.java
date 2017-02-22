
package spark.task.temp;

import org.apache.hadoop.io.Text;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
//import org.apache.spark.mllib.optimization.SquaredL2Updater;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;

import scala.Tuple2;


/**
 * Logistic regression based classification using ML Lib.
 */
public final class JavaDataAnalysis {

  public static void main(String[] args) {

    SparkConf sparkConf = new SparkConf().setAppName("JavaDataAnalysis");
    JavaSparkContext sc = new JavaSparkContext(sparkConf);
    JavaPairRDD<Text,Text>  webInfoResultPercent = sc.sequenceFile("/ntes_weblog/portalUserAnalysis/statistics/userProfile/monthwebInfoResultPercent/20160406/", Text.class, Text.class);
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
    JavaRDD<LabeledPoint>  malePoint = points.filter(new Function<LabeledPoint,Boolean>(){
		public Boolean call(LabeledPoint lp) throws Exception {
			// TODO Auto-generated method stub
			if(lp.label()==1){
				return true;
			}else {
				return false;
			}
			
		}
    });
    JavaRDD<LabeledPoint>  femalePoint = points.filter(new Function<LabeledPoint,Boolean>(){
		public Boolean call(LabeledPoint lp) throws Exception {
			// TODO Auto-generated method stub
			if(lp.label()==0){
				return true;
			}else {
				return false;
			}
			
		}
    });
    
    JavaRDD<Vector> maleVectors = malePoint.map(new Function<LabeledPoint,Vector>(){

		public Vector call(LabeledPoint lp) throws Exception {
			// TODO Auto-generated method stub
			return lp.features();
		}
    	
    });
    
    JavaRDD<Vector> femaleVectors = femalePoint.map(new Function<LabeledPoint,Vector>(){

		public Vector call(LabeledPoint lp) throws Exception {
			// TODO Auto-generated method stub
			return lp.features();
		}
    	
    });
    
    RowMatrix  maleMatrix = new RowMatrix(maleVectors.rdd());
    RowMatrix  femaleMatrix = new RowMatrix(femaleVectors.rdd());
    
    MultivariateStatisticalSummary maleColumnSummary = maleMatrix.computeColumnSummaryStatistics();
    MultivariateStatisticalSummary femaleColumnSummary = femaleMatrix.computeColumnSummaryStatistics();
    
    System.out.println(maleVectors.count()+":"+femaleVectors.count());
    System.out.println(maleColumnSummary.mean());
    System.out.println(femaleColumnSummary.mean());
    System.out.println(maleColumnSummary.variance());
    System.out.println(femaleColumnSummary.variance());
    System.out.println(maleColumnSummary.numNonzeros());
    System.out.println(femaleColumnSummary.numNonzeros());

    sc.stop();
  }
}
