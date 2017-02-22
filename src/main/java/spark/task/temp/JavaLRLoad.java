
package spark.task.temp;

import java.util.List;

import org.apache.hadoop.io.Text;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;

import scala.Tuple2;



/**
 * Logistic regression based classification using ML Lib.
 */
public final class JavaLRLoad {
	
	  static class FeaturesToString implements Function<Tuple2<Vector, Double>, String> {
		    public String call(Tuple2<Vector, Double> element) {
		      return element._1() + "," + element._2();
		    }
		  }

  public static void main(String[] args) {

	    SparkConf sparkConf = new SparkConf().setAppName("JavaLRLoad");
	    JavaSparkContext sc = new JavaSparkContext(sparkConf);
	    LogisticRegressionModel model=LogisticRegressionModel.load(sc.sc(), "/ntes_weblog/sparkModel/LRModel");
	    JavaPairRDD<Text,Text>  webInfoResultPercent = sc.sequenceFile("/ntes_weblog/portalUserAnalysis/statistics/userProfile/monthwebInfoResultPercent/20160406/", Text.class, Text.class);
	    JavaRDD<LabeledPoint> jpoints =  webInfoResultPercent.map(new Function<Tuple2<Text,Text>,LabeledPoint>(){

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
	      JavaRDD<LabeledPoint>[] splits =jpoints.randomSplit(new double[]{0.6,0.4},11L);
	      JavaRDD<LabeledPoint> training = splits[0].cache();
	      JavaRDD<LabeledPoint> test = splits[1];

	      JavaRDD<Vector> testv= test.map(new Function<LabeledPoint, Vector>(){
	                          
			private static final long serialVersionUID = 1L;
			public Vector call(LabeledPoint lp) { 
		            return lp.features();
		          }
		        });
	     // JavaRDD<Double>  prediction = model.predict(testv) ; 
	      final Broadcast<LogisticRegressionModel> bmodel = sc.broadcast(model);
	      JavaRDD<Integer> labels = test.map(new Function<LabeledPoint, Integer>(){
	    	  LogisticRegressionModel lmodel = bmodel.getValue();
				public Integer call(LabeledPoint lp) { 
					if (lmodel.predict(lp.features())==lp.label()){
						   return 1;
					}else {
						   return 0;
					}
			         
			          }
			        }
	      );
	      
	      
	      JavaRDD<Tuple2<Object, Object>> scoreAndlabels = test.map(new Function<LabeledPoint, Tuple2<Object, Object>>(){
	    	  LogisticRegressionModel lmodel = bmodel.getValue();
				public Tuple2<Object, Object> call(LabeledPoint lp) { 	
						   return new Tuple2(lmodel.predict(lp.features()),lp.label());

			          }
			        }
	      );
	      
	      
	      JavaRDD<Tuple2<Vector, Double>> prediction = test.map(new Function<LabeledPoint, Tuple2<Vector, Double>>(){
	    	  LogisticRegressionModel lmodel = bmodel.getValue();
				public Tuple2<Vector, Double> call(LabeledPoint lp) { 	
						   return new Tuple2(lp.features(),lmodel.predict(lp.features()));

			          }
			        }
	      );
	      
	     Integer sum= labels.reduce(new Function2<Integer,Integer,Integer>(){
	    	 
	    	  public Integer call(Integer x,Integer y) { 
	    		  return x + y;
	    	  }
	      });
	     
	     
	    BinaryClassificationMetrics metric= new BinaryClassificationMetrics(scoreAndlabels.rdd());
	    System.out.println(metric.areaUnderROC());
	    System.out.println("--------------------------");
	    double accuracy  =sum.doubleValue()/(test.count());
	    System.out.println(accuracy);
	    System.out.println("--------------------------");
	    JavaPairRDD<Tuple2<Object, Object>,Integer>   scoreAndlabelPair = scoreAndlabels.mapToPair(new PairFunction<Tuple2<Object, Object>,Tuple2<Object, Object>,Integer>(){

			public Tuple2<Tuple2<Object, Object>, Integer> call(
					Tuple2<Object, Object> t) throws Exception {
				// TODO Auto-generated method stub
				return new Tuple2<Tuple2<Object, Object>,Integer>(t,1);
			}
	    	
	    });
	    JavaPairRDD<Tuple2<Object, Object>,Integer>   scoreAndlabelCount= scoreAndlabelPair.reduceByKey(new Function2<Integer,Integer,Integer>(){

			public Integer call(Integer i1, Integer i2) throws Exception {
				// TODO Auto-generated method stub
				return i1+i2;
			}
	    	
	    });
	    List<Tuple2<Tuple2<Object, Object>, Integer>> list = scoreAndlabelCount.collect();
	     for (Tuple2<Tuple2<Object, Object>, Integer> t :list){
	    	 System.out.println(t.toString());
	     }
	   //  prediction.map(new FeaturesToString()).saveAsTextFile("/ntes_weblog/spark/productFeatures");
	    
	    sc.close();
	    sc.stop();
  }
}
