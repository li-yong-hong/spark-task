package spark.task.temp2

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithSGD}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.util.MLUtils

object LRSave {
  def main(args: Array[String]): Unit = {
        val conf = new SparkConf().setAppName("LRSave")
        val sc = new SparkContext(conf)
    
    
    //1 读取样本数据
  val data_path = "/ntes_weblog/sparkData/mllib/sample_libsvm_data.txt"

  val examples = MLUtils.loadLibSVMFile(sc, data_path).cache()   
   

  //2 样本数据划分训练样本与测试样本

  val splits = examples.randomSplit(Array(0.6, 0.4), seed = 11L)

  val training = splits(0).cache()

  val test = splits(1)

  val numTraining = training.count()

  val numTest = test.count()

  println(s"Training: $numTraining, test: $numTest.")

 

  //3 新建逻辑回归模型，并设置训练参数

  val numIterations = 1000

  val stepSize = 1

  val miniBatchFraction = 1.0

  val model = LogisticRegressionWithSGD.train(training, numIterations, stepSize, miniBatchFraction)
 
  
  //4 Save     
   model.save(sc, "/ntes_weblog/sparkModel/LRModel")
   val  model2=LogisticRegressionModel.load(sc, "/ntes_weblog/sparkModel/LRModel") 
     //4 对测试样本进行测试
  val prediction = model2.predict(test.map(_.features))

  val predictionAndLabel = prediction.zip(test.map(_.label))

 

  //5 计算测试误差

  val metrics = new MulticlassMetrics(predictionAndLabel)

  val precision = metrics.precision
  
  println("Precision = " + precision)
   sc.stop()
  }
}