package spark.task.temp2

import org.apache.spark.SparkContext

object CountTest {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext();
    
    var lrRDD = sc.textFile("/ntes_weblog/channelAnalysis/midLayer/app/userDocids/20160302/", 8);
    
    println("total records: " + lrRDD.count());
  }
}