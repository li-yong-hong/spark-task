#! /bin/sh

baseDir=$(cd "$(dirname "$0")"; pwd)

sh $baseDir/runSparkJob.sh spark.task.temp2.SparkPi