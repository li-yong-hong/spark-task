#! /bin/sh

class=$@
SPARK_HOME=/opt/spark/spark
JAR=/home/mrd/liyonghong/spark-task-1.0-SNAPSHOT.jar
echo ${SPARK_HOME}/bin/spark-submit --class $class --master yarn-client ${JAR}
${SPARK_HOME}/bin/spark-submit --class $class --master yarn-client ${JAR}