package org.apache.hadoop.examples;

import java.io.IOException;
import java.util.StringTokenizer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class KMeans {

  public static int MAX_ITER = 1;  // Maximum number of iterations
  public static int POINT_COUNT = 4601;  // Number of data points
  public static int POINT_DIM = 58;  // Number of dimension for each data point
  public static int K_CLUSTERS = 10;  // Number of clusters (k)

  /* Mapper for initializing the centroids */
  /* Input: <list of initial centroids> */
  /* Output: <c> <centroid> */
  public static class InitCentroidsMapper extends Mapper<Object, Text, Text, Text> {
    private Text keyText = new Text();
    private Text valueText = new Text();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
      boolean first = true;
      String centroidStr = new String("");

      // Read the input file, and get the initial centroids
      StringTokenizer itr = new StringTokenizer(value.toString());
      while (itr.hasMoreTokens()) {
        for (int i = 0; i < POINT_DIM; i++) {
          String num = itr.nextToken();
          if (!first) {
            centroidStr += ",";
          }
          centroidStr += num;
          first = false;
        }

        // Set the key-value pair for the cluster centroid
        keyText.set("c");
        valueText.set(centroidStr);
        context.write(keyText, valueText);
      }

      // Set the key-value pair for the costs
      keyText.set("total cost");
      valueText.set("");
      context.write(keyText, valueText);
    }
  }

  /* Mapper for initializing the data points */
  /* Input: <list of data points> */
  /* Output: <p> <data point> */
  public static class InitPointsMapper extends Mapper<Object, Text, Text, Text> {
    private Text keyText = new Text();
    private Text valueText = new Text();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
      boolean first = true;
      String pointStr = new String("");

      // Read the input file, and get the data points
      StringTokenizer itr = new StringTokenizer(value.toString());
      while (itr.hasMoreTokens()) {
        for (int i = 0; i < POINT_DIM; i++) {
          String num = itr.nextToken();
          if (!first) {
            pointStr += ",";
          }
          pointStr += num;
          first = false;
        }

        // Set the key-value pair for the data point
        keyText.set("p");
        valueText.set(pointStr);
        context.write(keyText, valueText);
      }
    }
  }

  /* Reducer for initializing the centroids and data points */
  /* Input: <c> <centroid> */
  /*        <p> <data point> */
  /* Output: <cnum,e/m> <centroid> */
  /*         <p> <list of data points> */
  /*         <total cost> <list of costs> */
  public static class InitReducer extends Reducer<Text, Text, Text, Text> {
    private Text keyText = new Text();
    private Text valueText = new Text();

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
      int clusterIdx = 1;
      String keyStr = key.toString();

      // Make 2 kinds of clusters, for Euclidean and Manhattan distance
      if (keyStr.indexOf('c') != -1 && keyStr.indexOf('o') == -1) {
        for (Text val : values) {
          keyText.set(String.valueOf(clusterIdx) + ",e");
          context.write(keyText, val);
          keyText.set(String.valueOf(clusterIdx) + ",m");
          context.write(keyText, val);
          clusterIdx++;
        }
      }

      // Set the key-value pair for data points
      else if (keyStr.indexOf('p') != -1) {
        boolean first = true;
        String valueStr = new String("");

        for (Text val : values) {
          if (!first) {
            valueStr += "|";
          }
          valueStr += val.toString();
          first = false;
        }
        valueText.set(valueStr);
        context.write(key, valueText);
      }

      // Set the key-value pair for the costs
      else {
        valueText.set("");
        context.write(key, valueText);
      }
    }
  }

  /* Mapper for applying K-Means algorithm. Assign the points to clusters */
  /* Input: <c> <centroid> */
  /*        <p> <data point> */
  /*        <cost> <list of costs> */
  /* Output: <cnum> <data point> */
  /*         <cost> <list of costs> */
  /*
  public static class AssignMapper extends Mapper<Object, Text, Text, Text> {
    private Text keyText = new Text();
    private Text valueText = new Text();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
      boolean first = true;
      String pointStr = new String("");

      // Read the input file, and get the data points
      StringTokenizer itr = new StringTokenizer(value.toString());
      while (itr.hasMoreTokens()) {
        for (int i = 0; i < POINT_DIM; i++) {
          String num = itr.nextToken();
          if (!first) {
            pointStr += ",";
          }
          pointStr += num;
          first = false;
        }

        // Set the key-value pair for the data point
        keyText.set("p");
        valueText.set(pointStr);
        context.write(keyText, valueText);
      }
    }
  }
  */

  /* Reducer for applying K-Means algorithm. CAssign the points to clusters */
  /* Input: <cluster num> <centroid> */
  /*        <p> <data point> */
  /* Output: <cluster num> <centroid> */
  /*         <p> <data point> */
  /*         <cost> <list of costs> */
  /*
  public static class AssignReducer extends Reducer<Text, Text, Text, Text> {
    private Text keyText = new Text();
    private Text valueText = new Text();

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
      for (Text val : values) {
        // Set the key-value pair for clusters and data points
        context.write(key, val);
      }

      // Set the key-value pair for the cost
      keyText.set("cost");
      valueText.set("");
      context.write(keyText, valueText);
    }
  }
  */

  /* Mapper for applying K-Means algorithm. Assign the points to clusters */
  /* Input: <c> <centroid> */
  /*        <p> <data point> */
  /*        <cost> <list of costs> */
  /* Output: <cnum> <data point> */
  /*         <cost> <list of costs> */
  /*
  public static class ComputeMapper extends Mapper<Object, Text, Text, Text> {
    private Text keyText = new Text();
    private Text valueText = new Text();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
      boolean first = true;
      String pointStr = new String("");

      // Read the input file, and get the data points
      StringTokenizer itr = new StringTokenizer(value.toString());
      while (itr.hasMoreTokens()) {
        for (int i = 0; i < POINT_DIM; i++) {
          String num = itr.nextToken();
          if (!first) {
            pointStr += ",";
          }
          pointStr += num;
          first = false;
        }

        // Set the key-value pair for the data point
        keyText.set("p");
        valueText.set(pointStr);
        context.write(keyText, valueText);
      }
    }
  }
  */

  /* Reducer for applying K-Means algorithm. CAssign the points to clusters */
  /* Input: <cluster num> <centroid> */
  /*        <p> <data point> */
  /* Output: <cluster num> <centroid> */
  /*         <p> <data point> */
  /*         <cost> <list of costs> */
  /*
  public static class ComputeReducer extends Reducer<Text, Text, Text, Text> {
    private Text keyText = new Text();
    private Text valueText = new Text();

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
      for (Text val : values) {
        // Set the key-value pair for clusters and data points
        context.write(key, val);
      }

      // Set the key-value pair for the cost
      keyText.set("cost");
      valueText.set("");
      context.write(keyText, valueText);
    }
  }
  */

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
    if (otherArgs.length != 3) {
      System.err.println("Usage: kmeans <input-initial-centroids> <input-data-points> <output-file>");
      System.exit(2);
    }

    /* Initialize the cluster centroids and data points */
    Job job1 = new Job(conf, "Initialize");
    job1.setJarByClass(KMeans.class);
    job1.setMapperClass(InitCentroidsMapper.class);
    job1.setMapperClass(InitPointsMapper.class);
    job1.setReducerClass(InitReducer.class);
    job1.setOutputKeyClass(Text.class);
    job1.setOutputValueClass(Text.class);
    MultipleInputs.addInputPath(job1, new Path(otherArgs[0]), TextInputFormat.class, InitCentroidsMapper.class);
    MultipleInputs.addInputPath(job1, new Path(otherArgs[1]), TextInputFormat.class, InitPointsMapper.class);
    FileOutputFormat.setOutputPath(job1, new Path(otherArgs[2] + "_0"));
    job1.waitForCompletion(true);

    // Apply K-Means algorithm for MAX_ITER iterations
    /*
    for (int iter = 1; iter <= MAX_ITER; i++) {

      // Assign each data points to one of the cluster
      Job job2 = new Job(conf, "K-Means Assign");
      job2.setJarByClass(KMeans.class);
      job2.setMapperClass(AssignMapper.class);
      job2.setReducerClass(AssignReducer.class);
      job2.setOutputKeyClass(Text.class);
      job2.setOutputValueClass(Text.class);
      FileInputFormat.setInputPaths(job2, new Path(otherArgs[2] + "_" + String.valueOf(iter - 1)));
      FileOutputFormat.setOutputPath(job2, new Path(otherArgs[2] + "_" + String.valueOf(iter) + "a"));
      job2.waitForCompletion(true);

      // Compute the new centroids
      Job job3 = new Job(conf, "K-Means Recompute");
      job3.setJarByClass(KMeans.class);
      job3.setMapperClass(ComputeMapper.class);
      job3.setReducerClass(ComputeReducer.class);
      job3.setOutputKeyClass(Text.class);
      job3.setOutputValueClass(Text.class);
      FileInputFormat.setInputPaths(job3, new Path(otherArgs[2] + "_" + String.valueOf(iter) + "a"));
      if (iter == MAX_ITER) {
        FileOutputFormat.setOutputPath(job3, new Path(otherArgs[2]));
      }
      else {
        FileOutputFormat.setOutputPath(job3, new Path(otherArgs[2] + "_" + String.valueOf(iter)));
      }
      job3.waitForCompletion(true);
    }
    */

    System.exit(0);
  }
}
