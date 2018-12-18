package org.apache.hadoop.examples;

import java.io.IOException;
import java.util.StringTokenizer;
import java.lang.Math;
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

  public static int MAX_ITER = 20;  // Maximum number of iterations
  public static int POINT_DIM = 58;  // Number of dimension for each data point
  public static int K_CLUSTERS = 10;  // Number of clusters (k)

  /* Mapper for initializing the centroids */
  /* Input: <list of initial centroids> */
  /* Output: <key> <c%centroid> */
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
        keyText.set("key");
        valueText.set("c%" + centroidStr);
        context.write(keyText, valueText);
      }
    }
  }

  /* Mapper for initializing the data points */
  /* Input: <list of data points> */
  /* Output: <key> <p%data point> */
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
        keyText.set("key");
        valueText.set("p%" + pointStr);
        context.write(keyText, valueText);
      }
    }
  }

  /* Reducer for initializing the centroids and data points */
  /* Input: <key> <c%centroid> */
  /*        <key> <p%data point> */
  /* Output: <data point> <e%list of e_centroids> */
  /*         <data point> <m%list of m_centroids> */
  public static class InitReducer extends Reducer<Text, Text, Text, Text> {
    private Text keyText = new Text();
    private Text valueText = new Text();

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
      String keyStr = key.toString();
      String centroidStr = new String("");
      boolean first = true;

      for (Text val : values) {
        String valueStr = val.toString();

        // Get all the initial centroids
        if (valueStr.contains("c%")) {
          if (!first) {
            centroidStr += "/";
          }
          centroidStr += valueStr.substring(2);
          first = false;
        }

        // Get all the data points, and set the key-value pairs
        if (valueStr.contains("p%")) {
          String dataPoint = valueStr.substring(2);
          keyText.set(dataPoint);
          valueText.set("e%" + centroidStr);
          context.write(keyText, valueText);
          keyText.set(dataPoint);
          valueText.set("m%" + centroidStr);
          context.write(keyText, valueText);
        }
      }
    }
  }

  /* Mapper for applying K-Means algorithm. Assign the points to clusters */
  /* Input: <data point> <e%list of e_centroids> */
  /*        <data point> <m%list of m_centroids> */
  /* Output: <e%e_centroid> <y/n%data point> */
  /*         <m%m_centroid> <y/n%data point> */
  /*         <total cost,e> <e_cost> */
  /*         <total cost,m> <m_cost> */
  public static class AssignMapper extends Mapper<Object, Text, Text, Text> {
    private Text keyText = new Text();
    private Text valueText = new Text();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
      StringTokenizer itr = new StringTokenizer(value.toString());
      while (itr.hasMoreTokens()) {
        double[] dataPoint = new double[POINT_DIM];
        double[][] centroidList = new double[K_CLUSTERS][POINT_DIM];

        // Read the input file, and get the data point and the centroids
        String dataPointStr = itr.nextToken();
        String centroidStr = itr.nextToken();
        String typeStr = centroidStr.substring(0, 2);  // e% or m%

        // Store the data point
        String[] pointDim = dataPointStr.split(",");
        for (int d = 0; d < POINT_DIM; d++) {
          dataPoint[d] = Double.parseDouble(pointDim[d]);
        }

        // Store the centroids
        String[] centroidStrList = centroidStr.substring(2).split("/");
        for (int c = 0; c < K_CLUSTERS; c++) {
          String[] centroidDim = centroidStrList[c].split(",");
          for (int d = 0; d < POINT_DIM; d++) {
            centroidList[c][d] = Double.parseDouble(centroidDim[d]);
          }
        }

        // Assign the data point to its nearest centroid
        double minCost = Double.MAX_VALUE;
        int bestCluster = 0;
        for (int c = 0; c < K_CLUSTERS; c++) {
          double curCost = 0.0;
          for (int d = 0; d < POINT_DIM; d++) {

            // Use Euclidean distance
            if (typeStr.indexOf('e') != -1) {
              curCost += (dataPoint[d] - centroidList[c][d]) * (dataPoint[d] - centroidList[c][d]);
            }

            // Use Manhattan distance
            else {
              curCost += Math.abs(dataPoint[d] - centroidList[c][d]);
            }
          }

          if (curCost < minCost) {
            minCost = curCost;
            bestCluster = c;
          }
        }

        // Set the key-value pairs
        for (int c = 0; c < K_CLUSTERS; c++) {
          keyText.set(typeStr + centroidStrList[c]);
          if (c == bestCluster) {
            valueText.set("y%" + dataPointStr);
          }
          else {
            valueText.set("n%" + dataPointStr);
          }
          context.write(keyText, valueText);
        }
        keyText.set("total cost," + typeStr.charAt(0));
        valueText.set(String.valueOf(minCost));
        context.write(keyText, valueText);
      }
    }
  }

  /* Reducer for applying K-Means algorithm. Assign the points to clusters */
  /* Input: <e%e_centroid> <y/n%data point> */
  /*        <m%m_centroid> <y/n%data point> */
  /*        <total cost,e> <e_cost> */
  /*        <total cost,m> <m_cost> */
  /* Output: <e%e_centroid> <list of y/n%data points> */
  /*         <m%m_centroid> <list of y/n%data points> */
  /*         <total cost,e> <sum of e_costs> */
  /*         <total cost,m> <sum of m_costs> */
  public static class AssignReducer extends Reducer<Text, Text, Text, Text> {
    private Text valueText = new Text();

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
      String keyStr = key.toString();

      // Gather all the data points belong and not belong to the cluster
      if (keyStr.indexOf('%') != -1) {
        boolean first = true;
        String pointList = new String("");
        for (Text val : values) {
          if (!first) {
            pointList += "/";
          }
          pointList += val.toString();
          first = false;
        }
        valueText.set(pointList);
        context.write(key, valueText);
      }

      // Gather all the costs and sum them up
      else {
        double costSum = 0.0;
        for (Text val : values) {
          double cost = Double.parseDouble(val.toString());
          costSum += cost;
        }
        valueText.set(String.format("%.3f", costSum));
        context.write(key, valueText);
      }
    }
  }

  /* Mapper for applying K-Means algorithm. Compute the new centroids */
  /* Input: <e%e_centroid> <list of y/n%data points> */
  /*        <m%m_centroid> <list of y/n%data points> */
  /*        <total cost,e> <sum of e_costs> */
  /*        <total cost,m> <sum of m_costs> */
  /* Output: <data point> <e%e_centroid> */
  /*         <data point> <m%m_centroid> */
  public static class NewCentroidsMapper extends Mapper<Object, Text, Text, Text> {
    private Text keyText = new Text();
    private Text valueText = new Text();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
      StringTokenizer itr = new StringTokenizer(value.toString());
      while (itr.hasMoreTokens()) {

        // Read the input file, and get the centroid and the list of data points
        String centroidStr = itr.nextToken();
        String dataPointStr = new String("");
        if (itr.hasMoreTokens()) {
          dataPointStr = itr.nextToken();
        }
        String typeStr = centroidStr.substring(0, 2);  // e% or m%

        if (centroidStr.indexOf('%') != -1) {
          String[] dataPointListStr = dataPointStr.split("/");
          int numOfPoint = 0;
          double[] newCentroid = new double[POINT_DIM];
          for (int d = 0; d < POINT_DIM; d++) {
            newCentroid[d] = 0.0;
          }

          // Sum up the value of each data points
          for (int i = 0; i < dataPointListStr.length; i++) {
            if (dataPointListStr[i].indexOf('y') != -1) {
              numOfPoint++;
              String[] pointDim = dataPointListStr[i].substring(2).split(",");
              for (int d = 0; d < POINT_DIM; d++) {
                newCentroid[d] += Double.parseDouble(pointDim[d]);
              }
            }
          }

          // Get the average of the sum as the new centroid
          String newCentroidStr = new String("");
          for (int d = 0; d < POINT_DIM; d++) {
            newCentroid[d] /= (double)numOfPoint;
            if (d != 0) {
              newCentroidStr += ",";
            }
            newCentroidStr += String.format("%.3f", newCentroid[d]);
          }

          // Set the key-value pairs
          for (int i = 0; i < dataPointListStr.length; i++) {
            keyText.set(dataPointListStr[i].substring(2));
            valueText.set(typeStr + newCentroidStr);
            context.write(keyText, valueText);
          }
        }
      }
    }
  }

  /* Reducer for applying K-Means algorithm. Compute the new centroids */
  /* Input: <data point> <e%e_centroid> */
  /*        <data point> <m%m_centroid> */
  /* Output: <data point> <e%list of e_centroids> */
  /*         <data point> <m%list of m_centroids> */
  public static class NewCentroidsReducer extends Reducer<Text, Text, Text, Text> {
    private Text valueText = new Text();

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
      String typeStr = new String("");
      String e_centroidListStr = new String("e%");
      String m_centroidListStr = new String("m%");
      boolean e_first = true;
      boolean m_first = true;

      // Gather all the recomputed centroids
      for (Text val : values) {
        String centroidStr = val.toString();

        // Centroids for Euclidean distance
        if (centroidStr.indexOf('e') != -1) {
          if (!e_first) {
            e_centroidListStr += "/";
          }
          e_centroidListStr += centroidStr.substring(2);
          e_first = false;
        }

        // Centroids for Manhattan distance
        else {
          if (!m_first) {
            m_centroidListStr += "/";
          }
          m_centroidListStr += centroidStr.substring(2);
          m_first = false;
        }
      }

      // Set the key-value pairs
      valueText.set(e_centroidListStr);
      context.write(key, valueText);
      valueText.set(m_centroidListStr);
      context.write(key, valueText);
    }
  }

  /* Mapper for calculating the distance between each pair of centroids */
  /* Input: <data point> <e%list of e_centroids> */
  /*        <data point> <m%list of m_centroids> */
  /* Output: <e> <e_distance> */
  /*         <m> <m_distance> */
  public static class PairDistanceMapper extends Mapper<Object, Text, Text, Text> {
    private Text keyText = new Text();
    private Text valueText = new Text();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
      double[][] e_centroidList = new double[K_CLUSTERS][POINT_DIM];
      double[][] m_centroidList = new double[K_CLUSTERS][POINT_DIM];
      String ONE_DATA_POINT = "0,0.64,0.64,0,0.32,0,0,0,0,0,0,0.64,0,0,0,0.32,0,1.29,1.93,0,0.96,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.778,0,0,3.756,61,278,1";

      StringTokenizer itr = new StringTokenizer(value.toString());
      if (itr.hasMoreTokens()) {
        String dataPointStr = itr.nextToken();

        // Only need to read one key-value pair
        if (dataPointStr.equals(ONE_DATA_POINT)) {
          String centroidListStr = new String("");
          if (itr.hasMoreTokens()) {
            centroidListStr = itr.nextToken();
          }
          String typeStr = centroidListStr.substring(0, 1);  // e or m

          // Store the centroids for Euclidean and Manhattan distances
          String[] centroidStrList = centroidListStr.substring(2).split("/");
          for (int c = 0; c < K_CLUSTERS; c++) {
            String[] centroidDim = centroidStrList[c].split(",");
            for (int d = 0; d < POINT_DIM; d++) {
              if (typeStr.indexOf('e') != -1) {
                e_centroidList[c][d] = Double.parseDouble(centroidDim[d]);
              }
              else {
                m_centroidList[c][d] = Double.parseDouble(centroidDim[d]);
              }
            }
          }

          // Calculate the distance between each pair of clusters
          for (int c1 = 0; c1 < K_CLUSTERS; c1++) {
            for (int c2 = c1 + 1; c2 < K_CLUSTERS; c2++) {

              // Use Euclidean distance
              if (typeStr.indexOf('e') != -1) {
                double e_distance = 0.0;
                for (int d = 0; d < POINT_DIM; d++) {
                  e_distance += (e_centroidList[c1][d] - e_centroidList[c2][d]) * (e_centroidList[c1][d] - e_centroidList[c2][d]);
                }
                e_distance = Math.sqrt(e_distance);
                keyText.set("e");
                valueText.set(String.format("%.3f", e_distance));
                context.write(keyText, valueText);
              }

              // Use Manhattan distance
              else {
                double m_distance = 0.0;
                for (int d = 0; d < POINT_DIM; d++) {
                  m_distance += Math.abs(m_centroidList[c1][d] - m_centroidList[c2][d]);
                }
                keyText.set("m");
                valueText.set(String.format("%.3f", m_distance));
                context.write(keyText, valueText);
              }
            }
          }
        }
      }
    }
  }

  /* Reducer for calculating the distance between each pair of centroids */
  /* Input: <e> <e_distance> */
  /*         <m> <m_distance> */
  /* Output: <e> <list of e_distances> */
  /*         <m> <list of m_distances> */
  public static class PairDistanceReducer extends Reducer<Text, Text, Text, Text> {
    private Text valueText = new Text();

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
      String distanceStr = new String("");
      boolean first = true;

      // Gather all the distances, and make them a list
      for (Text val : values) {
        if (!first) {
          distanceStr += ", ";
        }
        distanceStr += val.toString();
        first = false;
      }

      // Set the key-value pair for the distances
      valueText.set(distanceStr);
      context.write(key, valueText);
    }
  }

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
    for (int iter = 1; iter <= MAX_ITER; iter++) {

      // Assign each data points to one of the cluster
      Job job2 = new Job(conf, "K-Means Assign");
      job2.setJarByClass(KMeans.class);
      job2.setMapperClass(AssignMapper.class);
      job2.setReducerClass(AssignReducer.class);
      job2.setOutputKeyClass(Text.class);
      job2.setOutputValueClass(Text.class);
      FileInputFormat.setInputPaths(job2, new Path(otherArgs[2] + "_" + String.valueOf(iter - 1)));
      FileOutputFormat.setOutputPath(job2, new Path(otherArgs[2] + "_" + String.valueOf(iter) + "_c"));
      job2.waitForCompletion(true);

      // Compute the new centroids
      Job job3 = new Job(conf, "K-Means Recompute");
      job3.setJarByClass(KMeans.class);
      job3.setMapperClass(NewCentroidsMapper.class);
      job3.setReducerClass(NewCentroidsReducer.class);
      job3.setOutputKeyClass(Text.class);
      job3.setOutputValueClass(Text.class);
      FileInputFormat.setInputPaths(job3, new Path(otherArgs[2] + "_" + String.valueOf(iter) + "_c"));
      FileOutputFormat.setOutputPath(job3, new Path(otherArgs[2] + "_" + String.valueOf(iter)));
      job3.waitForCompletion(true);
    }

    // Calculate the distance between each pair of centroids
    Job job4 = new Job(conf, "Pair Distance");
    job4.setJarByClass(KMeans.class);
    job4.setMapperClass(PairDistanceMapper.class);
    job4.setReducerClass(PairDistanceReducer.class);
    job4.setOutputKeyClass(Text.class);
    job4.setOutputValueClass(Text.class);
    FileInputFormat.setInputPaths(job4, new Path(otherArgs[2] + "_" + String.valueOf(MAX_ITER)));
    FileOutputFormat.setOutputPath(job4, new Path(otherArgs[2]));
    job4.waitForCompletion(true);

    System.exit(0);
  }
}
