package utils;

import java.util.*;

import deepnets.*;

public class RandomUtils {

	public static double[] randomBits(int len) {
		double[] result = new double[len];
		for (int i = 0; i < len; i++) result[i] = Math.random() < 0.5 ? 0 : 1;
		return result;
	}
	public static double randomOf(double[] d) {
		return d[(int) (Math.random() * d.length)];
	}
	public static double[] randomOf(List<double[]> list) {
		return list.get((int) (Math.random() * list.size()));
	}
	public static double randBetween(double x1, double x2) {
		return Math.random() * (x2 - x1) + x1;
	}
	
	public static double[] concat(double[] d1, double[] d2) {
		double[] result = new double[d1.length + d2.length];
		System.arraycopy(d1, 0, result, 0, d1.length);
		System.arraycopy(d2, 0, result, d1.length, d2.length);
		return result;
	}
	
	public static void randomizeWeights(FFNeuralNetwork ann) {
		Collection<Connection> conns = Connection.getAllConnections(ann);
		for (Connection conn : conns) conn.getWeight().setWeight(DefaultParameters.RANDOM_WEIGHT());
	}
	
}
