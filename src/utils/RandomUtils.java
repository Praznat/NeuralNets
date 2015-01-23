package utils;

import java.util.List;

public class RandomUtils {

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
	
}
