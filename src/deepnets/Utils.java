package deepnets;

import java.lang.reflect.Array;


public class Utils {

	public static double oneOf(double[] ds) {
		return ds[(int) (Math.random() * ds.length)];
	}
	
	public static int round(double d) {
		return (int) Math.round(d);
	}
	public static double round(double d, int e) {
		final double t = (int) Math.pow(10, e);
		return Math.round(d * t) / t;
	}
	
	public static double between(double num, double denom, double lo, double hi) {
		return (num / denom) * (hi - lo) + lo;
	}
	
	public static double gaussianProbLE(double x, double variance) {
		return 0.5 * (1 + erf(x / Math.sqrt(2 * variance)));
	}
	
	public static double randomGaussianExpRate(double rate) {
		return Math.exp(rate * erf(Math.random()));
	}
	
	public static double erf(double z) {
		double t = 1 / (1 + 0.5 * Math.abs(z));
		double e = -z * z - 1.26551223 +
				t * (1.00002368 + 
				t * (0.37409196 +
				t * (0.09678418 +
				t * (-.18628806 +
				t * (0.2788607 +
				t * (-1.13520398 +
				t * (1.48851587 +
				t * (-.82215223 +
				t * (0.17087277)))))))));
		double ans = 1 - t * Math.exp(e);
		return z >= 0 ? ans : -ans;
	}
	
	@SuppressWarnings("unchecked")
	public static final <T> T[] append(T[] array, T item) {
		return append((Class<T>)array.getClass().getComponentType(), array, item);
	}
	@SuppressWarnings("unchecked")
	public static final <T> T[] append(T[] array, T[] array2) {
		return append((Class<T>)array.getClass().getComponentType(), array, array2);
	}

	public static final <T> T[] appendItem(Class<T> clasz, T[] array, T item) {
		if (array == null || array.length == 0) {
			final T[] result = createArray(clasz, 1);
			result[0] = item;
			return result;
		} 
		
		final T[] result = createArray(clasz, array.length + 1);
		System.arraycopy(array, 0, result, 0, array.length);
		result[array.length] = item;
		return result;
	}
	@SuppressWarnings("unchecked")
	public static final <T> T[] append(Class<T> clasz, T[] array, T... items) {
		if(items != null && items.length == 1){
			return appendItem(clasz, array, items[0]);
		}
		
		if (array == null || array.length == 0) {
			return items;
		} 
		
		final T[] result = createArray(clasz, array.length + items.length);
		System.arraycopy(array, 0, result, 0, array.length);
		System.arraycopy(items, 0, result, array.length, items.length);
		return result;
	}
	@SuppressWarnings("unchecked")
	public static final <T> T[] createArray(Class<T> clasz, int len) {
		return (T[])Array.newInstance(clasz, len);
	}
}
