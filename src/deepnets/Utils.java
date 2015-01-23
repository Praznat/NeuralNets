package deepnets;


public class Utils {

	public static double oneOf(double[] ds) {
		return ds[(int) (Math.random() * ds.length)];
	}
	
	public static int round(double d) {
		return (int) Math.round(d);
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
}
