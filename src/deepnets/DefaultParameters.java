package deepnets;

public class DefaultParameters {
	public static final ActivationFunction DEFAULT_ACT_FN = ActivationFunction.SIGMOID0p5;
	private static final double[] INIT_WGTS = {-1, -.5, -.2, 0, .2, .5, 1};

	public static double RANDOM_WEIGHT() {
		return Math.random() * 2 - 1; //Utils.oneOf(INIT_WGTS);
	}
}
