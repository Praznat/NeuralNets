package deepnets;

public class DefaultParameters {
	private static final double M = 0.9;// 0.1;
	public static final ActivationFunction DEFAULT_ACT_FN = ActivationFunction.SIGMOID0p5;
	private static final double[] INIT_WGTS = {-1, -.5*M, -.2*M, 0*M, .2*M, .5*M, 1*M};

	public static double RANDOM_WEIGHT() {
		return (Math.random() * 2 - 1)*M; //Utils.oneOf(INIT_WGTS);
	}
}
