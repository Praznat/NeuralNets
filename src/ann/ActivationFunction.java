package ann;

import java.io.Serializable;

@SuppressWarnings("serial")
public interface ActivationFunction extends Serializable {
	
	public double feed(double input);
	public double backfeed(double input);
	public double derivative(double activation);
	
	public static class Sigmoidal implements ActivationFunction {
		private final double threshold;
		public Sigmoidal(double threshold) {
			this.threshold = threshold;
		}
		@Override
		public double feed(double input) {
			return 1 / (1 + Math.exp(threshold - input));
		}
		@Override
		public double derivative(double activation) {
			return activation * (1 - activation);
		}
		@Override
		public double backfeed(double output) {
			// TODO double-check?
			return threshold - Math.log(1 / output - 1);
		}
	};
	public static class SuperSigmoidal implements ActivationFunction {
		private final double threshold;
		public SuperSigmoidal(double threshold) {
			this.threshold = threshold;
		}
		@Override
		public double feed(double input) {
			return 2 / (1 + Math.exp(threshold - input)) - 1;
		}
		@Override
		public double derivative(double activation) {
			return (1 + activation) * (1 - activation) / 2;
		}
		@Override
		public double backfeed(double output) {
			// TODO double-check?
			return threshold - Math.log(2 / (output + 1) - 1);
		}
	};
	public static class Gaussian implements ActivationFunction {
		private final double center;
		private final double sharpness;
		public Gaussian(double center, double sharpness) {
			this.center = center;
			this.sharpness = sharpness;
		}
		@Override
		public double feed(double input) {
			final double u = sharpness * (input - center);
			return Math.exp(-u*u);
		}
		@Override
		public double derivative(double activation) {
			throw new IllegalStateException("can't calculate from just activation");
		}
		@Override
		public double backfeed(double output) {
			return Math.sqrt(-Math.log(output)) / sharpness + center;
		}
		public double[] backfeed2(double output) {
			final double sqrt = Math.sqrt(-Math.log(output));
			return new double[] {sqrt / sharpness + center, -sqrt / sharpness + center};
		}
		public double getCenter() {
			return center;
		}
	};

	public static final ActivationFunction SUPERSIGMOID = new SuperSigmoidal(0.5);
	
	public static final ActivationFunction SIGMOID0p5 = new Sigmoidal(0.5);

	public static final ActivationFunction LINEAR = new ActivationFunction() {
		@Override
		public double feed(double input) {
			return input;
		}
		@Override
		public double derivative(double activation) {
			return 1;
		}
		@Override
		public double backfeed(double output) {
			return output;
		}
	};
	public static final ActivationFunction RECTIFIER = new ActivationFunction() {
		@Override
		public double feed(double input) {
			return Math.max(0, input);
		}
		@Override
		public double derivative(double activation) {
			return activation < 0 ? 0 : 1;
		}
		@Override
		public double backfeed(double output) {
			throw new IllegalStateException("can't calculate exact input");
		}
	};
}
