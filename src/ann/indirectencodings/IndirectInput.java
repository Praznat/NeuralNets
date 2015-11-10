package ann.indirectencodings;

import java.io.Serializable;

@SuppressWarnings("serial")
public class IndirectInput implements Serializable {

	private final String name;
	private final double[] vector;

	public IndirectInput(String name, int onK, int len) {
		this(name, createRepVector(onK, len));
	}
	
	public IndirectInput(String name, double... vector) {
		this.name = name;
		this.vector = vector;
	}
	
	private static double[] createRepVector(int onK, int len) {
		double[] result = new double[len];
		result[onK] = 1;
		return result;
	}
	
	@Override
	public String toString() {
		return name;
	}

	public double[] getVector() {
		return vector;
	}
}
