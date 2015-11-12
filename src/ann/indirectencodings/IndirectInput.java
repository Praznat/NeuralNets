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
	public int hashCode() {
		return name.hashCode();
	}
	
	@Override
	public boolean equals(Object other) {
		if (this == other) return true;
		if (other instanceof IndirectInput) {
			return this.name.equals(((IndirectInput)other).name);
		}
		return false;
	}
	
	@Override
	public String toString() {
		return name;
	}

	public double[] getVector() {
		return vector;
	}
}
