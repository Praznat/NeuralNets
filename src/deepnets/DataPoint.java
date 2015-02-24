package deepnets;

import java.util.*;

public class DataPoint {
	private final double[] input;
	private final double[] output;
	
	public DataPoint(double[] input, double[] output) {
		this.input = input;
		this.output = output;
	}

	public double[] getInput() {
		return input;
	}

	public double[] getOutput() {
		return output;
	}
	
	public static Collection<DataPoint> createData(double[][] inputs, double[][] outputs) {
		int n = Math.min(inputs.length, outputs.length);
		Collection<DataPoint> result = new ArrayList<DataPoint>();
		for (int i = 0; i < n; i++) {
			result.add(new DataPoint(inputs[i], outputs[i]));
		}
		return result;
	}

	public static DataPoint create(double[] inputActivations, double[] outputTargets) {
		return new DataPoint(inputActivations.clone(), outputTargets.clone());
	}
	
	@Override
	public String toString() {
		String s = "I: ";
		for (double d : input) s += d+",";
		s += " O: ";
		for (double d : output) s += d+",";
		return s;
	}

}
