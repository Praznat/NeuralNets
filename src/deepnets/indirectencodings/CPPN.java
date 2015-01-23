package deepnets.indirectencodings;

import deepnets.*;

public class CPPN {

	private double[] nodeToInputValue(Node node) {
		return new double[] {node.hashCode(), Math.random()}; // TODO PLACE IN ANN
	}
	
	public double calculateConnectionWeight(Connection c) {
		return doit(nodeToInputValue(c.getInputNode()), nodeToInputValue(c.getOutputNode()));
	}

	private double doit(double[] nodeToInputValue, double[] nodeToInputValue2) {
		// TODO run through 4-input CPPN
		return 0;
	}
}
