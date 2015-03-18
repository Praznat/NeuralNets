package sparsenets;

import deepnets.*;

public class SparseFFNetwork extends FFNeuralNetwork {

	private static Node.Factory<? extends Node> createSparseNodeFactory(final int maxConns) {
		return new Node.Factory<Node>() {
			@Override
			public Node create(ActivationFunction activationFunction,
					Layer<? extends Node> parentLayer, String nodeInLayer) {
				return new SparseNode(activationFunction, parentLayer, nodeInLayer, maxConns);
			}
		};
	};

	public SparseFFNetwork(ActivationFunction actFn, int numInputs,
			int numOutputs, int[] numHidden, final int maxConnsIO) {
		super(actFn, numInputs, numOutputs, createSparseNodeFactory(maxConnsIO), numHidden);
	}


}
