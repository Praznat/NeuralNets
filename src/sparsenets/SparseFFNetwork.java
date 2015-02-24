package sparsenets;

import deepnets.*;

public class SparseFFNetwork extends FFNeuralNetwork {

	private final int maxConnsIO;
	private final Node.Factory<? extends Node> sparseNodeFactory;

	public SparseFFNetwork(ActivationFunction actFn, int numInputs,
			int numOutputs, int[] numHidden, final int maxConnsIO) {
		super(actFn, numInputs, numOutputs, numHidden);
		this.maxConnsIO = maxConnsIO;
		sparseNodeFactory = new Node.Factory<Node>() {
			@Override
			public Node create(ActivationFunction activationFunction,
					Layer<? extends Node> parentLayer, String nodeInLayer) {
				return new SparseNode(activationFunction, parentLayer, nodeInLayer, maxConnsIO);
			}
		};
	}

	@Override
	protected Node.Factory<? extends Node> getNodeFactory() {
		return sparseNodeFactory;
	}

}
