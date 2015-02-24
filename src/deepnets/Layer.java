package deepnets;

import java.util.*;

import deepnets.Node.Factory;

public class Layer<N extends Node> {
	private final ArrayList<N> nodes = new ArrayList<N>();
	
	public static Layer<Node> create(int n, ActivationFunction actFn, Factory<? extends Node> nodeFactory) {
		Layer<Node> result = new Layer<Node>();
		for (int i = 0; i < n; i++) result.addNode(nodeFactory.create(actFn, result, String.valueOf(i)));
		return result;
	}
	public static Layer<Node> createInputLayer(int n, Node.Factory<? extends Node> nodeFactory) {
		return create(n, ActivationFunction.LINEAR, nodeFactory);
	}

	public static Layer<Node> createHiddenFromInputLayer(Collection<? extends Node> nodes, int n,
			ActivationFunction actFn, Node.Factory<? extends Node> nodeFactory) {
		Layer<Node> result = create(n, actFn, nodeFactory);
		for (Node inputNode : nodes) {
			for (Node outputNode : result.getNodes()) {
				Connection.getOrCreate(inputNode, outputNode);
			}
		}
		return result;
	}
	public static Layer<Node> createHiddenFromInputLayer(Layer<? extends Node> inputLayer, int n,
			ActivationFunction actFn, Node.Factory<? extends Node> nodeFactory) {
		return createHiddenFromInputLayer(inputLayer.getNodes(), n, actFn, nodeFactory);
	}

	@SuppressWarnings("unchecked")
	protected void addNode(Node n) {
		getNodes().add((N) n);
	}

	public ArrayList<N> getNodes() {
		return nodes;
	}
	public void clamp(double... inputs) {
		int i = 0;
		for (Node n : nodes) {
			if (i >= nodes.size()) break;
			n.clamp(inputs[i++]);
		}
	}
	public void activate() {
		for (Node n : nodes) n.activate();
	}
	
}
