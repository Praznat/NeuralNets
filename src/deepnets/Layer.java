package deepnets;

import java.util.*;

public class Layer<N extends Node> {
	private final Collection<N> nodes = new ArrayList<N>();
	
	public static Layer<Node> create(int n, ActivationFunction actFn) {
		Layer<Node> result = new Layer<Node>();
		for (int i = 0; i < n; i++) result.addNode(new Node(actFn, result, String.valueOf(i)));
		return result;
	}
	public static Layer<Node> createInputLayer(int n) {
		return create(n, ActivationFunction.LINEAR);
	}

	public static Layer<Node> createHiddenFromInputLayer(Collection<? extends Node> nodes, int n,
			ActivationFunction actFn) {
		Layer<Node> result = create(n, actFn);
		for (Node inputNode : nodes) {
			for (Node outputNode : result.getNodes()) {
				Connection.getOrCreate(inputNode, outputNode);
			}
		}
		return result;
	}
	public static Layer<Node> createHiddenFromInputLayer(Layer<? extends Node> inputLayer, int n,
			ActivationFunction actFn) {
		return createHiddenFromInputLayer(inputLayer.getNodes(), n, actFn);
	}

	protected void addNode(N n) {
		getNodes().add(n);
	}

	public Collection<N> getNodes() {
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
