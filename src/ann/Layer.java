package ann;

import java.io.Serializable;
import java.util.*;

import ann.Node.Factory;

@SuppressWarnings("serial")
public class Layer<N extends Node> implements Serializable {
	private final ArrayList<N> nodes = new ArrayList<N>();
	private String name = "";
	
	public static Layer<Node> create(int n, ActivationFunction actFn, Factory<? extends Node> nodeFactory) {
		Layer<Node> result = new Layer<Node>();
		for (int i = 0; i < n; i++) result.addNode(nodeFactory.create(actFn, result, String.valueOf(i)));
		return result;
	}
	public static Layer<Node> createInputLayer(int n, Node.Factory<? extends Node> nodeFactory) {
		return create(n, ActivationFunction.LINEAR, nodeFactory);
	}

	public static void fullyConnect(Collection<? extends Node> inNodes, Collection<? extends Node> outNodes) {
		for (Node inputNode : inNodes) {
			for (Node outputNode : outNodes) {
				Connection.getOrCreate(inputNode, outputNode);
			}
		}
	}
	public static Layer<Node> createHiddenFromInputLayer(Collection<? extends Node> nodes, int n,
			ActivationFunction actFn, Node.Factory<? extends Node> nodeFactory) {
		Layer<Node> result = create(n, actFn, nodeFactory);
		fullyConnect(nodes, result.getNodes());
		return result;
	}
	public static Layer<Node> createHiddenFromInputLayer(Layer<? extends Node> inputLayer, int n,
			ActivationFunction actFn, Node.Factory<? extends Node> nodeFactory) {
		return createHiddenFromInputLayer(inputLayer.getNodes(), n, actFn, nodeFactory);
	}

	@SuppressWarnings("unchecked")
	public void addNode(Node n) {
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
	
	public void setName(String name) {
		this.name = name;
	}
	
	@Override
	public String toString() {
		return name;
	}
	
}
